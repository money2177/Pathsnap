"""Stage 4: train and compare candidate classifiers.

Protocol
  * Training data: reliable train trips (78), points whose label is within R of the GPS point.
    Every 3rd GPS point of each trip is used (consecutive 1 Hz points are near-duplicates),
    keeping all candidates of each used point.
  * 5-fold GroupKFold by trip on that data -> CV metrics (no trip spans train and validation).
  * Refit on all training data -> evaluate on all 10 student_test trips (all points).
  * Recorded per model: configuration, fold metrics, test metrics, confusion matrix,
    training time, inference time, epoch/iteration histories where the algorithm has them.

Run: python scripts/train_compare.py [--models LogisticRegression MLP ...]
"""
import argparse, copy, json, os, platform, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from src.data_io import REPO_ROOT
from src.evaluation import all_metrics, confusion
from src.features import FEATURES
from src.models import MODELS, SEED, sample_points

FEAT = os.path.join(REPO_ROOT, "outputs", "features")
OUT = os.path.join(REPO_ROOT, "outputs", "models", "compare")
POINT_STRIDE = 3
N_FOLDS = 5
MLP_MAX_EPOCHS, MLP_PATIENCE = 40, 5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_train():
    """(fit_df, eval_df): fit_df = every 3rd point with a usable label (for fitting);
    eval_df = every point of the reliable trips (for scoring, same protocol as student_test)."""
    full = pd.read_parquet(f"{FEAT}/train.parquet")
    full = full[full.trip_reliable == 1].reset_index(drop=True)
    fit_df = full[(full.label_ok == 1) & (full.point_idx % POINT_STRIDE == 0)].reset_index(drop=True)
    return fit_df, full


def fit_mlp(spec, X, y, groups):
    """Epoch-wise training with a trip-grouped 10% validation split; keeps the best epoch."""
    pipe = spec["build"]()
    scaler, mlp = pipe.steps[0][1], pipe.steps[1][1]
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(X, y, groups))
    scaler.fit(X[tr]); Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    hist, best, best_loss, bad = [], None, np.inf, 0
    for ep in range(1, MLP_MAX_EPOCHS + 1):
        mlp.partial_fit(Xtr, y[tr], classes=[0, 1])
        tl = log_loss(y[tr], np.clip(mlp.predict_proba(Xtr)[:, 1], 1e-7, 1 - 1e-7))
        vl = log_loss(y[va], np.clip(mlp.predict_proba(Xva)[:, 1], 1e-7, 1 - 1e-7))
        hist.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
        if vl < best_loss - 1e-5:
            best_loss, best, bad = vl, copy.deepcopy(mlp), 0
        else:
            bad += 1
            if bad >= MLP_PATIENCE:
                break
    return Pipeline([("scaler", scaler), ("mlp", best)]), hist


def fit_lgbm(spec, X, y, groups):
    """LightGBM with early stopping on a trip-grouped 10% split; predicts with the best round."""
    import lightgbm as lgb
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(X, y, groups))
    model = spec["build"]()
    model.fit(X[tr], y[tr], eval_set=[(X[tr], y[tr]), (X[va], y[va])], eval_names=["train", "val"],
              eval_metric="binary_logloss", callbacks=[lgb.early_stopping(50, verbose=False)])
    ev = model.evals_result_
    hist = [{"iteration": i + 1, "train_loss": a, "val_loss": b}
            for i, (a, b) in enumerate(zip(ev["train"]["binary_logloss"], ev["val"]["binary_logloss"]))]
    return model, hist


def fit(name, df):
    spec = MODELS[name]
    d = sample_points(df, spec["max_train_rows"])
    X, y = d[FEATURES].to_numpy(np.float32), d.y.to_numpy()
    t0 = time.time()
    hist = None
    if spec.get("epochs"):
        model, hist = fit_mlp(spec, X, y, d.trip.to_numpy())
    elif spec.get("lgbm"):
        model, hist = fit_lgbm(spec, X, y, d.trip.to_numpy())
    else:
        model = spec["build"]().fit(X, y)
        if name == "HistGradientBoosting":
            hist = [{"iteration": i, "train_loss": -a, "val_loss": -b}
                    for i, (a, b) in enumerate(zip(model.train_score_, model.validation_score_))]
    return model, time.time() - t0, len(d), hist


def predict(model, df):
    t0 = time.time()
    p = model.predict_proba(df[FEATURES].to_numpy(np.float32))[:, 1]
    return p, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    train, train_eval = load_train()
    test = pd.read_parquet(f"{FEAT}/student_test.parquet")
    part_train = {t: "train" for t in train.trip.unique()}
    part_test = {t: "student_test" for t in test.trip.unique()}
    log(f"train rows {len(train):,} ({train.trip.nunique()} trips, {train.groupby(['trip','point_idx']).ngroups:,} points); "
        f"student_test rows {len(test):,}")

    folds = list(GroupKFold(n_splits=N_FOLDS).split(train, train.y, train.trip))
    preds_path = f"{OUT}/student_test_predictions.parquet"
    preds = pd.read_parquet(preds_path) if os.path.exists(preds_path) else test[["trip", "point_idx", "arc_id", "y"]].copy()

    for name in args.models:
        spec = MODELS[name]
        log(f"=== {name} ===")
        fold_rows = []
        for k, (tr, va) in enumerate(folds):
            model, ttrain, ntr, _ = fit(name, train.iloc[tr])
            val = train_eval[train_eval.trip.isin(train.trip.iloc[va].unique())]
            p, tinf = predict(model, val)
            m = all_metrics(val, p, part_train)
            fold_rows.append(dict(model=name, fold=k, n_train_rows=ntr, train_time_s=ttrain, **m))
            log(f"  fold {k}: top1={m['top1_accuracy']:.4f} macroF1={m['macro_f1']:.4f} logloss={m['log_loss']:.4f} "
                f"cont={m['route_continuity']:.4f} ({ttrain:.1f}s)")
        # training-fold score on the last fold, for over/underfitting evidence
        trn = train_eval[train_eval.trip.isin(train.trip.iloc[tr].unique())]
        p_tr, _ = predict(model, trn)
        m_tr = all_metrics(trn, p_tr)

        model, ttrain, ntr, hist = fit(name, train)
        p, tinf = predict(model, test)
        m = all_metrics(test, p, part_test)
        cm = confusion(test, p)
        preds[name] = p.astype(np.float32)
        log(f"  TEST: top1={m['top1_accuracy']:.4f} macroF1={m['macro_f1']:.4f} logloss={m['log_loss']:.4f} "
            f"cont={m['route_continuity']:.4f} train={ttrain:.1f}s infer={tinf:.2f}s")

        res = dict(model=name, family=spec["family"], config=spec["config"], loss=spec["loss"],
                   n_train_rows=ntr, train_time_s=ttrain, inference_time_s=tinf,
                   inference_ms_per_1k_rows=1000 * tinf / len(test) * 1000,
                   confusion_matrix=cm.tolist(), history=hist,
                   last_fold_train_metrics=m_tr, test=m)
        json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=2, default=float)
        pd.DataFrame(fold_rows).to_csv(f"{OUT}/{name}_cv.csv", index=False)
        joblib.dump(model, f"{OUT}/{name}.joblib", compress=3)
        preds.to_parquet(preds_path, index=False)

    json.dump(dict(seed=SEED, point_stride=POINT_STRIDE, n_folds=N_FOLDS, cv="GroupKFold by trip",
                   training_filter="fit: trip_reliable==1 & label_ok==1 & point_idx%3==0; CV scoring: all points of held-out reliable trips", features=FEATURES,
                   python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                   machine=platform.platform()),
              open(f"{OUT}/experiment_config.json", "w"), indent=2)
    log("done")
