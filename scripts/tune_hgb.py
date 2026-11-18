"""Stage 6: Bayesian hyperparameter optimisation of HistGradientBoosting (Optuna TPE).

Objective      maximise mean 5-fold GroupKFold (by trip) Top-1 accuracy on the 78 reliable
               train trips, using exactly the folds and protocol of scripts/train_compare.py
               (fit on every 3rd labelled point, score every point of the held-out trips).
Sampler        TPE (Tree-structured Parzen Estimator), seed 42, 10 random start-up trials.
Pruner         MedianPruner: after each fold, a trial whose running mean is below the median
               of earlier trials at the same fold is stopped (saves compute on bad regions).
Anchor         trial 0 is the untuned configuration from stage 4 (enqueued), so the search
               starts from the known baseline.
The student_test partition is used only once, after the search, to evaluate the refit model.

Run: python scripts/tune_hgb.py [--trials 60] [--timeout 2700]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib, numpy as np, optuna, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from src.data_io import REPO_ROOT
from src.evaluation import all_metrics, top1_accuracy
from src.features import FEATURES
from src.models import SEED
from train_compare import N_FOLDS, load_train, log

OUT = os.path.join(REPO_ROOT, "outputs", "models", "tuning")
STUDY = "hgb_top1"

SPACE = {  # documented search space (also written to search_space.json)
    "learning_rate":     ("log-uniform", 0.01, 0.3, "step size; untuned 0.1 showed loss spikes"),
    "max_iter":          ("int, step 50", 100, 800, "number of boosting rounds (early stopping off, so it is tuned directly)"),
    "max_leaf_nodes":    ("log-int", 15, 255, "tree complexity"),
    "min_samples_leaf":  ("log-int", 10, 1000, "smoothing; data has 340k rows per fold"),
    "l2_regularization": ("log-uniform", 1e-3, 30.0, "shrinks leaf values, which counters the overshooting seen in stage 4"),
    "max_features":      ("uniform", 0.3, 1.0, "feature subsampling per split (decorrelates trees)"),
}


STUDY2 = "hgb_top1_stage2"
SPACE2 = {  # stage 2: refined around the stage-1 optimum; max_iter widened because it hit its 800 ceiling
    "learning_rate":     ("log-uniform", 0.01, 0.15, "stage-1 top trials used 0.02-0.08"),
    "max_iter":          ("int, step 100", 300, 2000, "stage-1 best trials sat at the 800 upper bound"),
    "max_leaf_nodes":    ("log-int", 31, 255, "stage-1 top trials used 64-151"),
    "min_samples_leaf":  ("log-int", 20, 300, "stage-1 top trials used 39-122"),
    "l2_regularization": ("log-uniform", 1e-3, 3.0, "stage-1 top trials used 0.008-0.58"),
    "max_features":      ("uniform", 0.45, 0.9, "stage-1 top trials used 0.59-0.80"),
}


def suggest2(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_iter": trial.suggest_int("max_iter", 300, 2000, step=100),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 255, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 300, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 3.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.45, 0.9),
    }


def make_model(p):
    return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **p)


def suggest(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 1000, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 30.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=2700)
    ap.add_argument("--stage2", action="store_true", help="refined search seeded with the stage-1 top 3 trials")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.stage2:
        STUDY, SPACE, suggest, tag = STUDY2, SPACE2, suggest2, "_stage2"
    else:
        tag = ""

    train, train_eval = load_train()
    X, y = train[FEATURES].to_numpy(np.float32), train.y.to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(train, y, train.trip))
    val_sets = [train_eval[train_eval.trip.isin(train.trip.iloc[va].unique())] for _, va in folds]
    val_X = [v[FEATURES].to_numpy(np.float32) for v in val_sets]

    def objective(trial):
        p = suggest(trial)
        scores, lls, t0 = [], [], time.time()
        for k, (tr, _) in enumerate(folds):
            m = make_model(p).fit(X[tr], y[tr])
            prob = m.predict_proba(val_X[k])[:, 1]
            scores.append(top1_accuracy(val_sets[k], prob))
            lls.append(log_loss(val_sets[k].y, np.clip(prob, 1e-7, 1 - 1e-7)))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                trial.set_user_attr("folds_done", k + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", scores)
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(scores)))
        trial.set_user_attr("seconds", time.time() - t0)
        log(f"trial {trial.number}: top1={np.mean(scores):.4f} ll={np.mean(lls):.4f} ({time.time()-t0:.0f}s) {p}")
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=STUDY, direction="maximize", storage=f"sqlite:///{OUT}/optuna.db", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4 if args.stage2 else 10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5 if args.stage2 else 8, n_warmup_steps=1))
    if len(study.trials) == 0 and not args.stage2:  # anchor: the untuned stage-4 configuration
        study.enqueue_trial({"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31,
                             "min_samples_leaf": 20, "l2_regularization": 1e-3, "max_features": 1.0})
    if len(study.trials) == 0 and args.stage2:  # anchors: stage-1 top 3 (re-evaluated, same folds)
        s1 = optuna.load_study(study_name="hgb_top1", storage=f"sqlite:///{OUT}/optuna.db")
        top = sorted([t for t in s1.trials if t.state.name == "COMPLETE"], key=lambda t: -t.value)[:3]
        for t in top:
            study.enqueue_trial(t.params)
    t_search = time.time()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    search_s = time.time() - t_search
    best = study.best_params
    log(f"BEST cv top1={study.best_value:.4f} params={best} (search {search_s/60:.1f} min)")

    # refit on all fitting data, evaluate once on student_test
    test = pd.read_parquet(os.path.join(REPO_ROOT, "outputs", "features", "student_test.parquet"))
    t0 = time.time(); model = make_model(best).fit(X, y); fit_s = time.time() - t0
    t0 = time.time(); p = model.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]; inf_s = time.time() - t0
    m = all_metrics(test, p, {t: "student_test" for t in test.trip.unique()})
    log(f"TEST tuned: {m}")
    joblib.dump(model, f"{OUT}/hgb_tuned{tag}.joblib", compress=3)
    pred_path = os.path.join(REPO_ROOT, "outputs", "models", "compare", "student_test_predictions.parquet")
    preds = pd.read_parquet(pred_path); preds[f"HistGradientBoosting_tuned{tag}"] = p.astype(np.float32)
    preds.to_parquet(pred_path, index=False)

    # loss vs boosting iteration for the tuned configuration (10% of train trips held out)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(X, y, train.trip))
    hm = make_model(best).fit(X[tr], y[tr])
    hist = [{"iteration": i + 1,
             "train_loss": log_loss(y[tr], np.clip(a[:, 1], 1e-7, 1 - 1e-7)),
             "val_loss": log_loss(y[va], np.clip(b[:, 1], 1e-7, 1 - 1e-7))}
            for i, (a, b) in enumerate(zip(hm.staged_predict_proba(X[tr]), hm.staged_predict_proba(X[va])))
            if i % 5 == 4 or i == 0]

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials.to_csv(f"{OUT}/trials{tag}.csv", index=False)
    json.dump({k: dict(zip(["distribution", "low", "high", "why"], v)) for k, v in SPACE.items()},
              open(f"{OUT}/search_space{tag}.json", "w"), indent=2)
    bt = study.best_trial
    json.dump(dict(stage=("2 (refined)" if args.stage2 else "1"), study=STUDY, objective="mean 5-fold GroupKFold Top-1 accuracy (maximise)", sampler=f"TPE(seed=42, {4 if args.stage2 else 10} startup)",
                   pruner=f"MedianPruner({5 if args.stage2 else 8} startup, 1 warm-up fold)",
                   n_trials=len(study.trials),
                   n_complete=sum(t.state.name == "COMPLETE" for t in study.trials),
                   n_pruned=sum(t.state.name == "PRUNED" for t in study.trials),
                   search_seconds=search_s, best_params=best, best_cv_top1=study.best_value,
                   best_cv_top1_std=bt.user_attrs.get("cv_top1_std"), best_cv_log_loss=bt.user_attrs.get("cv_log_loss"),
                   best_fold_top1=bt.user_attrs.get("fold_top1"),
                   refit_seconds=fit_s, inference_seconds=inf_s,
                   inference_ms_per_1k_rows=1e6 * inf_s / len(test), test=m, history=hist),
              open(f"{OUT}/tuning_result{tag}.json", "w"), indent=2, default=float)
    log("done")
