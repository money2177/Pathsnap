"""Stage 6: Bayesian hyperparameter optimisation of LightGBM (Optuna TPE).

Objective      maximise mean 5-fold GroupKFold (by trip) Top-1 accuracy on the 78 reliable
               train trips, using exactly the folds and protocol of scripts/train_compare.py
               (fit on every 3rd labelled point, score every point of the held-out trips).
Rounds         not searched: each fit uses early stopping (patience 50, <=3000 rounds) on a
               trip-grouped 10% split of its own training data, the same as the untuned model.
Sampler        TPE (Tree-structured Parzen Estimator), seed 42, 10 random start-up trials.
Pruner         MedianPruner after each fold (from trial 8, after the first fold).
Anchor         trial 0 is the untuned stage-4 LightGBM configuration.
The student_test partition is used only once, after the search, to evaluate the refit model.

Run: python scripts/tune_lgbm.py [--trials 40] [--timeout 1800]
"""
import argparse, json, os, sys, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import joblib, numpy as np, optuna, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from src.data_io import REPO_ROOT
from src.evaluation import all_metrics, top1_accuracy
from src.features import FEATURES
from src.models import SEED
from train_compare import N_FOLDS, fit_lgbm, load_train, log

OUT = os.path.join(REPO_ROOT, "outputs", "models", "tuning_lgbm")
STUDY = "lgbm_top1"
UNTUNED = {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "subsample": 0.8,
           "colsample_bytree": 0.8, "reg_lambda": 1e-3, "reg_alpha": 1e-3}
SPACE = {  # documented search space (also written to search_space.json)
    "learning_rate":     ("log-uniform", 0.01, 0.2, "step size; rounds adapt through early stopping"),
    "num_leaves":        ("log-int", 15, 255, "tree complexity (leaf-wise growth)"),
    "min_child_samples": ("log-int", 5, 500, "minimum rows per leaf; smoothing and overfitting control"),
    "subsample":         ("uniform", 0.5, 1.0, "row bagging per round; consecutive points are correlated"),
    "colsample_bytree":  ("uniform", 0.4, 1.0, "feature subsampling; many distance features are correlated"),
    "reg_lambda":        ("log-uniform", 1e-3, 30.0, "L2 penalty on leaf values"),
    "reg_alpha":         ("log-uniform", 1e-3, 10.0, "L1 penalty on leaf values"),
}


def spec_for(p):
    return {"build": lambda: LGBMClassifier(n_estimators=3000, subsample_freq=1, random_state=SEED,
                                            n_jobs=-1, verbose=-1, **p)}


def suggest(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    train, train_eval = load_train()
    X, y, g = train[FEATURES].to_numpy(np.float32), train.y.to_numpy(), train.trip.to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(train, y, g))
    val_sets = [train_eval[train_eval.trip.isin(train.trip.iloc[va].unique())] for _, va in folds]
    val_X = [v[FEATURES].to_numpy(np.float32) for v in val_sets]

    def objective(trial):
        p = suggest(trial)
        scores, lls, iters, t0 = [], [], [], time.time()
        for k, (tr, _) in enumerate(folds):
            m, _ = fit_lgbm(spec_for(p), X[tr], y[tr], g[tr])
            prob = m.predict_proba(val_X[k])[:, 1]
            scores.append(top1_accuracy(val_sets[k], prob))
            lls.append(log_loss(val_sets[k].y, np.clip(prob, 1e-7, 1 - 1e-7)))
            iters.append(int(m.best_iteration_ or m.n_estimators))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", scores)
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(scores)))
        trial.set_user_attr("best_iterations", iters)
        trial.set_user_attr("seconds", time.time() - t0)
        log(f"trial {trial.number}: top1={np.mean(scores):.4f} ll={np.mean(lls):.4f} rounds~{int(np.median(iters))} "
            f"({time.time()-t0:.0f}s) {p}")
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=STUDY, direction="maximize", storage=f"sqlite:///{OUT}/optuna.db", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    if len(study.trials) == 0:
        study.enqueue_trial(UNTUNED)
    t_search = time.time()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    search_s = time.time() - t_search
    best = study.best_params
    log(f"BEST cv top1={study.best_value:.4f} params={best} (search {search_s/60:.1f} min)")

    # refit on all fitting data (same early-stopping procedure), evaluate once on student_test
    test = pd.read_parquet(os.path.join(REPO_ROOT, "outputs", "features", "student_test.parquet"))
    t0 = time.time(); model, hist = fit_lgbm(spec_for(best), X, y, g); fit_s = time.time() - t0
    t0 = time.time(); p = model.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]; inf_s = time.time() - t0
    m = all_metrics(test, p, {t: "student_test" for t in test.trip.unique()})
    log(f"TEST tuned: {m}")
    joblib.dump(model, f"{OUT}/lgbm_tuned.joblib", compress=3)
    pred_path = os.path.join(REPO_ROOT, "outputs", "models", "compare", "student_test_predictions.parquet")
    preds = pd.read_parquet(pred_path); preds["LightGBM_tuned"] = p.astype(np.float32)
    preds.to_parquet(pred_path, index=False)

    study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration")) \
         .to_csv(f"{OUT}/trials.csv", index=False)
    json.dump({k: dict(zip(["distribution", "low", "high", "why"], v)) for k, v in SPACE.items()},
              open(f"{OUT}/search_space.json", "w"), indent=2)
    bt = study.best_trial
    json.dump(dict(objective="mean 5-fold GroupKFold Top-1 accuracy (maximise)", sampler="TPE(seed=42, 10 startup)",
                   pruner="MedianPruner(8 startup, 1 warm-up fold)", untuned_params=UNTUNED,
                   n_trials=len(study.trials),
                   n_complete=sum(t.state.name == "COMPLETE" for t in study.trials),
                   n_pruned=sum(t.state.name == "PRUNED" for t in study.trials),
                   search_seconds=search_s, best_params=best, best_trial=bt.number, best_cv_top1=study.best_value,
                   best_cv_top1_std=bt.user_attrs.get("cv_top1_std"), best_cv_log_loss=bt.user_attrs.get("cv_log_loss"),
                   best_fold_top1=bt.user_attrs.get("fold_top1"), final_rounds=int(model.best_iteration_ or 0),
                   refit_seconds=fit_s, inference_seconds=inf_s,
                   inference_ms_per_1k_rows=1e6 * inf_s / len(test), test=m, history=hist),
              open(f"{OUT}/tuning_result.json", "w"), indent=2, default=float)
    log("done")
