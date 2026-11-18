"""Refit the tuned HistGradientBoosting with the stage-6 best hyperparameters on the current features,
without repeating the Bayesian search.

Used after the candidate tie-break fix (candidates sorted by distance, then arc_id), which only changes
the order and dist_rank of exactly tied twin arcs. It rewrites the refit/test fields of
tuning_result_stage2.json and keeps the search record (trials, search time, CV values of the search)
unchanged. The current CV of these hyperparameters is recomputed in notebooks/04_tuning.ipynb.

Run: python scripts/refit_tuned.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib, numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from src.data_io import REPO_ROOT
from src.evaluation import all_metrics
from src.features import FEATURES
from src.models import SEED
from train_compare import load_train, log
from tune_hgb import make_model

OUT = os.path.join(REPO_ROOT, "outputs", "models", "tuning")
RES = f"{OUT}/tuning_result_stage2.json"

if __name__ == "__main__":
    res = json.load(open(RES))
    best = res["best_params"]
    train, _ = load_train()
    X, y = train[FEATURES].to_numpy(np.float32), train.y.to_numpy()
    test = pd.read_parquet(os.path.join(REPO_ROOT, "outputs", "features", "student_test.parquet"))
    t0 = time.time(); model = make_model(best).fit(X, y); fit_s = time.time() - t0
    t0 = time.time(); p = model.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]; inf_s = time.time() - t0
    m = all_metrics(test, p, {t: "student_test" for t in test.trip.unique()})
    log(f"TEST tuned (refit): {m}")
    joblib.dump(model, f"{OUT}/hgb_tuned_stage2.joblib", compress=3)
    pred_path = os.path.join(REPO_ROOT, "outputs", "models", "compare", "student_test_predictions.parquet")
    preds = pd.read_parquet(pred_path)
    assert (preds[["trip", "point_idx", "arc_id"]].to_numpy() == test[["trip", "point_idx", "arc_id"]].to_numpy()).all()
    preds["HistGradientBoosting_tuned_stage2"] = p.astype(np.float32)
    preds.to_parquet(pred_path, index=False)

    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(X, y, train.trip))
    hm = make_model(best).fit(X[tr], y[tr])
    hist = [{"iteration": i + 1,
             "train_loss": log_loss(y[tr], np.clip(a[:, 1], 1e-7, 1 - 1e-7)),
             "val_loss": log_loss(y[va], np.clip(b[:, 1], 1e-7, 1 - 1e-7))}
            for i, (a, b) in enumerate(zip(hm.staged_predict_proba(X[tr]), hm.staged_predict_proba(X[va])))
            if i % 5 == 4 or i == 0]
    res.update(refit_seconds=fit_s, inference_seconds=inf_s, inference_ms_per_1k_rows=1e6 * inf_s / len(test),
               test=m, history=hist,
               refit_note="refit with the same best_params after the candidate tie-break fix (distance, then arc_id); "
                          "search fields are from the original search")
    json.dump(res, open(RES, "w"), indent=2, default=float)
    log("done")
