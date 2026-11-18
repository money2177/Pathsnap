"""Observation-level (per GPS point) metrics for every model on student_test.

The row-level metrics in stage 4 (macro F1, precision, recall, log loss) treat each
(GPS point, candidate arc) pair as one binary example. This script also reports
metrics per GPS observation, which is the unit the task is judged on:

  top1_accuracy          the highest-scoring candidate is the reference arc (all points)
  obs_log_loss           -log of the probability given to the reference arc after normalising the
                         point's candidate probabilities to sum to 1 (a softmax-free renormalisation).
                         Computed on points whose reference arc is among the candidates
                         (the remaining ~0.2% are candidate misses and have no defined value).
  obs_top3_accuracy      the reference arc is among the 3 highest-scoring candidates
  mean_true_prob         average normalised probability of the reference arc

Output: outputs/models/compare/observation_level_metrics.csv
Run:    python scripts/observation_metrics.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from src.data_io import REPO_ROOT

OUT = os.path.join(REPO_ROOT, "outputs", "models", "compare")
MODELS = ["LogisticRegression", "GaussianNB", "KNN", "SVM_RBF", "RandomForest", "HistGradientBoosting",
          "LightGBM", "MLP", "HistGradientBoosting_tuned_stage2"]

if __name__ == "__main__":
    p = pd.read_parquet(f"{OUT}/student_test_predictions.parquet")
    key = p.trip + "/" + p.point_idx.astype(str)
    has_true = p.groupby(key).y.transform("max").to_numpy() == 1
    rows = []
    for m in MODELS:
        s = np.clip(p[m].to_numpy(np.float64), 1e-7, None)
        norm = s / pd.Series(s).groupby(key.to_numpy()).transform("sum").to_numpy()
        rank = pd.Series(-p[m].to_numpy(np.float64)).groupby(key.to_numpy()).rank(method="first").to_numpy()  # raw scores: clipping would create ties
        t = (p.y.to_numpy() == 1)
        tp = norm[t & has_true]
        n_pts = key.nunique()
        rows.append(dict(model=m, n_points=n_pts, n_points_scored_log_loss=int(t.sum()),
                         top1_accuracy=float(((rank == 1) & t).sum() / n_pts),
                         obs_top3_accuracy=float(((rank <= 3) & t).sum() / n_pts),
                         obs_log_loss=float(-np.log(np.clip(tp, 1e-15, 1)).mean()),
                         mean_true_prob=float(tp.mean())))
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(f"{OUT}/observation_level_metrics.csv")
    print(df.round(4).to_string())
