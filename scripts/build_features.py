"""Build the (GPS point, candidate arc) feature tables for train and student_test.

Outputs (outputs/features/):
  train.parquet, student_test.parquet   one row per candidate, FEATURES + label y
  trip_summary.csv                      per-trip label quality and candidate miss rate
  config.json                           parameters used (needed again at inference time)

Run: python scripts/build_features.py [--radius 50] [--k 12]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.data_io import Trip, list_trips, REPO_ROOT
from src.features import build_trip_features, FEATURES, SIGMA_M, MIN_MOVE_M

OUT = os.path.join(REPO_ROOT, "outputs", "features")
UNRELIABLE_TRIP_MEDIAN_M = 30.0   # trip-level: median point->route distance above this means unreliable labels


def one(tid, part, radius, k):
    t0 = time.time()
    f = build_trip_features(Trip(tid, part), radius_m=radius, k=k)
    n_pts = f.point_idx.nunique()
    hit = f.groupby("point_idx").y.max()
    med = f.groupby("point_idx").label_dist.first().median()
    f["trip_reliable"] = np.int8(med <= UNRELIABLE_TRIP_MEDIAN_M)
    f["label_ok"] = (f.label_dist <= radius).astype(np.int8)
    s = dict(trip=tid, partition=part, n_points=n_pts, n_rows=len(f), cands_per_point=len(f) / n_pts,
             candidate_miss_rate=1 - hit.mean(), median_label_dist=med,
             pct_points_label_far=float((f.groupby("point_idx").label_dist.first() > radius).mean() * 100),
             trip_reliable=bool(med <= UNRELIABLE_TRIP_MEDIAN_M), seconds=time.time() - t0)
    return f, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summaries = []
    for part in ["train", "student_test"]:
        res = Parallel(n_jobs=-1)(delayed(one)(t, part, a.radius, a.k) for t in list_trips(part))
        df = pd.concat([r[0] for r in res], ignore_index=True)
        df.to_parquet(f"{OUT}/{part}.parquet", index=False)
        summaries += [r[1] for r in res]
        print(f"{part}: {df.trip.nunique()} trips, {df.groupby(['trip','point_idx']).ngroups} points, "
              f"{len(df)} rows, positive rate {df.y.mean():.3f}")
    s = pd.DataFrame(summaries)
    s.to_csv(f"{OUT}/trip_summary.csv", index=False)
    json.dump(dict(radius_m=a.radius, k=a.k, k_fallback=3, sigma_m=SIGMA_M, min_move_m=MIN_MOVE_M,
                   unreliable_trip_median_m=UNRELIABLE_TRIP_MEDIAN_M, features=FEATURES),
              open(f"{OUT}/config.json", "w"), indent=2)
    print("\nCandidate miss rate (point-weighted):")
    for (part, rel), g in s.groupby(["partition", "trip_reliable"]):
        mr = np.average(g.candidate_miss_rate, weights=g.n_points)
        print(f"  {part:13s} reliable={rel!s:5s} trips={len(g):2d} miss={mr:.2%}")
    print("Unreliable trips:", s.loc[~s.trip_reliable, ["trip", "partition", "median_label_dist"]].to_dict("records"))
