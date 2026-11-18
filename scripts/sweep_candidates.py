"""Candidate-generation sweep on the TRAIN partition: miss rate vs (radius, K).

Candidate Miss Rate = share of GPS points whose true arc is not in their candidate set.
Run: python scripts/sweep_candidates.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.data_io import Trip, list_trips, REPO_ROOT
from src.candidates import TripGraph, generate_candidates
from src.labels import align_points_to_route

RADII = [15, 20, 30, 40, 50, 75, 100]
KS = [2, 4, 6, 8, 12, 16, 1000]


def one(tid):
    t = Trip(tid, "train"); g = TripGraph(t)
    lab = align_points_to_route(t)
    c = generate_candidates(g, radius_m=max(RADII), k=max(KS))
    tc = g.canonical[lab["true_arc"]]
    c["y"] = g.canonical[c.arc_id.values] == tc[c.point_idx.values]
    c = c[~c.fallback]
    out = []
    n = len(t.track)
    for R in RADII:
        for K in KS:
            s = c[(c.dist <= R) & (c.dist_rank < K)]
            hit = np.zeros(n, bool); hit[s.point_idx[s.y].values] = True
            out.append(dict(trip=tid, R=R, K=K, n_points=n, hits=hit.sum(), n_cands=len(s),
                            median_label_dist=np.median(lab["label_dist"])))
    return out


if __name__ == "__main__":
    rows = sum(Parallel(n_jobs=-1)(delayed(one)(t) for t in list_trips("train")), [])
    df = pd.DataFrame(rows)
    df["reliable_trip"] = df.median_label_dist <= 30
    os.makedirs(f"{REPO_ROOT}/outputs/features", exist_ok=True)
    df.to_csv(f"{REPO_ROOT}/outputs/features/candidate_sweep_train.csv", index=False)
    d = df[df.reliable_trip].groupby(["R", "K"])[["hits", "n_points", "n_cands"]].sum()
    d["miss_rate_%"] = (1 - d.hits / d.n_points) * 100
    d["cands_per_point"] = d.n_cands / d.n_points
    print("excluded trips:", sorted(df.loc[~df.reliable_trip, "trip"].unique()))
    print(d[["miss_rate_%", "cands_per_point"]].round(2).unstack("K").to_string())
