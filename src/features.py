"""Feature engineering for (GPS point, candidate arc) pairs.

Each row answers: "is this candidate arc the road the vehicle was on at this GPS point?"
Features only use the GPS track and the road network. The reference route is used only
for the label `y`, so the same code runs unchanged at inference time (API) when no route
is available.

Feature groups
  geometry     dist, dist_rank, dist_minus_min, dist_ratio_min, gauss_share, n_cands,
               proj_frac, dist_to_end_min, arc_len
  direction    heading_diff (0-180, uses arc direction), heading_diff_undirected (0-90),
               heading_reliable
  motion       speed, accel, dt_prev, dt_next, turn_rate
  context      dist_prev1/2, dist_next1/2 (distance from neighbouring GPS points to this
               arc), ctx_mean_dist
  topology     from_out_deg, to_in_deg, to_out_deg, has_reverse
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from .candidates import TripGraph, generate_candidates
from .data_io import Trip
from .labels import align_points_to_route

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min", "gauss_share", "n_cands",
    "proj_frac", "dist_to_end_min", "arc_len",
    "heading_diff", "heading_diff_undirected", "heading_reliable",
    "speed", "accel", "dt_prev", "dt_next", "turn_rate",
    "dist_prev1", "dist_prev2", "dist_next1", "dist_next2", "ctx_mean_dist",
    "from_out_deg", "to_in_deg", "to_out_deg", "has_reverse",
]
SIGMA_M = 5.0          # GPS noise scale for gauss_share (median GPS->route distance in EDA is about 5 m)
MIN_MOVE_M = 2.0       # below this displacement the GPS heading is unreliable


def _ang_diff(a, b):
    d = np.abs(a - b) % 360
    return np.where(d > 180, 360 - d, d)


def point_motion(trip: Trip) -> pd.DataFrame:
    """Point-level motion features from the GPS track alone."""
    xy = trip.track_xy
    t = trip.track.t.to_numpy()
    n = len(xy)
    prv = np.r_[0, np.arange(n - 1)]
    nxt = np.r_[np.arange(1, n), n - 1]
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    # heading by central difference (point i-1 to i+1), smoother than one-sided
    dx, dy = xy[nxt, 0] - xy[prv, 0], xy[nxt, 1] - xy[prv, 1]
    move = np.hypot(dx, dy)
    heading = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    reliable = move >= MIN_MOVE_M
    h = pd.Series(np.where(reliable, heading, np.nan)).ffill().bfill().fillna(0).to_numpy()
    span = np.maximum(t[nxt] - t[prv], 1e-6)
    speed = move / span
    accel = np.r_[0, np.diff(speed)] / np.where(np.isnan(dt_prev), 1, np.maximum(dt_prev, 1e-6))
    turn = _ang_diff(h, np.r_[h[0], h[:-1]]) / np.where(np.isnan(dt_prev), 1, np.maximum(dt_prev, 1e-6))
    return pd.DataFrame({
        "heading": h, "heading_reliable": reliable.astype(np.int8), "speed": speed, "accel": accel,
        "dt_prev": np.nan_to_num(dt_prev, nan=1.0), "dt_next": np.nan_to_num(dt_next, nan=1.0), "turn_rate": turn,
    })


def build_trip_features(trip: Trip, radius_m: float = 50.0, k: int = 12, with_labels: bool = True) -> pd.DataFrame:
    g = TripGraph(trip)
    c = generate_candidates(g, radius_m=radius_m, k=k)
    pi, arc = c.point_idx.to_numpy(), c.arc_id.to_numpy()
    n = len(trip.track)

    # --- geometry ---
    grp = c.groupby("point_idx")["dist"]
    dmin = grp.transform("min").to_numpy()
    c["dist_minus_min"] = c.dist.to_numpy() - dmin
    c["dist_ratio_min"] = (c.dist.to_numpy() + 1.0) / (dmin + 1.0)
    w = np.exp(-0.5 * (c.dist.to_numpy() / SIGMA_M) ** 2 + 0.5 * (dmin / SIGMA_M) ** 2)  # stable softmax
    c["gauss_share"] = w / pd.Series(w).groupby(pi).transform("sum").to_numpy()
    c["n_cands"] = grp.transform("size").to_numpy()
    lines = g.lines[arc]
    pts = g.points[pi]
    c["proj_frac"] = shapely.line_locate_point(lines, pts, normalized=True)
    c["arc_len"] = g.length[arc]
    c["dist_to_end_min"] = np.minimum(c.proj_frac, 1 - c.proj_frac) * c.arc_len

    # --- direction & motion ---
    mot = point_motion(trip)
    h = mot.heading.to_numpy()[pi]
    c["heading_diff"] = _ang_diff(h, g.bearing[arc])
    c["heading_diff_undirected"] = np.minimum(c.heading_diff, 180 - c.heading_diff)
    for col in ["heading_reliable", "speed", "accel", "dt_prev", "dt_next", "turn_rate"]:
        c[col] = mot[col].to_numpy()[pi]

    # --- context: do neighbouring GPS points also lie near this arc? ---
    for name, off in [("dist_prev1", -1), ("dist_prev2", -2), ("dist_next1", 1), ("dist_next2", 2)]:
        j = np.clip(pi + off, 0, n - 1)
        c[name] = shapely.distance(g.points[j], lines)
    c["ctx_mean_dist"] = c[["dist", "dist_prev1", "dist_prev2", "dist_next1", "dist_next2"]].mean(axis=1)

    # --- topology ---
    u, v = g.arcs[arc, 0], g.arcs[arc, 1]
    c["from_out_deg"] = g.out_deg[u]
    c["to_in_deg"] = g.in_deg[v]
    c["to_out_deg"] = g.out_deg[v]
    c["has_reverse"] = g.has_reverse[arc].astype(np.int8)

    c.insert(0, "trip", trip.trip_id)
    c.insert(1, "partition", trip.partition)
    c["t"] = trip.track.t.to_numpy()[pi]

    if with_labels and trip.route is not None:
        lab = align_points_to_route(trip)
        true_c = g.canonical[lab["true_arc"]]
        c["true_arc"] = lab["true_arc"][pi]
        c["label_dist"] = lab["label_dist"][pi]
        c["y"] = (g.canonical[arc] == true_c[pi]).astype(np.int8)
    for col in FEATURES:
        if c[col].dtype == np.float64:
            c[col] = c[col].astype(np.float32)
    return c
