"""Candidate road segments for every GPS point.

A candidate is a road arc within `radius_m` of the point, keeping at most the `k` nearest.
Arcs with the same (from, to) node pair are merged into one canonical arc (the dataset
contains exact duplicates). Zero-length arcs are skipped because they have no direction.
Points with no arc inside the radius fall back to their `k_fallback` nearest arcs, so every
point always has at least one candidate (needed by the HMM later).
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
import shapely

from .data_io import Trip


class TripGraph:
    """Road-network geometry and topology for one trip, built once and reused."""

    def __init__(self, trip: Trip):
        self.trip = trip
        a = trip.arcs.to_numpy()
        self.arcs = a
        # canonical id: first arc with the same (u, v)
        key = a[:, 0] * (len(trip.nodes) + 1) + a[:, 1]
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        self.canonical = first[inv].astype(np.int64)
        self.length = trip.arc_length
        A, B = trip.arc_xy
        self.A, self.B = A, B
        self.bearing = (np.degrees(np.arctan2(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])) + 360) % 360
        self.eligible = np.flatnonzero((self.canonical == np.arange(len(a))) & (self.length > 0.01))
        self.lines = shapely.linestrings(np.stack([A, B], axis=1))
        self.tree = shapely.STRtree(self.lines[self.eligible])
        n = len(trip.nodes)
        self.out_deg = np.bincount(a[:, 0], minlength=n)
        self.in_deg = np.bincount(a[:, 1], minlength=n)
        pairs = pd.MultiIndex.from_arrays([a[:, 0], a[:, 1]])
        rev = pd.MultiIndex.from_arrays([a[:, 1], a[:, 0]])
        self.has_reverse = rev.isin(pairs)

    @cached_property
    def points(self):
        return shapely.points(self.trip.track_xy)


def generate_candidates(g: TripGraph, radius_m: float = 50.0, k: int = 12, k_fallback: int = 3) -> pd.DataFrame:
    """Long table with one row per (point_idx, arc_id) candidate, nearest first, plus `dist`."""
    pts = g.points
    pi, ti = g.tree.query(pts, predicate="dwithin", distance=radius_m)
    arc = g.eligible[ti]
    dist = shapely.distance(pts[pi], g.lines[arc])
    df = pd.DataFrame({"point_idx": pi, "arc_id": arc, "dist": dist, "fallback": False})

    # Fallback: widen the search for points with nothing inside the radius.
    missing = np.setdiff1d(np.arange(len(pts)), np.unique(pi))
    for wider in (4 * radius_m, 20 * radius_m):
        if not len(missing):
            break
        mi, mt = g.tree.query(pts[missing], predicate="dwithin", distance=wider)
        if len(mi):
            arc = g.eligible[mt]
            fb = pd.DataFrame({"point_idx": missing[mi], "arc_id": arc,
                               "dist": shapely.distance(pts[missing[mi]], g.lines[arc]), "fallback": True})
            df = pd.concat([df, fb], ignore_index=True)
        missing = np.setdiff1d(missing, missing[mi])
    if len(missing):                                     # last resort: single nearest arc
        (mi, mt), md = g.tree.query_nearest(pts[missing], return_distance=True, all_matches=False)
        df = pd.concat([df, pd.DataFrame({"point_idx": missing[mi], "arc_id": g.eligible[mt],
                                          "dist": md, "fallback": True})], ignore_index=True)

    # arc_id breaks exact distance ties (the two directions of a two-way road have identical geometry),
    # so candidate order, dist_rank and the k cut-off are defined by the data, not by the spatial index.
    df = df.sort_values(["point_idx", "dist", "arc_id"], kind="stable")
    df["dist_rank"] = df.groupby("point_idx").cumcount()
    df = df[df.dist_rank < np.where(df.fallback, k_fallback, k)].reset_index(drop=True)
    return df
