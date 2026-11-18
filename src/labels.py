"""Per-point ground-truth labels derived from the reference route.

The .route file lists arcs in driving order but has no link to individual GPS points
(about 5 points per arc). Each point is assigned to one route arc with a monotonic
alignment: point i may only be matched to a route position at or after the position
of point i-1. This respects driving order, handles routes that pass the same arc twice,
and stops a noisy point from snapping to a far-away part of the route.

Dynamic programme (positions j = 0..m-1 along the route):
    D[0, j] = d(0, j)
    D[i, j] = d(i, j) + min_{k <= j} D[i-1, k]
where d(i, j) is the distance in metres from point i to route arc j. Backtracking gives
the globally best order-preserving assignment.
"""
from __future__ import annotations

import numpy as np

from .data_io import Trip


def _dist_to_segments(p: np.ndarray, A: np.ndarray, B: np.ndarray, L2: np.ndarray) -> np.ndarray:
    AB = B - A
    t = np.clip(((p - A) * AB).sum(1) / L2, 0.0, 1.0)
    return np.hypot(*(p - (A + t[:, None] * AB)).T)


def align_points_to_route(trip: Trip) -> dict[str, np.ndarray]:
    """Return, for every GPS point:
    route_pos  index into trip.route that the point was assigned to
    true_arc   arc_id at that position
    label_dist distance (m) from the point to its assigned arc
    """
    route = trip.route
    A, B = trip.arc_xy
    A, B = A[route], B[route]
    L2 = ((B - A) ** 2).sum(1)
    zero = L2 < 1e-4                      # zero-length arcs cannot be told apart geometrically
    L2 = np.where(zero, 1.0, L2)
    P = trip.track_xy
    n, m = len(P), len(route)

    back = np.empty((n, m), dtype=np.int16 if m < 32767 else np.int32)
    ar = np.arange(m)
    d = _dist_to_segments(P[0], A, B, L2); d[zero] = np.inf
    D = d
    for i in range(1, n):
        pm = np.minimum.accumulate(D)                                  # prefix minimum
        back[i] = np.maximum.accumulate(np.where(D == pm, ar, 0))      # its argmin
        d = _dist_to_segments(P[i], A, B, L2); d[zero] = np.inf
        D = d + pm

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(D))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    true_arc = route[pos]
    dist = _dist_to_segments_pairs(P, A[pos], B[pos])
    return {"route_pos": pos, "true_arc": true_arc, "label_dist": dist}


def _dist_to_segments_pairs(P, A, B):
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((P - A) * AB).sum(1) / L2, 0.0, 1.0)
    return np.hypot(*(P - (A + t[:, None] * AB)).T)
