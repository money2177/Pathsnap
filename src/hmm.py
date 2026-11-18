"""Stage 7: HMM map matching with Viterbi decoding.

Hidden states
    At GPS observation t the hidden state is "the vehicle is on candidate arc c", one state per
    candidate produced by src/candidates.py (radius 50 m, at most 12 arcs, fallback widening, so
    every observation has at least one state). The candidate table keeps the arc_id of each state,
    so a decoded state maps straight back to a road segment.

Emission score   (log domain)
    e_t(c) = w * log(max(p_t(c), floor))
    p_t(c) is the tuned HistGradientBoosting probability that candidate c is the true arc. A
    discriminative posterior stands in for the likelihood (the "hybrid HMM" approximation, with a
    uniform prior over candidates). The weight w sets how much the decoder trusts the classifier
    relative to the road network.

Transition score  (log domain, Newson & Krumm 2009 style)
    route_d = network driving distance from the projection of GPS point t-1 on arc i to the
              projection of point t on arc j:
                same arc, forward (or backward by <= BACKWARD_TOL_M of GPS jitter): |f_j - f_i| * L
                otherwise: (1 - f_i) * L_i + shortest_path(to_node(i), from_node(j)) + f_j * L_j
    gc      = straight-line distance between the two GPS points
    T(i, j) = -|route_d - gc| / beta - uturn_cost * [j is the reverse twin of i]
    A vehicle's driven distance between two fixes is close to their straight-line distance, so
    candidate pairs that need a long detour are unlikely.

Invalid transitions and gaps
    No network path within the search limit (max(ROUTE_FACTOR * gc + 200 m, 400 m), capped at
    MAX_SEARCH_M) means the transition is invalid. It gets the finite score -break_cost instead of
    -inf, so Viterbi never dead-ends. If a whole step is invalid, the chain "breaks" and continues
    from the best emission.
    A time gap > gap_s between fixes splits the chain: the transition is set flat (0), so both
    sides decode independently. Very long gaps would otherwise force an arbitrary connection.

Viterbi recurrence
    delta_1(c)   = e_1(c)
    delta_t(j)   = e_t(j) + max_i [delta_{t-1}(i) + T_t(i, j)],   psi_t(j) = argmax_i [...]
    then backtrack from argmax_c delta_N(c). This is the most likely state sequence, in O(N K^2).
"""
from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .candidates import TripGraph
from .data_io import Trip

BACKWARD_TOL_M = 8.0      # backward movement along the same arc accepted as GPS jitter
ROUTE_FACTOR = 3.0        # shortest-path search limit relative to the straight-line distance
MIN_SEARCH_M = 400.0
MAX_SEARCH_M = 8000.0


@dataclass
class HMMParams:
    emission_weight: float = 1.0   # w
    beta: float = 20.0             # metres, scale of the |route - straight line| mismatch
    break_cost: float = 30.0       # score of an invalid (unreachable) transition
    uturn_cost: float = 2.0        # extra penalty for reversing onto the twin arc
    gap_s: float = 60.0            # time gap that splits the chain
    prob_floor: float = 1e-4

    def to_dict(self):
        return asdict(self)


class RoadNetwork:
    """Directed road graph of one trip with cached, distance-bounded Dijkstra searches."""

    def __init__(self, trip: Trip, g: TripGraph | None = None):
        self.trip = trip
        self.g = g if g is not None else TripGraph(trip)
        a, L = self.g.arcs, self.g.length
        self.u, self.v = a[:, 0], a[:, 1]
        self.adj: dict[int, list] = {}
        for arc in np.flatnonzero(self.g.canonical == np.arange(len(a))):   # one arc per (u, v)
            self.adj.setdefault(int(a[arc, 0]), []).append((int(a[arc, 1]), float(L[arc]), int(arc)))
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def search(self, src: int, limit: float):
        """(dist, pred_arc) for all nodes within `limit` metres of node `src`."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1], hit[2]
        dist, pred, pq = {src: 0.0}, {}, [(0.0, src)]
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, np.inf) or d > limit:
                continue
            for m, w, arc in self.adj.get(n, ()):
                nd = d + w
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m], pred[m] = nd, arc
                    heapq.heappush(pq, (nd, m))
        self._cache[src] = (limit, dist, pred)
        return dist, pred

    def path(self, src: int, dst: int, limit: float = MAX_SEARCH_M):
        """Arc ids of the shortest path src -> dst, or None if beyond `limit`."""
        if src == dst:
            return []
        dist, pred = self.search(src, limit)
        if dst not in dist:
            return None
        arcs, n = [], dst
        while n != src:
            arc = pred[n]; arcs.append(arc); n = int(self.u[arc])
        return arcs[::-1]


@dataclass
class Lattice:
    """Parameter-independent HMM ingredients for one trip (built once, decoded many times)."""
    trip_id: str
    point_idx: np.ndarray        # (N,) GPS point index of each step
    starts: np.ndarray           # (N+1,) offsets into the candidate rows
    arc: np.ndarray              # (R,) candidate arc ids, rows grouped by step
    prob: np.ndarray             # (R,) classifier probability
    dist: np.ndarray             # (R,) GPS-to-arc distance (for the geometric-emission ablation)
    y: np.ndarray | None         # (R,) 1 if the candidate is the reference arc
    mismatch: list               # per step t>=1: (K_{t-1}, K_t) |route - gc| in metres, nan if unreachable
    uturn: list                  # per step t>=1: (K_{t-1}, K_t) bool
    dt: np.ndarray               # (N,) seconds since the previous fix


def build_lattice(net: RoadNetwork, cands: pd.DataFrame, prob: np.ndarray) -> Lattice:
    """`cands`: candidate rows of one trip (point_idx, arc_id, proj_frac, dist[, y])."""
    order = np.lexsort((cands.arc_id.to_numpy(), -prob, cands.point_idx.to_numpy()))   # arc_id breaks probability ties
    c = cands.iloc[order]
    p = np.asarray(prob)[order]
    pts = c.point_idx.to_numpy()
    step_pts, starts = np.unique(pts, return_index=True)
    starts = np.r_[starts, len(c)]
    arc, frac = c.arc_id.to_numpy(), c.proj_frac.to_numpy()
    xy = net.trip.track_xy[step_pts]
    t = net.trip.track.t.to_numpy()[step_pts]
    dt = np.r_[0.0, np.diff(t)]
    gc = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    L, u, v = net.g.length, net.u, net.v
    mism, utn = [], []
    for s in range(1, len(step_pts)):
        a0, a1 = arc[starts[s - 1]:starts[s]], arc[starts[s]:starts[s + 1]]
        f0, f1 = frac[starts[s - 1]:starts[s]], frac[starts[s]:starts[s + 1]]
        limit = min(max(ROUTE_FACTOR * gc[s] + 200.0, MIN_SEARCH_M), MAX_SEARCH_M)
        R = np.full((len(a0), len(a1)), np.nan)
        for i, ai in enumerate(a0):
            dist, _ = net.search(int(v[ai]), limit)
            head = (1 - f0[i]) * L[ai]
            for j, aj in enumerate(a1):
                if ai == aj and (f1[j] - f0[i]) * L[ai] >= -BACKWARD_TOL_M:
                    R[i, j] = abs(f1[j] - f0[i]) * L[ai]
                    continue
                d = dist.get(int(u[aj]))
                if d is not None:
                    R[i, j] = head + d + f1[j] * L[aj]
        mism.append(np.abs(R - gc[s]))
        utn.append((u[a0][:, None] == v[a1][None, :]) & (v[a0][:, None] == u[a1][None, :]))
    return Lattice(net.trip.trip_id, step_pts, starts, arc, p, c.dist.to_numpy(),
                   c.y.to_numpy() if "y" in c else None, mism, utn, dt)


def viterbi(lat: Lattice, prm: HMMParams, emission: str = "classifier"):
    """Most likely candidate per step. Returns (row index into lattice arrays, info dict).

    emission="classifier" uses the classifier probability; "distance" uses the classic Gaussian
    GPS-noise emission -0.5 (dist / 5 m)^2 (ablation without the classifier)."""
    if emission == "classifier":
        e_all = prm.emission_weight * np.log(np.maximum(lat.prob, prm.prob_floor))
    else:
        e_all = -0.5 * (lat.dist / 5.0) ** 2
    st = lat.starts
    N = len(st) - 1
    gap = lat.dt > prm.gap_s
    back = []
    delta = e_all[st[0]:st[1]].copy()
    n_break = n_split = 0
    for s in range(1, N):
        e = e_all[st[s]:st[s + 1]]
        if gap[s]:
            T = np.zeros((len(delta), len(e))); n_split += 1
        else:
            m = lat.mismatch[s - 1]
            T = np.where(np.isnan(m), -prm.break_cost, -np.nan_to_num(m) / prm.beta) - prm.uturn_cost * lat.uturn[s - 1]
            if np.isnan(m).all():
                n_break += 1
        sc = delta[:, None] + T
        b = sc.argmax(0)
        back.append(b)
        delta = sc[b, np.arange(len(e))] + e
        delta -= delta.max()
    k = np.empty(N, dtype=np.int64)
    k[-1] = int(delta.argmax())
    for s in range(N - 1, 0, -1):
        k[s - 1] = back[s - 1][k[s]]
    rows = st[:-1] + k
    # transitions actually used by the decoded path that are invalid (network-disconnected)
    invalid = sum(1 for s in range(1, N) if not gap[s]
                  and np.isnan(lat.mismatch[s - 1][k[s - 1], k[s]]))
    return rows, {"n_break_steps": n_break, "n_gap_splits": n_split, "n_invalid_used": invalid}


def classifier_rows(lat: Lattice) -> np.ndarray:
    """Classifier-only choice: highest probability per step (rows are sorted by -prob)."""
    return lat.starts[:-1].copy()


def path_connectivity(lat: Lattice, rows: np.ndarray, gap_s: float = 60.0) -> float:
    """Share of consecutive steps (excluding split gaps) whose chosen arcs are network-reachable."""
    k = rows - lat.starts[:-1]
    ok = [not np.isnan(lat.mismatch[s - 1][k[s - 1], k[s]]) for s in range(1, len(k)) if lat.dt[s] <= gap_s]
    return float(np.mean(ok)) if ok else 1.0


def reconstruct_route(net: RoadNetwork, arcs: np.ndarray) -> tuple[list, int]:
    """Connected arc sequence through the decoded arcs: consecutive distinct arcs are joined by
    the shortest network path. Returns (route arcs, number of joins that had no path)."""
    seq = [int(a) for i, a in enumerate(arcs) if i == 0 or a != arcs[i - 1]]
    if not seq:
        return [], 0
    route, holes = [seq[0]], 0
    for a, b in zip(seq[:-1], seq[1:]):
        if net.v[a] == net.u[b]:
            route.append(b); continue
        p = net.path(int(net.v[a]), int(net.u[b]))
        if p is None:
            holes += 1
        else:
            route.extend(p)
        route.append(b)
    return route, holes


def route_overlap(g: TripGraph, pred_arcs, true_arcs) -> dict:
    """Length-weighted precision / recall / F1 of the predicted arc set against the reference route
    (arcs compared as canonical (from, to) pairs)."""
    P = set(g.canonical[np.asarray(pred_arcs, dtype=np.int64)].tolist()) if len(pred_arcs) else set()
    T = set(g.canonical[np.asarray(true_arcs, dtype=np.int64)].tolist())
    L = g.length
    inter = sum(L[a] for a in P & T)
    lp, lt = sum(L[a] for a in P), sum(L[a] for a in T)
    prec = inter / lp if lp else 0.0
    rec = inter / lt if lt else 0.0
    return {"route_precision": prec, "route_recall": rec,
            "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
