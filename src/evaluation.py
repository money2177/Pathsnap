"""Metrics for candidate classification and route matching.

Row level (one row = one (GPS point, candidate arc) pair):
  accuracy, precision, recall, macro F1, log loss, confusion matrix (threshold 0.5)
Point level:
  Top-1 accuracy   share of GPS points whose highest-scoring candidate is the true arc.
                   Points whose true arc is not among the candidates count as errors.
  Candidate miss   share of points with no positive candidate.
Sequence level:
  Route continuity share of consecutive point pairs whose chosen arcs are physically
                   connected: same arc, directly adjacent (to_node == from_node), or one
                   intermediate arc apart (short arcs can be crossed in under 1 s).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from .data_io import DATA_ROOT, Trip


def row_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
    }


def top1_choice(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """One row per GPS point: the candidate with the highest score."""
    d = df[["trip", "point_idx", "arc_id", "y"]].assign(score=score)
    idx = d.groupby(["trip", "point_idx"], sort=True).score.idxmax()
    return d.loc[idx.values].reset_index(drop=True)


def top1_accuracy(df: pd.DataFrame, score: np.ndarray) -> float:
    return float(top1_choice(df, score).y.mean())


def candidate_miss_rate(df: pd.DataFrame) -> float:
    return float(1 - df.groupby(["trip", "point_idx"]).y.max().mean())


class _Topo:
    """Arc endpoints and 1-hop successors for one trip (cached per trip id)."""
    cache: dict = {}

    @classmethod
    def get(cls, trip_id: str, partition: str, root: str = DATA_ROOT):
        key = (trip_id, partition, root)
        if key not in cls.cache:
            a = Trip(trip_id, partition, root).arcs.to_numpy()
            succ = pd.Series(a[:, 1]).groupby(a[:, 0]).apply(set).to_dict()
            cls.cache[key] = (a, succ)
        return cls.cache[key]


def continuity(arcs_seq: np.ndarray, trip_id: str, partition: str, root: str = DATA_ROOT) -> float:
    a, succ = _Topo.get(trip_id, partition, root)
    p, n = arcs_seq[:-1], arcs_seq[1:]
    same = p == n
    vp, un = a[p, 1], a[n, 0]
    adj = vp == un
    two = np.array([un_ in succ.get(vp_, ()) for vp_, un_ in zip(vp, un)]) if len(vp) else np.array([])
    rev = (a[p, 0] == a[n, 1]) & (a[p, 1] == a[n, 0])   # u-turn onto the twin arc
    ok = same | adj | two | rev
    return float(ok.mean()) if len(ok) else 1.0


def route_continuity(choice: pd.DataFrame, partition_of: dict) -> float:
    """Point-weighted continuity over trips; `choice` from top1_choice (sorted by point_idx)."""
    tot, num = 0, 0.0
    for tid, g in choice.groupby("trip"):
        c = continuity(g.arc_id.to_numpy(), tid, partition_of[tid])
        num += c * (len(g) - 1); tot += len(g) - 1
    return num / max(tot, 1)


def all_metrics(df: pd.DataFrame, p: np.ndarray, partition_of: dict | None = None) -> dict:
    m = row_metrics(df.y.to_numpy(), p)
    ch = top1_choice(df, p)
    m["top1_accuracy"] = float(ch.y.mean())
    if partition_of is not None:
        m["route_continuity"] = route_continuity(ch, partition_of)
    return m


def confusion(df: pd.DataFrame, p: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return confusion_matrix(df.y.to_numpy(), (p >= thr).astype(int), labels=[0, 1])
