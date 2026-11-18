"""Stage 8: the complete PathSnap inference pipeline, loaded from the exported artifacts.

    pipe = PathSnapPipeline.load("artifacts")
    pred = pipe.predict_trip(trip_dir_or_trip)      # never reads the .route file
    score = score_prediction(reference_trip, pred)  # server-side only, uses .route

Prediction steps (identical to stages 3-7):
    1. candidate arcs per GPS point          src/candidates.py  (radius, k, fallback)
    2. 26 features per (point, candidate)    src/features.py
    3. probability per candidate             tuned HistGradientBoosting
    4. Viterbi decoding of the candidate HMM src/hmm.py         (tuned decoder parameters)
    5. connected route through decoded arcs  src/hmm.reconstruct_route
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

from . import features as F
from . import hmm as H
from .candidates import TripGraph
from .data_io import DATA_ROOT, Trip
from .features import FEATURES, build_trip_features
from .labels import align_points_to_route

ARTIFACT_VERSION = 1


class InferenceTrip(Trip):
    """A Trip whose reference route is unavailable by construction: the prediction pipeline can
    only see .track, .nodes and .arcs, even if a .route file exists on disk."""

    @cached_property
    def route(self):
        return None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PathSnapPipeline:
    def __init__(self, model, config: dict, artifacts_dir: str | None = None):
        self.model = model
        self.config = config
        self.dir = artifacts_dir
        self.hmm_params = H.HMMParams(**config["hmm"]["params"])
        self._check_code_constants()

    # ---------- loading ----------
    @classmethod
    def load(cls, artifacts_dir: str, verify_checksums: bool = True) -> "PathSnapPipeline":
        config = json.load(open(os.path.join(artifacts_dir, "config.json")))
        manifest = json.load(open(os.path.join(artifacts_dir, "manifest.json")))
        model_path = os.path.join(artifacts_dir, config["model"]["file"])
        if verify_checksums:
            for rel, info in manifest["files"].items():
                got = sha256(os.path.join(artifacts_dir, rel))
                if got != info["sha256"]:
                    raise ValueError(f"artifact {rel} is corrupted or was changed (sha256 mismatch)")
        model = joblib.load(model_path)
        return cls(model, config, artifacts_dir)

    def _check_code_constants(self):
        """The code that computes features / transitions must match what the model was built with."""
        c = self.config
        if list(c["features"]["names"]) != list(FEATURES):
            raise ValueError("feature list in src/features.py differs from the exported model's")
        expect = {"SIGMA_M": F.SIGMA_M, "MIN_MOVE_M": F.MIN_MOVE_M}
        for k, v in c["features"]["constants"].items():
            if expect[k] != v:
                raise ValueError(f"features.{k}={expect[k]} but the model was trained with {v}")
        expect = {"BACKWARD_TOL_M": H.BACKWARD_TOL_M, "ROUTE_FACTOR": H.ROUTE_FACTOR,
                  "MIN_SEARCH_M": H.MIN_SEARCH_M, "MAX_SEARCH_M": H.MAX_SEARCH_M}
        for k, v in c["hmm"]["constants"].items():
            if expect[k] != v:
                raise ValueError(f"hmm.{k}={expect[k]} but the decoder was tuned with {v}")

    # ---------- prediction ----------
    def predict_trip(self, trip: Trip | str, partition: str | None = None, root: str = DATA_ROOT) -> dict:
        """Full prediction for one trip. `trip` is a Trip, or a trip id plus its partition folder.
        Returns per-point arcs/probabilities, the connected route, and timings."""
        if isinstance(trip, str):
            trip = InferenceTrip(trip, partition, root)
        elif not isinstance(trip, InferenceTrip):
            trip = InferenceTrip(trip.trip_id, trip.partition, trip.root)
        cand_cfg = self.config["candidates"]
        t = {}
        t0 = time.perf_counter()
        g = TripGraph(trip)
        feats = build_trip_features(trip, radius_m=cand_cfg["radius_m"], k=cand_cfg["k"], with_labels=False)
        t["features_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        prob = self.model.predict_proba(feats[FEATURES].to_numpy(np.float32))[:, 1]
        t["classifier_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        net = H.RoadNetwork(trip, g)
        lat = H.build_lattice(net, feats, prob)
        rows, info = H.viterbi(lat, self.hmm_params)
        route, holes = H.reconstruct_route(net, lat.arc[rows])
        t["hmm_s"] = time.perf_counter() - t0
        clf_rows = H.classifier_rows(lat)
        points = pd.DataFrame({
            "point_idx": lat.point_idx,
            "lon": trip.track.lon.to_numpy()[lat.point_idx], "lat": trip.track.lat.to_numpy()[lat.point_idx],
            "t": trip.track.t.to_numpy()[lat.point_idx],
            "arc_id": lat.arc[rows], "prob": lat.prob[rows],
            "arc_id_classifier": lat.arc[clf_rows], "prob_classifier": lat.prob[clf_rows],
        })
        t["total_s"] = sum(t.values())
        return {"trip_id": trip.trip_id, "n_points": len(points), "n_candidate_rows": len(feats),
                "points": points, "route_arcs": [int(a) for a in route],
                "route_continuity_check": {"unjoinable_transitions": holes, **info,
                                           "network_connectivity": H.path_connectivity(lat, rows, self.hmm_params.gap_s)},
                "timings": t, "_net": net}


def score_prediction(reference: Trip, pred: dict) -> dict:
    """Server-side evaluation against the reference route (never exposed to the pipeline/UI).
    Point accuracy: decoded arc equals the arc each GPS point is aligned to on .route (the stage-3
    labels; duplicate (from, to) arcs compare as equal). Route metrics: length-weighted overlap."""
    if reference.route is None:
        raise ValueError(f"no reference route for trip {reference.trip_id}")
    g = pred["_net"].g if "_net" in pred else TripGraph(reference)
    lab = align_points_to_route(reference)
    pts = pred["points"]
    true_arc = g.canonical[lab["true_arc"][pts.point_idx.to_numpy()]]
    ok = g.canonical[pts.arc_id.to_numpy()] == true_arc
    ok_clf = g.canonical[pts.arc_id_classifier.to_numpy()] == true_arc
    from .evaluation import continuity
    return {"top1_accuracy": float(ok.mean()), "top1_accuracy_classifier_only": float(ok_clf.mean()),
            "route_continuity": continuity(pts.arc_id.to_numpy(), reference.trip_id, reference.partition, reference.root),
            **H.route_overlap(g, pred["route_arcs"], reference.route),
            "point_correct": ok.astype(int).tolist(),
            "reference_route_arcs": [int(a) for a in reference.route]}
