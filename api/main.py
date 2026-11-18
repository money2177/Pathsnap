"""PathSnap FastAPI backend (Laptop 1).

Serves the trips in an evaluation folder, runs the complete pipeline (candidates -> features ->
tuned HistGradientBoosting -> HMM/Viterbi -> connected route) on request, and scores the result
server-side against the reference route. The reference route is never returned.

Configuration (environment variables):
    PATHSNAP_EVAL_DIR   folder containing one sub-folder per trip (<id>/<id>.track/.nodes/.arcs[/.route])
                        default: <PATHSNAP_DATA or <repo>/data>/student_test
    PATHSNAP_ARTIFACTS  exported artifacts folder, default: <repo>/artifacts
    PATHSNAP_LOG        log file, default: <repo>/api/logs/api.log

Run:  python -m api            (binds 0.0.0.0:8000, prints the LAN address)
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import Error, Health, PredictRequest, PredictResponse, Track, TripInfo, TripList
from src.data_io import DATA_ROOT, REPO_ROOT, Trip
from src.hmm import reconstruct_route
from src.pipeline import InferenceTrip, PathSnapPipeline, score_prediction

EVAL_DIR = os.path.abspath(os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(DATA_ROOT, "student_test")))
ARTIFACTS = os.environ.get("PATHSNAP_ARTIFACTS", os.path.join(REPO_ROOT, "artifacts"))
LOG_FILE = os.environ.get("PATHSNAP_LOG", os.path.join(REPO_ROOT, "api", "logs", "api.log"))
ROOT, EVAL_SET = os.path.dirname(EVAL_DIR), os.path.basename(EVAL_DIR)

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)):
        h.setFormatter(fmt); log.addHandler(h)
    log.propagate = False

STATE: dict = {}


def list_eval_trips() -> list[str]:
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(d for d in os.listdir(EVAL_DIR)
                  if os.path.isfile(os.path.join(EVAL_DIR, d, f"{d}.track")))


def has_reference(tid: str) -> bool:
    return os.path.isfile(os.path.join(EVAL_DIR, tid, f"{tid}.route"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    STATE["pipe"] = PathSnapPipeline.load(ARTIFACTS)          # verifies sha256 of every artifact
    STATE["trips"] = list_eval_trips()
    import json
    STATE["manifest"] = json.load(open(os.path.join(ARTIFACTS, "manifest.json")))
    log.info(f"model loaded from {ARTIFACTS} in {time.perf_counter()-t0:.2f}s (checksums verified)")
    log.info(f"serving {len(STATE['trips'])} trips from {EVAL_DIR}: {', '.join(STATE['trips'])}")
    log.info(f"reference routes available for scoring: {sum(map(has_reference, STATE['trips']))}/{len(STATE['trips'])}")
    yield


app = FastAPI(title="PathSnap API", version="1.0",
              description="Map-matching and route reconstruction: tuned HistGradientBoosting + HMM/Viterbi. "
                          "Ground-truth routes stay on the server; only metrics are returned.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def check_trip(tid: str) -> str:
    if tid not in STATE["trips"]:      # also rules out path traversal: only listed folder names are accepted
        raise HTTPException(404, f"unknown trip_id '{tid}'. Available: {STATE['trips']}")
    return tid


def to_polylines(trip: Trip, arcs: list[int]) -> list[list[list[float]]]:
    """Arc sequence -> polylines of [lat, lon]; a new polyline starts wherever arcs are not joined."""
    a = trip.arcs.to_numpy()
    lat, lon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
    lines, prev_v = [], None
    for arc in arcs:
        u, v = a[arc]
        if prev_v is None or u != prev_v:
            lines.append([[round(lat[u], 6), round(lon[u], 6)]])
        lines[-1].append([round(lat[v], 6), round(lon[v], 6)])
        prev_v = v
    return lines


def bounds(tr: pd.DataFrame):
    return [[round(tr.lat.min(), 6), round(tr.lon.min(), 6)], [round(tr.lat.max(), 6), round(tr.lon.max(), 6)]]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.url.path not in ("/docs", "/openapi.json", "/favicon.ico"):
        log.info(f"{request.client.host if request.client else '?'} {request.method} {request.url.path} "
                 f"-> {response.status_code} ({(time.perf_counter()-t0)*1000:.0f} ms)")
    return response


@app.get("/health", response_model=Health)
def health():
    m = STATE["manifest"]
    return Health(status="ok", model="HistGradientBoostingClassifier (tuned) + HMM/Viterbi",
                  artifact_version=m["artifact_version"], artifacts_created=m["created"], eval_set=EVAL_SET,
                  n_trips=len(STATE["trips"]), scoring_available=all(map(has_reference, STATE["trips"])))


@app.get("/trips", response_model=TripList)
def trips():
    out = []
    for tid in STATE["trips"]:
        tr = InferenceTrip(tid, EVAL_SET, ROOT).track
        out.append(TripInfo(trip_id=tid, n_points=len(tr), duration_s=float(tr.t.iloc[-1] - tr.t.iloc[0]),
                            start=[round(tr.lat.iloc[0], 6), round(tr.lon.iloc[0], 6)],
                            end=[round(tr.lat.iloc[-1], 6), round(tr.lon.iloc[-1], 6)]))
    return TripList(eval_set=EVAL_SET, trips=out)


@app.get("/trips/{trip_id}/track", response_model=Track, responses={404: {"model": Error}})
def track(trip_id: str):
    tr = InferenceTrip(check_trip(trip_id), EVAL_SET, ROOT).track
    return Track(trip_id=trip_id, n_points=len(tr), bounds=bounds(tr),
                 gps_points=np.round(tr[["lat", "lon"]].to_numpy(), 6).tolist(), t=tr.t.astype(float).tolist())


@app.post("/predict", response_model=PredictResponse, responses={404: {"model": Error}, 500: {"model": Error}})
def predict(req: PredictRequest, request: Request):
    tid = check_trip(req.trip_id)
    rid = uuid.uuid4().hex[:8]
    client = request.client.host if request.client else "?"
    log.info(f"[{rid}] PREDICT trip={tid} from {client}: running pipeline")
    try:
        pipe = STATE["pipe"]
        pred = pipe.predict_trip(tid, EVAL_SET, root=ROOT)          # InferenceTrip: no access to .route
        trip = InferenceTrip(tid, EVAL_SET, ROOT)
        pts = pred["points"]
        net = pred["_net"]
        clf_route, _ = reconstruct_route(net, pts.arc_id_classifier.to_numpy())
        t0 = time.perf_counter()
        metrics = None
        if has_reference(tid):                                        # server-side scoring only
            s = score_prediction(Trip(tid, EVAL_SET, ROOT), pred)
            metrics = {k: s[k] for k in ["top1_accuracy", "top1_accuracy_classifier_only", "route_continuity",
                                         "route_precision", "route_recall", "route_f1"]}
            metrics["hmm_improvement_pp"] = 100 * (s["top1_accuracy"] - s["top1_accuracy_classifier_only"])
        scoring_s = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - report any pipeline failure to the client and the log
        log.exception(f"[{rid}] FAILED trip={tid}: {e}")
        raise HTTPException(500, f"pipeline failed for trip {tid}: {e}")

    tm = pred["timings"]
    timings = {"features_ms": 1000 * tm["features_s"], "classifier_ms": 1000 * tm["classifier_s"],
               "hmm_ms": 1000 * tm["hmm_s"], "scoring_ms": 1000 * scoring_s,
               "total_ms": 1000 * (tm["total_s"] + scoring_s)}
    cc = pred["route_continuity_check"]
    route = pred["route_arcs"]
    length = float(net.g.length[np.asarray(route, dtype=int)].sum()) if route else 0.0
    if metrics:
        log.info(f"[{rid}] RESULT trip={tid} points={pred['n_points']} candidates={pred['n_candidate_rows']} | "
                 f"top1={metrics['top1_accuracy']:.4f} (classifier only {metrics['top1_accuracy_classifier_only']:.4f}, "
                 f"{metrics['hmm_improvement_pp']:+.2f} pp) continuity={metrics['route_continuity']:.4f} "
                 f"route_f1={metrics['route_f1']:.4f} | total {timings['total_ms']:.0f} ms")
    else:
        log.info(f"[{rid}] RESULT trip={tid} points={pred['n_points']} (no reference route: unscored) "
                 f"| total {timings['total_ms']:.0f} ms")
    return PredictResponse(
        request_id=rid, trip_id=tid, eval_set=EVAL_SET, n_points=pred["n_points"],
        n_candidate_rows=pred["n_candidate_rows"], n_route_segments=len(route), route_length_m=length,
        duration_s=float(pts.t.iloc[-1] - pts.t.iloc[0]), bounds=bounds(trip.track),
        gps_points=np.round(pts[["lat", "lon"]].to_numpy(), 6).tolist(),
        matched_segment=pts.arc_id.astype(int).tolist(), match_probability=np.round(pts.prob, 4).tolist(),
        predicted_route=to_polylines(trip, route), classifier_only_route=to_polylines(trip, clf_route),
        metrics=metrics, timings=timings,
        decoder={"gap_splits": cc["n_gap_splits"], "break_steps": cc["n_break_steps"],
                 "invalid_transitions_used": cc["n_invalid_used"], "unjoinable_transitions": cc["unjoinable_transitions"],
                 "network_connectivity": cc["network_connectivity"]})
