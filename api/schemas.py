"""Request / response models of the PathSnap API (also rendered at /docs).

Coordinates are always [lat, lon] in WGS84 degrees. Times are seconds. Metrics are fractions in [0, 1].
The reference (ground-truth) route is used only on the server for scoring and is never returned.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

LatLon = list[float]          # [lat, lon]


class Health(BaseModel):
    status: str = Field(examples=["ok"])
    model: str = Field(examples=["HistGradientBoostingClassifier (tuned) + HMM/Viterbi"])
    artifact_version: int
    artifacts_created: str
    eval_set: str = Field(description="name of the folder the trips are served from")
    n_trips: int
    scoring_available: bool = Field(description="True if reference routes exist on the server for these trips")


class TripInfo(BaseModel):
    trip_id: str
    n_points: int = Field(description="number of GPS observations")
    duration_s: float
    start: LatLon
    end: LatLon


class TripList(BaseModel):
    eval_set: str
    trips: list[TripInfo]


class Track(BaseModel):
    trip_id: str
    n_points: int
    bounds: list[LatLon] = Field(description="[[min_lat, min_lon], [max_lat, max_lon]] for map fitting")
    gps_points: list[LatLon]
    t: list[float] = Field(description="timestamp of each GPS point (s)")


class PredictRequest(BaseModel):
    trip_id: str = Field(examples=["00000077"])


class Metrics(BaseModel):
    top1_accuracy: float = Field(description="share of GPS points matched to the correct road segment (classifier + HMM)")
    top1_accuracy_classifier_only: float = Field(description="same, using the classifier alone without the HMM")
    hmm_improvement_pp: float = Field(description="(top1_accuracy - top1_accuracy_classifier_only) * 100")
    route_continuity: float = Field(description="share of consecutive points whose segments are physically connected")
    route_precision: float = Field(description="share of predicted route length that lies on the reference route")
    route_recall: float = Field(description="share of reference route length recovered by the prediction")
    route_f1: float


class Timings(BaseModel):
    features_ms: float
    classifier_ms: float
    hmm_ms: float
    scoring_ms: float
    total_ms: float


class DecoderInfo(BaseModel):
    gap_splits: int = Field(description="GPS gaps longer than the threshold, decoded as separate pieces")
    break_steps: int = Field(description="steps where no candidate pair was connected in the road network")
    invalid_transitions_used: int
    unjoinable_transitions: int = Field(description="consecutive decoded segments that could not be joined by a road path")
    network_connectivity: float


class PredictResponse(BaseModel):
    request_id: str
    trip_id: str
    eval_set: str
    n_points: int
    n_candidate_rows: int
    n_route_segments: int
    route_length_m: float
    duration_s: float
    bounds: list[LatLon]
    gps_points: list[LatLon] = Field(description="raw GPS observations in time order")
    matched_segment: list[int] = Field(description="predicted road segment (arc id) for each GPS point (HMM)")
    match_probability: list[float] = Field(description="classifier probability of each chosen segment")
    predicted_route: list[list[LatLon]] = Field(description="HMM route as polylines (each a list of [lat, lon])")
    classifier_only_route: list[list[LatLon]] = Field(description="route from the classifier alone, for comparison")
    metrics: Optional[Metrics] = Field(description="null if no reference route exists on the server")
    timings: Timings
    decoder: DecoderInfo


class Error(BaseModel):
    detail: str
