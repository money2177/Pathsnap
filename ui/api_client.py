"""HTTP boundary for the documented PathSnap evaluator API."""
import math
from urllib.parse import quote, urlsplit

import requests


class APIError(Exception):
    """A message safe to display in the evaluator UI."""


def normalize_url(value: str) -> str:
    value = value.strip().rstrip("/")
    try:
        url = urlsplit(value)
        valid = url.scheme in ("http", "https") and url.hostname and not (
            url.username or url.password or url.path or url.query or url.fragment
        )
        _ = url.port
    except ValueError:
        valid = False
    if not valid:
        raise APIError("Enter an HTTP(S) base URL with a host and optional port.")
    return value


def number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def coordinate(value) -> bool:
    return (
        isinstance(value, list) and len(value) == 2 and all(map(number, value))
        and -90 <= value[0] <= 90 and -180 <= value[1] <= 180
    )


def positions(value) -> bool:
    return isinstance(value, list) and all(map(coordinate, value))


def geometry(data: dict) -> None:
    if not positions(data.get("gps_points")):
        raise APIError("The API returned missing or invalid GPS coordinates.")
    bounds = data.get("bounds")
    if bounds is not None and (not positions(bounds) or len(bounds) not in (0, 2)):
        raise APIError("The API returned invalid map bounds.")


def require(data: dict, fields: dict) -> None:
    if not isinstance(data, dict) or any(
        type(data.get(key)) is not kind for key, kind in fields.items()
    ):
        raise APIError("The API returned an unexpected response. Check the backend version.")


def check_numbers(data: dict, fields: tuple[str, ...]) -> None:
    if any(key in data and data[key] is not None and not number(data[key]) for key in fields):
        raise APIError("The API returned a malformed numeric field.")


METRICS = (
    "top1_accuracy", "top1_accuracy_classifier_only", "hmm_improvement_pp",
    "route_continuity", "route_precision", "route_recall", "route_f1",
)
TIMINGS = ("features_ms", "classifier_ms", "hmm_ms", "scoring_ms", "total_ms")


class APIClient:
    def __init__(self, base_url: str):
        self.base_url = normalize_url(base_url)

    def _request(self, method: str, path: str, *, body=None, timeout=10) -> dict:
        try:
            response = requests.request(
                method, self.base_url + path, json=body, timeout=(4, timeout),
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            raise APIError("The request timed out. Check the network and the Laptop 1 logs, then retry.") from exc
        except requests.RequestException as exc:
            raise APIError("Could not reach Laptop 1. Check the URL, server, network and firewall.") from exc

        try:
            data = response.json()
        except ValueError:
            data = None
        if not 200 <= response.status_code < 300:
            detail = data.get("detail") if isinstance(data, dict) else None
            if isinstance(detail, list):
                detail = "; ".join(
                    str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                    for item in detail
                )
            fallback = {404: "Trip or endpoint not found.", 422: "Invalid request.",
                        500: "Prediction failed. Check the Laptop 1 logs."}
            raise APIError(f"HTTP {response.status_code} — {detail or fallback.get(response.status_code, response.reason)}")
        require(data, {})
        return data

    def health(self) -> dict:
        data = self._request("GET", "/health")
        require(data, {"status": str, "model": str, "eval_set": str, "n_trips": int})
        if data["status"] != "ok":
            raise APIError("The backend reported that it is unavailable.")
        return data

    def trips(self) -> list[dict]:
        data = self._request("GET", "/trips")
        require(data, {"eval_set": str, "trips": list})
        for trip in data["trips"]:
            require(trip, {"trip_id": str, "n_points": int})
            check_numbers(trip, ("duration_s",))
            if not trip["trip_id"]:
                raise APIError("The API returned an empty trip ID.")
        if len({trip["trip_id"] for trip in data["trips"]}) != len(data["trips"]):
            raise APIError("The API returned duplicate trip IDs.")
        return data["trips"]

    def track(self, trip_id: str) -> dict:
        data = self._request("GET", f"/trips/{quote(trip_id, safe='')}/track")
        self._check_trip(data, trip_id)
        return data

    def predict(self, trip_id: str) -> dict:
        data = self._request("POST", "/predict", body={"trip_id": trip_id}, timeout=90)
        self._check_trip(data, trip_id)
        require(data, {"request_id": str, "predicted_route": list})
        for key in ("predicted_route", "classifier_only_route"):
            route = data.get(key)
            if route is not None and (
                not isinstance(route, list) or not all(positions(line) for line in route)
            ):
                raise APIError("The API returned invalid route coordinates.")
        check_numbers(data, ("duration_s", "n_route_segments", "route_length_m"))
        for key, fields in (("metrics", METRICS), ("timings", TIMINGS)):
            if data.get(key) is not None:
                require(data[key], {})
                check_numbers(data[key], fields)
        return data

    @staticmethod
    def _check_trip(data: dict, trip_id: str) -> None:
        require(data, {"trip_id": str, "n_points": int})
        if data["trip_id"] != trip_id:
            raise APIError("The API returned a different trip. Refresh the trip list and retry.")
        geometry(data)
