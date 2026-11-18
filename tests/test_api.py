"""API tests: python -m pytest tests/ -q"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from fastapi.testclient import TestClient

import api.main as M
from src.data_io import REPO_ROOT, Trip

VERIFY = {r["trip"]: r for r in json.load(open(os.path.join(REPO_ROOT, "artifacts", "verification.json")))["per_trip"]}


@pytest.fixture(scope="module")
def client():
    with TestClient(M.app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["n_trips"] == 10 and r["scoring_available"]


def test_trips(client):
    r = client.get("/trips").json()
    assert [t["trip_id"] for t in r["trips"]] == sorted(VERIFY)
    assert all(t["n_points"] > 0 and len(t["start"]) == 2 for t in r["trips"])


def test_track(client):
    r = client.get("/trips/00000077/track").json()
    assert r["n_points"] == len(r["gps_points"]) == len(r["t"])


@pytest.mark.parametrize("tid", sorted(VERIFY))
def test_predict_matches_verified_pipeline(client, tid):
    r = client.post("/predict", json={"trip_id": tid})
    assert r.status_code == 200
    d = r.json()
    assert d["n_points"] == len(d["gps_points"]) == len(d["matched_segment"]) == len(d["match_probability"])
    assert d["metrics"]["top1_accuracy"] == pytest.approx(VERIFY[tid]["top1"], abs=1e-12)
    assert d["metrics"]["top1_accuracy_classifier_only"] == pytest.approx(VERIFY[tid]["top1_classifier"], abs=1e-12)
    assert d["predicted_route"] and all(len(p) >= 2 for p in d["predicted_route"])


def test_no_ground_truth_in_response(client):
    d = client.post("/predict", json={"trip_id": "00000077"}).json()
    text = json.dumps(d)
    assert "reference" not in text and "point_correct" not in text and "true_arc" not in text
    ref = Trip("00000077", "student_test").route
    lines = d["predicted_route"]
    assert lines != ref.tolist()


@pytest.mark.parametrize("bad", ["99999999", "../train/00000001", "", "00000077/../../x"])
def test_unknown_trip_rejected(client, bad):
    assert client.post("/predict", json={"trip_id": bad}).status_code == 404


def test_missing_body(client):
    assert client.post("/predict", json={}).status_code == 422


def test_trip_without_reference_is_predicted_but_unscored(tmp_path, monkeypatch):
    d = tmp_path / "eval_no_route" / "00000077"
    d.mkdir(parents=True)
    for ext in ("track", "nodes", "arcs"):
        shutil.copy(os.path.join(REPO_ROOT, "data", "student_test", "00000077", f"00000077.{ext}"), d)
    monkeypatch.setattr(M, "EVAL_DIR", str(tmp_path / "eval_no_route"))
    monkeypatch.setattr(M, "ROOT", str(tmp_path))
    monkeypatch.setattr(M, "EVAL_SET", "eval_no_route")
    with TestClient(M.app) as c:
        assert c.get("/health").json()["scoring_available"] is False
        r = c.post("/predict", json={"trip_id": "00000077"}).json()
        assert r["metrics"] is None and r["n_points"] > 0 and r["predicted_route"]
