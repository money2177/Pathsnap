"""Stage 8: export the final model and every inference asset to artifacts/, then verify them.

artifacts/
    model/hgb_tuned.joblib   tuned HistGradientBoosting (stage 6), fitted on the 78 reliable train trips
    config.json              candidate generation, feature list + constants, HMM parameters + constants
    manifest.json            sha256 of each file, library versions, provenance and reference metrics
    verification.json        result of re-running the full pipeline from the artifacts on student_test

The verification step loads the artifacts in a clean pipeline (src/pipeline.py), predicts all 10
student_test trips from the raw files without the .route file, then scores them server-side.
It must reproduce the stage-7 student_test numbers exactly.

Run: python scripts/export_artifacts.py
"""
import json, os, platform, shutil, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import joblib, lightgbm, optuna, pyproj, scipy, shapely, sklearn
from src import features as F, hmm as H
from src.candidates import generate_candidates
from src.data_io import REPO_ROOT, Trip, list_trips
from src.pipeline import ARTIFACT_VERSION, PathSnapPipeline, score_prediction, sha256

ART = os.path.join(REPO_ROOT, "artifacts")
SRC_MODEL = os.path.join(REPO_ROOT, "outputs", "models", "tuning", "hgb_tuned_stage2.joblib")
TUNING = os.path.join(REPO_ROOT, "outputs", "models", "tuning", "tuning_result_stage2.json")
HMMCFG = os.path.join(REPO_ROOT, "outputs", "hmm", "hmm_config.json")
FEATCFG = os.path.join(REPO_ROOT, "outputs", "features", "config.json")

if __name__ == "__main__":
    os.makedirs(f"{ART}/model", exist_ok=True)
    shutil.copyfile(SRC_MODEL, f"{ART}/model/hgb_tuned.joblib")
    tun, hmm, fc = json.load(open(TUNING)), json.load(open(HMMCFG)), json.load(open(FEATCFG))
    k_fallback = generate_candidates.__defaults__[2]

    config = {
        "artifact_version": ARTIFACT_VERSION,
        "model": {"file": "model/hgb_tuned.joblib", "class": "sklearn.ensemble.HistGradientBoostingClassifier",
                  "params": tun["best_params"], "output": "predict_proba[:, 1] = P(candidate arc is the true arc)"},
        "candidates": {"radius_m": fc["radius_m"], "k": fc["k"], "k_fallback": k_fallback,
                       "fallback_radii_m": [4 * fc["radius_m"], 20 * fc["radius_m"]], "then": "single nearest arc"},
        "features": {"names": F.FEATURES, "constants": {"SIGMA_M": F.SIGMA_M, "MIN_MOVE_M": F.MIN_MOVE_M}},
        "hmm": {"params": hmm["hmm_params"],
                "constants": {"BACKWARD_TOL_M": H.BACKWARD_TOL_M, "ROUTE_FACTOR": H.ROUTE_FACTOR,
                              "MIN_SEARCH_M": H.MIN_SEARCH_M, "MAX_SEARCH_M": H.MAX_SEARCH_M},
                "hidden_states": "candidate arcs of each GPS point",
                "emission": "emission_weight * log(max(p, prob_floor))",
                "transition": "-|network route distance - straight-line distance| / beta - uturn_cost * [reverse twin]; "
                              "unreachable -> -break_cost; time gap > gap_s -> flat (chain split)"},
        "projection": "per-trip WGS84 UTM zone (src/data_io.utm_epsg)",
    }
    json.dump(config, open(f"{ART}/config.json", "w"), indent=2)

    manifest = {
        "artifact_version": ARTIFACT_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {rel: {"sha256": sha256(f"{ART}/{rel}"), "bytes": os.path.getsize(f"{ART}/{rel}")}
                  for rel in ["model/hgb_tuned.joblib", "config.json"]},
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "shapely": shapely.__version__, "pyproj": pyproj.__version__,
                        "scipy": scipy.__version__, "joblib": joblib.__version__},
        "note": "load with the same scikit-learn version (joblib pickles are version-specific)",
        "provenance": {
            "training_data": "train partition, 78 reliable trips (trips with median label distance > 30 m excluded), "
                             "every 3rd GPS point with label within 50 m",
            "model_selection": "stage 4-5, notebooks/03_models.ipynb", "tuning": "stage 6, scripts/tune_hgb.py --stage2",
            "decoder_tuning": "stage 7, scripts/hmm_decode.py (train out-of-fold probabilities only)",
            "evaluator_test_partition_used": False},
        "reference_metrics_student_test": {
            "classifier_only_top1": hmm["results"]["student_test/classifier_only"]["top1_accuracy"],
            "hmm_top1": hmm["results"]["student_test/hmm_tuned"]["top1_accuracy"],
            "hmm_route_continuity": hmm["results"]["student_test/hmm_tuned"]["route_continuity"],
            "hmm_route_f1": hmm["results"]["student_test/hmm_tuned"]["route_f1"]},
    }
    json.dump(manifest, open(f"{ART}/manifest.json", "w"), indent=2)
    print("exported:", {k: v["bytes"] for k, v in manifest["files"].items()})

    # ---------- verification: fresh load, raw files in, server-side scoring ----------
    t0 = time.perf_counter(); pipe = PathSnapPipeline.load(ART); load_s = time.perf_counter() - t0
    decoded = pd.read_parquet(os.path.join(REPO_ROOT, "outputs", "hmm", "student_test_decoded.parquet"))
    rows, n_pts, n_ok, n_ok_clf, same = [], 0, 0, 0, 0
    for tid in list_trips("student_test"):
        pred = pipe.predict_trip(tid, "student_test")
        s = score_prediction(Trip(tid, "student_test"), pred)
        ref = decoded[decoded.trip == tid].sort_values("point_idx")
        same += int((ref.arc_hmm.to_numpy() == pred["points"].arc_id.to_numpy()).sum())
        n = pred["n_points"]; n_pts += n
        n_ok += s["top1_accuracy"] * n; n_ok_clf += s["top1_accuracy_classifier_only"] * n
        rows.append(dict(trip=tid, n_points=n, top1=s["top1_accuracy"], top1_classifier=s["top1_accuracy_classifier_only"],
                         route_continuity=s["route_continuity"], route_f1=s["route_f1"], **pred["timings"]))
        print(f"{tid}: top1={s['top1_accuracy']:.4f} routeF1={s['route_f1']:.4f} {pred['timings']['total_s']:.2f}s")
    pt = pd.DataFrame(rows)
    ver = {"load_seconds": load_s, "trips": len(pt), "points": n_pts,
           "top1_hmm": n_ok / n_pts, "top1_classifier_only": n_ok_clf / n_pts,
           "expected_top1_hmm": manifest["reference_metrics_student_test"]["hmm_top1"],
           "expected_top1_classifier_only": manifest["reference_metrics_student_test"]["classifier_only_top1"],
           "decoded_arcs_identical_to_stage7": same / n_pts,
           "mean_seconds_per_trip": float(pt.total_s.mean()), "max_seconds_per_trip": float(pt.total_s.max()),
           "per_trip": pt.to_dict(orient="records")}
    ver["passed"] = bool(abs(ver["top1_hmm"] - ver["expected_top1_hmm"]) < 1e-9 and
                         abs(ver["top1_classifier_only"] - ver["expected_top1_classifier_only"]) < 1e-9 and
                         ver["decoded_arcs_identical_to_stage7"] == 1.0)
    json.dump(ver, open(f"{ART}/verification.json", "w"), indent=2, default=float)
    print(f"verification: top1 {ver['top1_hmm']:.4f} (expected {ver['expected_top1_hmm']:.4f}), classifier "
          f"{ver['top1_classifier_only']:.4f}, identical arcs {ver['decoded_arcs_identical_to_stage7']:.4f}, "
          f"{ver['mean_seconds_per_trip']:.2f}s/trip -> {'PASSED' if ver['passed'] else 'FAILED'}")
