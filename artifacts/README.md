# PathSnap inference artifacts (stage 8)

Everything the API (Laptop 1) needs to turn a raw trip (`.track`, `.nodes`, `.arcs`) into a matched route.

| File | Content |
|---|---|
| `model/hgb_tuned.joblib` | Tuned HistGradientBoosting (stage 6), fitted on the 78 reliable train trips. Outputs P(candidate arc is the true arc). |
| `config.json` | Candidate generation (50 m, 12 arcs, fallback), the 26 features in model order plus constants, tuned HMM parameters plus constants |
| `manifest.json` | sha256 and size of each file, library versions, provenance, reference student_test metrics |
| `verification.json` | Result of re-running the whole pipeline from these files on the 10 student_test trips |

## Usage

```python
from src.pipeline import PathSnapPipeline, score_prediction
from src.data_io import Trip

pipe = PathSnapPipeline.load("artifacts")             # checks sha256 and code/config consistency
pred = pipe.predict_trip("00000077", "student_test")  # never reads .route
pred["points"]       # one row per GPS point: lon, lat, t, arc_id (HMM), prob, arc_id_classifier
pred["route_arcs"]   # connected route (arc ids)
score_prediction(Trip("00000077", "student_test"), pred)  # server-side scoring against .route
```

Prediction uses `InferenceTrip`, which has no access to the reference route even if the file exists. Scoring is a separate,
server-side step.

## Verification (`python scripts/export_artifacts.py`)
- Loaded fresh from these files, the pipeline reproduces stage 7 exactly: student_test Top-1 **0.9261** (classifier only 0.8814).
  The decoded arc is identical for all 19,160 points.
- About 0.4 s per trip (features, then classifier, then Viterbi, then route).
- It gives the same output when the trip folder has no `.route` file.

Load with scikit-learn 1.9.1 (pinned in `requirements.txt`), because joblib model files are version-specific.
