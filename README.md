# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon (PES University, RR Campus).

PathSnap reconstructs the road route a vehicle actually drove from noisy GPS observations:
a classifier scores candidate road segments for every GPS point, and an HMM/Viterbi decoder
turns those scores into a connected route. The model and API run on one laptop and the
evaluator UI on another.

## Status

| # | Stage (official work process) | Status |
|---|---|---|
| 1 | Understand the problem | Done |
| 2 | Explore the raw dataset | Done: `notebooks/01_eda.ipynb` |
| 3 | Data processing and feature engineering | Done: `notebooks/02_features.ipynb` |
| 4 | Train and compare at least 5 models | Done: `notebooks/03_models.ipynb` (8 models) |
| 5 | Select the best model | Done: HistGradientBoosting (`03_models.ipynb` section 7) |
| 6 | Bayesian hyperparameter optimisation | Done: `notebooks/04_tuning.ipynb` (HGB tuned; LightGBM search for comparison) |
| 7 | HMM and Viterbi route reconstruction | Done: `notebooks/05_hmm.ipynb` (Top-1 0.8814 → 0.9261, route F1 0.82 → 0.96 on student_test) |
| 8 | Export model and artifacts | Done: `artifacts/` (verified: reproduces student_test Top-1 0.9261 from a fresh load) |
| 9 | FastAPI backend (Laptop 1) | Done: `api/` (contract for the UI: `api/API.md`; 20 tests in `tests/`) |
| 10 | Evaluator UI (Laptop 2) | Done: `ui/web` (React, used in the demo) and `ui/app.py` (Streamlit) |
| 11 | Integration testing | Done: two laptops on one network, live `/predict` requests logged on Laptop 1 |
| 12 | Technical report | Done: `11_Report.pdf` |

## Results (student_test, 10 trips, 19,160 GPS points)

| System | Top-1 accuracy | Route continuity | Route F1 |
|---|---|---|---|
| Rule: nearest arc + heading | 0.8326 | – | – |
| HistGradientBoosting, untuned | 0.8722 | 0.9599 | – |
| HistGradientBoosting, tuned (classifier only) | 0.8814 | 0.9685 | 0.8179 |
| **Tuned HistGradientBoosting + HMM/Viterbi (final)** | **0.9261** | **0.9948** | **0.9619** |

Top-1 = share of GPS points matched to the correct road segment. The final pipeline is exported in `artifacts/`
and reproduces these numbers exactly from a fresh load (`artifacts/verification.json`).

## Repository layout

```
pathsnap/
├── src/
│   ├── data_io.py        # raw file loading, per-trip UTM projection, nearest-segment queries
│   ├── labels.py         # per-point ground-truth arc via order-preserving alignment to .route
│   ├── candidates.py     # road graph per trip + candidate arcs (radius R, nearest K)
│   ├── features.py       # 26 features per (GPS point, candidate arc) pair
│   ├── evaluation.py     # Top-1, candidate miss rate, route continuity, row-level metrics
│   ├── models.py         # the 8 compared model definitions
│   ├── hmm.py            # HMM states/emissions/transitions, Viterbi, route reconstruction
│   ├── pipeline.py       # end-to-end inference from artifacts/ + server-side scoring
│   └── viz.py            # Folium map helpers (GPS, road network, reference/predicted routes)
├── scripts/
│   ├── sweep_candidates.py   # miss rate vs (R, K) on train
│   ├── build_features.py     # builds outputs/features/{train,student_test}.parquet
│   ├── train_compare.py      # stage 4: 5-fold GroupKFold comparison of 8 models + student_test
│   ├── tune_hgb.py           # stage 6: Optuna TPE tuning of HistGradientBoosting
│   ├── tune_lgbm.py          # stage 6: LightGBM search (comparison)
│   ├── refit_tuned.py        # refits the tuned HGB with the stage-6 parameters (after the tie-break fix)
│   ├── observation_metrics.py # per-GPS-point Top-1, Top-3 and log loss for every model
│   ├── hmm_decode.py         # stage 7: OOF probabilities, decoder tuning, HMM vs classifier
│   ├── zoom_maps.py          # zoomed classifier-only vs HMM Folium maps for the report
│   └── export_artifacts.py   # stage 8: writes and verifies artifacts/
├── notebooks/
│   ├── 01_eda.ipynb      # dataset exploration (train + student_test partitions)
│   ├── 02_features.ipynb # labelling, candidate generation, features, baselines
│   ├── 03_models.ipynb   # model comparison (stage 4) and selection (stage 5)
│   ├── 04_tuning.ipynb   # Bayesian optimisation (stage 6)
│   └── 05_hmm.ipynb      # HMM / Viterbi route reconstruction (stage 7)
├── outputs/
│   ├── eda/              # figures, interactive Folium maps, trip_summary.csv, trip_flags.csv
│   ├── features/         # sweep results, config.json, figures, maps (parquet tables are rebuilt, not committed)
│   ├── models/           # compare/ (stage 4), tuning/ and tuning_lgbm/ (stage 6)
│   └── hmm/              # stage 7: hmm_config.json, comparison tables, figures, Folium maps
├── api/                  # FastAPI backend (Laptop 1): main.py, schemas.py, API.md (UI contract), examples/
├── ui/                   # evaluator UI (Laptop 2): web/ (React + MapLibre), app.py (Streamlit + Folium)
├── tests/                # API tests (pytest)
├── 11_Report.pdf         # technical report
├── artifacts/            # final model, config, manifest, verification (see artifacts/README.md)
├── data/                 # raw dataset (not committed, see below)
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone <repo-url> pathsnap && cd pathsnap
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

The raw dataset is not committed. Put it under `data/` so the layout is:

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
└── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

Alternatively, set the `PATHSNAP_DATA` environment variable to any folder that contains `train/` and `student_test/`:

```bash
export PATHSNAP_DATA="/path/to/Student folder"
```

## Running the EDA

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb   # about 30 s
```

Or open it interactively with `jupyter lab`. Figures, maps and CSV summaries are written to `outputs/eda/`.

## Building the feature tables

```bash
python scripts/sweep_candidates.py   # optional: radius / K sweep on train (about 5 s)
python scripts/build_features.py     # about 10 s; writes outputs/features/*.parquet and config.json
```

Then run `notebooks/02_features.ipynb` the same way as the EDA notebook.

## Models, tuning and route reconstruction

```bash
python scripts/train_compare.py        # stage 4: 8 models, about 10 min
python scripts/tune_hgb.py             # stage 6: HGB search, round 1 (then --stage2 for the refined round)
python scripts/refit_tuned.py          # refit the tuned HGB with the saved best parameters (no new search)
python scripts/hmm_decode.py           # stage 7: about 5 min (OOF probabilities, 60 decoder trials, evaluation)
python scripts/observation_metrics.py  # per-point metrics table used in the report
python scripts/zoom_maps.py            # zoomed comparison maps used in the report
```

Then run notebooks `03` to `05` to rebuild the tables, figures and maps.

```bash
python scripts/export_artifacts.py     # stage 8: export to artifacts/ and verify (about 10 s)
```

## Running the API (Laptop 1)

```bash
source .venv/bin/activate
python -m api                        # serves on all interfaces, port 8000; prints the address for Laptop 2
python -m api --port 8001            # if port 8000 is taken
```

On start-up the server prints its LAN address. It also checks the artifacts' checksums, lists the trips it serves and says
whether reference routes are available for scoring. Every request is logged to the terminal and to `api/logs/api.log`
(client IP, trip, accuracy, timings): this is the log the evaluator wants to see.

Which trips are served is set by an environment variable. On evaluation day, point it at the folder with the reserved trips:

```bash
export PATHSNAP_EVAL_DIR="/path/to/evaluation_trips"   # one sub-folder per trip: <id>/<id>.track, .nodes, .arcs (+ .route for scoring)
python -m api
```

`.route` files are read only by the server-side scoring step; prediction cannot access them (`src/pipeline.InferenceTrip`).

Endpoints: `GET /health`, `GET /trips`, `GET /trips/{id}/track`, `POST /predict`. See [`api/API.md`](api/API.md) for the full
contract and `http://<ip>:8000/docs` for interactive docs.

Tests: `python -m pytest tests/ -q`. They check all 10 student_test trips against the stage-8 verification, that no ground
truth leaks, that bad or path-traversal trip ids are rejected, and that trips without a `.route` file are predicted but unscored.

### Connecting the laptops
1. Put both laptops on the **same network**. Campus Wi-Fi often blocks traffic between devices; a phone hotspot is the reliable fallback.
2. Start the API on Laptop 1 and note the printed address, e.g. `http://192.168.1.23:8000`.
3. From Laptop 2, check it with `curl http://192.168.1.23:8000/health` or open `/docs` in a browser.
4. If it times out: allow incoming connections for Python in Laptop 1's firewall (macOS: System Settings → Network → Firewall).

## Running the UI (Laptop 2)

### React route evaluator

Requires Node.js 20.19+ or 22.12+ and pnpm. From the repository root:

```bash
cd ui/web
pnpm install
pnpm dev
```

Open http://127.0.0.1:5173. In **Settings**, enter the FastAPI base URL printed on Laptop 1.
The default http://127.0.0.1:8000 connects to a backend on the same computer.
For a two-laptop evaluation, use http://<laptop-1-ip>:8000 with the actual LAN address.

Start FastAPI on Laptop 1 using `python -m api`. Both laptops must share a network that allows
device-to-device connections. On Windows, `ipconfig` shows the active adapter's IPv4 address.
From Laptop 2, open the backend's /health endpoint to verify connectivity, then check the
connection in Settings, select a trip, and click **Run map-matching**.

Every Run sends a fresh prediction request. Scores, timings and the request ID come from the
backend. The request ID can be checked in the Laptop 1 logs. Trip or backend changes clear
the previous result. Reference routes and raw datasets are accessed only by the backend.

Run `pnpm test` for frontend contract tests, `pnpm build` for the production build, and
`pnpm preview` to preview the built frontend. Further details are in [ui/web/README.md](ui/web/README.md).

### Streamlit evaluator

From the repository root, using Python 3.10 or newer:

```bash
python -m pip install -r ui/requirements.txt
streamlit run ui/app.py
```

### Connection troubleshooting

If a request fails, confirm the server is running and check its address, port and firewall.
A shared hotspot can help if campus Wi-Fi blocks device-to-device traffic. HTTP 500 and
prediction timeouts should be checked against the Laptop 1 logs. An HTTPS frontend needs
an HTTPS API or an appropriately configured proxy.

The backend verifies the SHA-256 checksum of every artifact before starting. `.gitattributes` stops git from
converting their line endings on Windows; a checkout made before that file was added needs a fresh clone
(or `git rm --cached -r artifacts && git checkout -- artifacts`).

OpenStreetMap basemap tiles require internet access. Route overlays can still appear if
tiles fail. The React map requires WebGL and hardware acceleration in the browser.

**UI AI-assistance disclosure.** OpenAI Codex assisted with the Streamlit and React
evaluator interfaces, API integration, and verification.

## Data partitions and leakage rules

| Partition | Trips | Used for |
|---|---|---|
| `train` | 80 (78 used) | fitting, group-aware cross-validation, Bayesian optimisation, HMM decoder tuning. Trips 00000033 and 00000054 are excluded because their reference routes do not match their GPS (median label distance 122 m and 140 m; every other trip is at most 11 m) |
| `student_test` | 10 | development-stage model comparison and reporting; never used to choose models or parameters |
| evaluator test | 10 | final evaluation only. Not used for any development decision; loaded once, after the pipeline was frozen, for the two-laptop integration run. Never committed (`.gitignore`) |

Ground-truth routes for evaluation trips stay server-side and are never sent to the model or the UI.

## AI-assistance disclosure

AI coding assistants were used in this project: Claude Code for the data pipeline, model training and evaluation
code, the HMM decoder, the API and drafting parts of the documentation and report, and OpenAI Codex for the evaluator
UI. All experiments were run by the team, every reported number is produced by code in this repository and can be
regenerated from it, and the team reviewed the code, methods and results and can explain them.


