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
| 4 | Train and compare at least 5 models | Done: `notebooks/03_models.ipynb` (7 models) |
| 5 | Select the best model | Done: HistGradientBoosting (`03_models.ipynb` section 6) |
| 6 | Bayesian hyperparameter optimisation | Not started |
| 7 | HMM and Viterbi route reconstruction | Not started |
| 8 | Export model and artifacts | Not started |
| 9 | FastAPI backend (Laptop 1) | Not started |
| 10 | Evaluator UI (Laptop 2) | Not started |
| 11 | Integration testing | Not started |
| 12 | Technical report | Not started |

## Repository layout

```
pathsnap/
├── src/
│   ├── data_io.py        # raw file loading, per-trip UTM projection, nearest-segment queries
│   ├── labels.py         # per-point ground-truth arc via order-preserving alignment to .route
│   ├── candidates.py     # road graph per trip + candidate arcs (radius R, nearest K)
│   ├── features.py       # 26 features per (GPS point, candidate arc) pair
│   └── viz.py            # Folium map helpers (GPS, road network, reference/predicted routes)
├── scripts/
│   ├── sweep_candidates.py   # miss rate vs (R, K) on train
│   └── build_features.py     # builds outputs/features/{train,student_test}.parquet
├── notebooks/
│   ├── 01_eda.ipynb      # dataset exploration (train + student_test partitions)
│   └── 02_features.ipynb # labelling, candidate generation, features, baselines
├── outputs/
│   └── eda/
│       ├── figures/      # PNG figures used in the report
│       ├── maps/         # interactive Folium HTML maps
│       ├── trip_summary.csv
│       └── trip_flags.csv
│   └── features/         # sweep results, config.json, figures, maps (parquet tables are rebuilt, not committed)
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

## Data partitions and leakage rules

| Partition | Trips | Used for |
|---|---|---|
| `train` | 80 | fitting, group-aware cross-validation, Bayesian optimisation |
| `student_test` | 10 | development-stage model comparison |
| evaluator test | 10 | final evaluation only; never used during development |

Ground-truth routes for evaluation trips stay server-side and are never sent to the model or the UI.

## AI assistance

Parts of this project were developed with help from an AI coding assistant (Claude Code).
All experiments were run by the team, and every team member can explain the code and results.
