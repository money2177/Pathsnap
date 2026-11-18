# PathSnap API: contract for the UI (Laptop 2)

The API runs on **Laptop 1** and does all the work: candidate generation, features, the classifier, HMM/Viterbi decoding and
scoring. The UI only picks a trip, calls the API, and draws what comes back.

- **Base URL:** `http://<LAPTOP-1-IP>:8000`. Laptop 1 prints its address on start-up. Make it a setting in the UI; never hardcode it.
- **Interactive docs:** `http://<LAPTOP-1-IP>:8000/docs`. Every endpoint can be tried from the browser there.
- **Formats:** JSON, CORS enabled for any origin. Coordinates are always **`[lat, lon]`** in degrees. Times are seconds,
  durations are milliseconds where the name ends in `_ms`, and metrics are fractions from 0 to 1.
- **Real example responses:** in [`examples/`](examples/). You can build the UI against them before the API is reachable.

## Evaluation-day flow

```
UI start         GET  /health                  -> show "API connected" (or a clear "API unreachable" message)
                 GET  /trips                   -> fill the trip picker with the 10 trips
user picks trip  GET  /trips/{trip_id}/track   -> (optional) show the raw GPS points right away
user clicks Run  POST /predict {"trip_id"}     -> draw routes and show metrics. Every click must be a new live call.
```

Rules from the hackathon guidelines:
- **No hardcoded or cached results.** Every result shown must come from a live `/predict` call; the evaluator checks the logs on Laptop 1.
- **No ground truth in the UI.** The API never sends the reference route. Don't try to load it from anywhere else.

---

## `GET /health`
```json
{ "status": "ok", "model": "HistGradientBoostingClassifier (tuned) + HMM/Viterbi", "artifact_version": 1,
  "artifacts_created": "2026-11-18 15:26:49", "eval_set": "student_test", "n_trips": 10, "scoring_available": true }
```
`eval_set` is the name of the folder being served. On evaluation day it changes to the reserved trips.

## `GET /trips`
```json
{ "eval_set": "student_test",
  "trips": [ { "trip_id": "00000000", "n_points": 2503, "duration_s": 2503.0,
               "start": [56.251559, 54.963519], "end": [56.083507, 54.250254] }, ... ] }
```
Use `trip_id` exactly as given (it is a string with leading zeros).

## `GET /trips/{trip_id}/track`
Raw GPS only, which is cheap and needs no model:
```json
{ "trip_id": "00000052", "n_points": 262, "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
  "gps_points": [[56.79915, 35.858383], ...], "t": [0.0, 1.0, ...] }
```

## `POST /predict`
Request body: `{"trip_id": "00000052"}`. A large trip takes about 0.2–1 s; show a spinner.

| Field | Type | Meaning / how to show it |
|---|---|---|
| `request_id` | str | ID of this run; it also appears in the Laptop 1 log line (handy to show the evaluator they match) |
| `trip_id`, `eval_set` | str | echo of the request |
| `n_points` | int | number of GPS observations |
| `n_candidate_rows` | int | (point, candidate road) pairs the classifier scored |
| `n_route_segments`, `route_length_m` | int, float | size of the reconstructed route |
| `duration_s` | float | trip duration |
| `bounds` | `[[lat,lon],[lat,lon]]` | pass to the map's `fitBounds` |
| `gps_points` | `[[lat,lon], ...]` | raw GPS observations in time order. Draw them as small dots. |
| `matched_segment` | `[int, ...]` | road segment id chosen for each GPS point (same order as `gps_points`) |
| `match_probability` | `[float, ...]` | classifier confidence of each chosen segment (could colour the dots) |
| **`predicted_route`** | `[[[lat,lon], ...], ...]` | **the final route (classifier + HMM)**, as a list of polylines. Usually one; more if the route has a break. Draw it as the main line. |
| `classifier_only_route` | same | route from the classifier alone, without the HMM. Draw as an optional comparison layer. |
| `metrics` | object or `null` | see below. `null` means there is no reference route on the server, so the trip was not scored. |
| `timings` | object | `features_ms`, `classifier_ms`, `hmm_ms`, `scoring_ms`, `total_ms` |
| `decoder` | object | `gap_splits`, `break_steps`, `invalid_transitions_used`, `unjoinable_transitions`, `network_connectivity` (diagnostics, optional to show) |

`metrics` (computed on Laptop 1 against its internal reference):

| Field | Show as |
|---|---|
| `top1_accuracy` | **Point accuracy (HMM)**, the headline number, e.g. "98.1 %" |
| `top1_accuracy_classifier_only` | Point accuracy without the HMM |
| `hmm_improvement_pp` | e.g. "+1.15 pp from HMM" |
| `route_continuity` | share of consecutive points on connected roads |
| `route_precision`, `route_recall`, `route_f1` | route-level overlap with the reference (length-weighted) |

## Errors
Errors come back as `{"detail": "message"}`:
- `404`: unknown `trip_id` (the message lists the valid ones)
- `422`: missing or invalid request body
- `500`: the pipeline failed (the message says why; the full trace is in the Laptop 1 log)
- connection refused or timeout: the API is not reachable. See "Connecting the laptops" in the main README.

## Suggested map layers
| Layer | Data | Style |
|---|---|---|
| GPS observations | `gps_points` | small blue dots |
| Predicted route (HMM) | `predicted_route` | thick orange line, shown by default |
| Classifier-only route | `classifier_only_route` | thin purple line, off by default |
| Start / end | first and last of `gps_points` | green / red markers |

Include a legend and layer toggles; the report's map requirements ask for both.
