# PathSnap React evaluator

This frontend connects to the existing FastAPI service. OpenAI Codex assisted with implementation and verification.

## Run locally

Requires Node.js 20.19+ or 22.12+ and pnpm.

```powershell
cd A:\aditi\ml_hackathon\Pathsnap\ui\web
pnpm install
pnpm dev
```

Open http://127.0.0.1:5173. In Settings, enter the API base URL. The default is http://127.0.0.1:8000.
For two laptops, start FastAPI on Laptop 1 with `python -m api`, then use its printed LAN URL.
Confirm /health opens from Laptop 2. Both laptops must share a reachable network and the API port must be allowed through the firewall.
Browser requests use the existing backend CORS configuration. An HTTPS frontend needs an HTTPS API or a deployment proxy.

```powershell
pnpm test
pnpm build
pnpm preview
```

## Load the evaluator set

On the backend machine, set `PATHSNAP_EVAL_DIR` to the extracted folder containing the
numbered trip directories. For the supplied archive, select the inner `evaluator_test`
folder. The `__MACOSX` directory and files starting with `._` are archive metadata.

```powershell
$env:PATHSNAP_EVAL_DIR = 'C:\path\to\evaluator_test\evaluator_test'
python -m api
```

After restarting the backend, click **Refresh trip list**. The dashboard shows the
dataset name and trip count returned by the backend. Confirm **evaluator_test · 10 trips**
before running predictions on the supplied set.

## Evaluation behaviour

Select a trip to preview its GPS observations. Every Run action sends a fresh POST /predict.
Responses are held only for the current display and cleared when changing the backend or trip.
All accuracy and evaluation statistics originate on the backend. Unscored trips still render.
The UI does not read datasets or reference routes. Offline example responses are used only in tests.

Map layers preserve independent route segments. API latitude/longitude coordinates are converted to GeoJSON longitude/latitude at the visualization boundary.
OpenStreetMap raster tiles provide a key-free basemap with attribution. They require internet access. A tile error leaves route overlays on the neutral canvas.
The interactive map requires WebGL. Globe and synchronized split views are deferred enhancements.

The existing Streamlit evaluator remains available with `streamlit run ui/app.py` from the repository root.
