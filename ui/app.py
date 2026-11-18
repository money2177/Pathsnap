"""PathSnap's live evaluator workspace. Run with streamlit run ui/app.py."""
import streamlit as st
from streamlit import config

for option, value in {
    "theme.base": "dark",
    "theme.primaryColor": "#38BDF8",
    "theme.backgroundColor": "#0B0F14",
    "theme.secondaryBackgroundColor": "#151B23",
    "theme.textColor": "#F1F5F9",
}.items():
    if not config.is_manually_set(option):
        config.set_option(option, value)

from streamlit_folium import st_folium

from api_client import APIClient, APIError
from map_utils import build_map

DEFAULT_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="PathSnap | Route evaluator", page_icon="📍", layout="wide")
st.markdown("""
<style>
.stApp {background:#0B0F14;color:#F1F5F9}
.st-key-heading h1 {font-size:1.7rem;letter-spacing:-.04em}
.st-key-controls {background:#151B23;border:1px solid #293241;border-radius:10px;padding:18px}
.st-key-empty {border:1px dashed #293241;border-radius:10px;padding:64px 24px;text-align:center}
.st-key-workspace h3 {font-size:1.1rem}
.st-key-workspace iframe {border-radius:10px}
.st-key-heading p,.st-key-empty p {color:#94A3B8}
</style>
""", unsafe_allow_html=True)



def clear_trip() -> None:
    st.session_state.update(prediction=None, track=None, track_error=None,
                            track_loaded=False, run_error=None)


def connect(client: APIClient) -> None:
    state = st.session_state
    try:
        health = client.health()
        state.health = health
        state.trip_list = client.trips()
    except APIError as exc:
        state.health = None
        state.trip_list = []
        state.connection_error = str(exc)
        clear_trip()
    else:
        state.connection_error = None


def formatted(data: dict, key: str, pattern: str, scale: float = 1) -> str:
    value = data.get(key)
    return "—" if value is None else format(value * scale, pattern)


def show_results(result: dict) -> None:
    st.subheader("Evaluation results")
    metrics = result.get("metrics")
    if metrics is None:
        if "metrics" in result:
            st.info("Prediction complete. No reference available on the server, so this trip is unscored.")
        else:
            st.info("Prediction complete. The API did not return scoring information.")
    else:
        main, raw, delta = st.columns(3)
        main.metric("Point Accuracy (Top-1) with HMM", formatted(metrics, "top1_accuracy", ".2%"))
        raw.metric("Classifier-only Top-1", formatted(metrics, "top1_accuracy_classifier_only", ".2%"))
        improvement = formatted(metrics, "hmm_improvement_pp", "+.2f")
        delta.metric("HMM improvement", improvement if improvement == "—" else f"{improvement} pp")
        for column, (label, key) in zip(st.columns(4), (
            ("Route continuity", "route_continuity"), ("Route precision", "route_precision"),
            ("Route recall", "route_recall"), ("Route F1", "route_f1"),
        )):
            column.metric(label, formatted(metrics, key, ".2%"))

    st.subheader("Trip and processing details")
    trip, timing = st.columns(2)
    with trip:
        st.write("**Trip ID**", result["trip_id"])
        st.write("**GPS observations**", formatted(result, "n_points", ",d"))
        st.write("**Duration (s)**", formatted(result, "duration_s", ",.1f"))
        st.write("**Predicted route segments**", formatted(result, "n_route_segments", ",.0f"))
        st.write("**Route length (km)**", formatted(result, "route_length_m", ",.2f", 0.001))
    with timing:
        timings = result.get("timings") or {}
        for label, key in (
            ("Feature generation", "features_ms"), ("Classifier inference", "classifier_ms"),
            ("HMM/Viterbi decoding", "hmm_ms"), ("Scoring", "scoring_ms"), ("Total", "total_ms"),
        ):
            st.write(f"**{label} (ms)**", formatted(timings, key, ",.1f"))
    st.text(f"Request ID  {result['request_id']}")
    st.caption("Matches the prediction request in the backend logs on Laptop 1.")


state = st.session_state
with st.container(key="heading"):
    brand, status, settings = st.columns([6, 2, 2], vertical_alignment="center")
    with brand:
        st.title("PathSnap")
        st.caption("Route Evaluation Dashboard")
    status_slot = status.empty()
with settings.popover("Settings", icon=":material/settings:", width="stretch"):
    st.subheader("Backend connection")
    address = st.text_input("API base URL", DEFAULT_URL, help="Use the LAN address printed on Laptop 1.")
    changed_backend = state.get("backend_address") != address
    if changed_backend:
        state.update(backend_address=address, health=None, trip_list=[], connection_error=None)
        state.pop("trip_picker", None)
        clear_trip()

    check = st.button("Check Connection", width="stretch")
    try:
        client = APIClient(address)
    except APIError as exc:
        client = None
        state.connection_error = str(exc)
    if client and (changed_backend or check):
        with st.spinner("Connecting to Laptop 1…"):
            connect(client)

with st.container(key="workspace"):
    controls, canvas = st.columns([0.28, 0.72], gap="large")
with controls, st.container(key="controls"):
    st.subheader("Evaluation trip")
    # Refresh is drawn below the picker but handled before the widget is constructed.
    picker_area = st.container()
    st.divider()
    run_area = st.container()
    refresh = st.button("Refresh Trip List", width="stretch", disabled=client is None)
    if refresh and client and not (changed_backend or check):
        with st.spinner("Refreshing trips…"):
            connect(client)

    trip_by_id = {trip["trip_id"]: trip for trip in state.trip_list}
    if state.get("trip_picker") not in trip_by_id:
        state.pop("trip_picker", None)
    with picker_area:
        trip_id = st.selectbox("Evaluation trip", list(trip_by_id), key="trip_picker",
                               disabled=not trip_by_id)
        if trip_id:
            trip = trip_by_id[trip_id]
            st.caption(f"Trip {trip_id} · {trip['n_points']:,} observations")
            st.caption(f"Duration {formatted(trip, 'duration_s', ',.1f')} s")
        elif state.health:
            st.caption("No trips returned. Check the evaluation folder on Laptop 1 and refresh.")
        else:
            st.caption("Open Settings to connect to Laptop 1 and load its trips.")
    if state.get("active_trip") != trip_id:
        state.active_trip = trip_id
        clear_trip()
    with run_area:
        run = st.button("Run Map-Matching", type="primary", width="stretch",
                        disabled=not (client and state.health and trip_id))

    if state.health:
        st.caption("🟢 Connected")
        st.text(state.health["model"])
        st.text(f"Evaluation set  {state.health['eval_set']}")
        st.caption(f"{state.health['n_trips']} available trips")
    else:
        st.caption("🔴 Disconnected")
        if state.connection_error:
            st.markdown("**Backend unavailable**")
            st.caption(state.connection_error)
            with st.expander("Connection troubleshooting"):
                st.write("Check the API URL in Settings. Confirm FastAPI is running on Laptop 1 and that its firewall allows the API port.")

status_slot.badge("Connected" if state.health else "Disconnected",
                  color="green" if state.health else "red")

if client and state.health and trip_id:
    if run:
        state.prediction = None
        state.run_error = None
        try:
            with st.spinner(f"Running live map matching for trip {trip_id}…"):
                state.prediction = client.predict(trip_id)
        except APIError as exc:
            state.run_error = str(exc)
    if not state.track_loaded and state.prediction is None:
        state.track_loaded = True
        try:
            with st.spinner("Loading GPS observations…"):
                state.track = client.track(trip_id)
        except APIError as exc:
            state.track_error = str(exc)

with canvas:
    if state.run_error:
        st.error(state.run_error)
    result = state.prediction
    data = result if result is not None else state.track
    st.subheader("Route visualization")
    if result is not None:
        st.caption(f"Live prediction · Trip {trip_id}")
        if not any(len(line) >= 2 for line in result["predicted_route"]):
            st.warning("The API returned no reconstructed route to draw.")
    elif trip_id:
        st.caption("Input GPS observations · Run Map-Matching to generate a prediction.")
        if state.track_error:
            st.warning(state.track_error)
    else:
        with st.container(key="empty"):
            st.markdown("### Select a trip to begin")
            st.write("Connect in Settings, then choose an evaluation trip.")
            st.caption("GPS observations and reconstructed routes will appear here.")
    if data is not None and not data["gps_points"]:
        st.warning("The API returned no GPS observations for this trip.")

    if data is not None:
        st_folium(build_map(data), width="100%", height=580, returned_objects=[],
              key=f"route-map-{address}-{trip_id}-{result['request_id'] if result else 'track'}")
    if result is not None:
        show_results(result)
