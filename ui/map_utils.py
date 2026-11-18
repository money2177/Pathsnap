"""Build a Folium map using only geometry returned by the PathSnap API."""
import folium

GPS = "#3b82f6"
HMM = "#f97316"
CLASSIFIER = "#a855f7"


def build_map(data: dict | None = None) -> folium.Map:
    data = data or {}
    points = data.get("gps_points", [])
    hmm = [line for line in (data.get("predicted_route") or []) if len(line) >= 2]
    classifier = [line for line in (data.get("classifier_only_route") or []) if len(line) >= 2]
    view = folium.Map(tiles=None, zoom_start=2, control_scale=True, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", overlay=False).add_to(view)
    # The neutral map canvas remains usable if tile requests fail.
    view.get_root().header.add_child(folium.Element(
        "<style>.leaflet-container{background:#dce4ec}</style>"
    ))

    if points:
        layer = folium.FeatureGroup(name="GPS observations").add_to(view)
        for point in points:
            folium.CircleMarker(
                point, radius=3, color=GPS, weight=1, fill=True, fill_opacity=0.72,
            ).add_to(layer)

    for name, lines, color, width, show in (
        ("Reconstructed route (HMM)", hmm, HMM, 6, True),
        ("Classifier-only route", classifier, CLASSIFIER, 3, False),
    ):
        if lines:
            layer = folium.FeatureGroup(name=name, show=show).add_to(view)
            for line in lines:
                folium.PolyLine(line, color=color, weight=width, opacity=0.9).add_to(layer)

    if points:
        stops = folium.FeatureGroup(name="Start and end").add_to(view)
        for point, label, color in ((points[0], "Start", "#168249"), (points[-1], "End", "#dc2626")):
            folium.CircleMarker(
                point, radius=7, color="white", weight=2, fill=True,
                fill_color=color, fill_opacity=1, tooltip=label,
            ).add_to(stops)

    bounds = data.get("bounds") or view.get_bounds()
    if bounds and all(value is not None for corner in bounds for value in corner):
        view.fit_bounds(bounds, padding=(24, 24), max_zoom=16)
    if points or hmm or classifier:
        folium.LayerControl(collapsed=False).add_to(view)
    legend = []
    if points:
        legend.extend([(GPS, "GPS observations"), ("#168249", "Start"), ("#dc2626", "End")])
    if hmm:
        legend.append((HMM, "Reconstructed (HMM)"))
    if classifier:
        legend.append((CLASSIFIER, "Classifier-only"))
    if legend:
        entries = "".join(
            f'<div><span style="color:{color};font-size:16px">●</span> {label}</div>'
            for color, label in legend
        )
        view.get_root().html.add_child(folium.Element(
            '<div aria-label="Map legend" style="position:absolute;bottom:30px;left:12px;'
            'z-index:900;background:#111b28;color:white;padding:8px 12px;'
            f'border-radius:6px;font:12px system-ui;pointer-events:none">{entries}</div>'
        ))
    return view
