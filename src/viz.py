"""Folium map helpers (reused later for predicted-route maps)."""
from __future__ import annotations

import folium
import numpy as np

from .data_io import Trip

COLORS = {
    "network": "#9e9e9e",
    "gps": "#1f5fd6",
    "reference": "#1a9e3a",
    "predicted": "#e8590c",
    "gap": "#c2255c",
}


ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
BASEMAPS = [  # (name, url, attribution, max_native_zoom) — none of these need an API key
    ("Light grey (Esri)", ESRI.format("Canvas/World_Light_Gray_Base"), "Tiles &copy; Esri", 16),
    ("OpenStreetMap", "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "&copy; OpenStreetMap contributors", 19),
    ("Satellite (Esri)", ESRI.format("World_Imagery"), "Tiles &copy; Esri, Maxar, Earthstar Geographics", 19),
]


def base_map(**kwargs) -> folium.Map:
    """folium.Map with key-free basemaps (first one shown by default) selectable in the layer control."""
    m = folium.Map(tiles=None, control_scale=True, **kwargs)
    for i, (name, url, attr, maxz) in enumerate(BASEMAPS):
        folium.TileLayer(url, name=name, attr=attr, max_native_zoom=maxz, max_zoom=21, overlay=False,
                         control=True, show=(i == 0)).add_to(m)
    return m


def _legend(m: folium.Map, items: dict[str, str], title: str) -> None:
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;margin-right:6px;'
        f'vertical-align:middle;background:{c}"></span>{label}</div>'
        for label, c in items.items()
    )
    html = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 12px;border:1px solid #888;border-radius:6px;font:13px sans-serif;">'
        f"<b>{title}</b>{rows}</div>"
    )
    m.get_root().html.add_child(folium.Element(html))


def network_segments_near_track(trip: Trip, buffer_m: float = 150.0, max_arcs: int = 40000) -> list:
    """Road arcs (as [[lat,lon],[lat,lon]]) whose start or end node is within buffer_m of the
    track's projected bounding box. Bounds map size for huge networks."""
    xy = trip.track_xy
    lo, hi = xy.min(0) - buffer_m, xy.max(0) + buffer_m
    A, B = trip.arc_xy
    inside = np.all((A >= lo) & (A <= hi), axis=1) | np.all((B >= lo) & (B <= hi), axis=1)
    idx = np.flatnonzero(inside)[:max_arcs]
    a = trip.arcs.to_numpy()[idx]
    lat, lon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
    return [[[lat[u], lon[u]], [lat[v], lon[v]]] for u, v in a]


def arcs_to_lines(trip: Trip, arc_ids) -> list:
    a = trip.arcs.to_numpy()[np.asarray(arc_ids)]
    lat, lon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
    return [[[lat[u], lon[u]], [lat[v], lon[v]]] for u, v in a]


def trip_map(
    trip: Trip,
    predicted_arcs=None,
    show_network: bool = True,
    gap_threshold_s: float = 5.0,
    gps_every: int = 1,
) -> folium.Map:
    tr = trip.track
    m = base_map()
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    legend = {}

    if show_network:
        fg = folium.FeatureGroup("Road network (near track)", show=True)
        folium.PolyLine(network_segments_near_track(trip), color=COLORS["network"], weight=1, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend["Road network"] = COLORS["network"]

    if trip.route is not None:
        fg = folium.FeatureGroup("Reference route (.route)", show=True)
        folium.PolyLine(arcs_to_lines(trip, trip.route), color=COLORS["reference"], weight=5, opacity=0.8).add_to(fg)
        fg.add_to(m)
        legend["Reference route"] = COLORS["reference"]

    if predicted_arcs is not None:
        fg = folium.FeatureGroup("Predicted route", show=True)
        folium.PolyLine(arcs_to_lines(trip, predicted_arcs), color=COLORS["predicted"], weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)
        legend["Predicted route"] = COLORS["predicted"]

    fg = folium.FeatureGroup("Raw GPS observations (.track)", show=True)
    for r in tr.iloc[::gps_every].itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={r.t:.0f}s").add_to(fg)
    fg.add_to(m)
    legend["Raw GPS"] = COLORS["gps"]

    dt = np.diff(tr.t.to_numpy())
    gaps = np.flatnonzero(dt > gap_threshold_s)
    if len(gaps):
        fg = folium.FeatureGroup(f"GPS gaps > {gap_threshold_s:.0f}s", show=True)
        for i in gaps:
            a, b = tr.iloc[i], tr.iloc[i + 1]
            folium.PolyLine([[a.lat, a.lon], [b.lat, b.lon]], color=COLORS["gap"], weight=3, dash_array="6 6",
                            tooltip=f"gap {dt[i]:.0f}s").add_to(fg)
        fg.add_to(m)
        legend[f"Gap > {gap_threshold_s:.0f}s"] = COLORS["gap"]

    folium.Marker([tr.lat.iloc[0], tr.lon.iloc[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([tr.lat.iloc[-1], tr.lon.iloc[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _legend(m, legend, f"Trip {trip.trip_id} ({trip.partition})")
    return m
