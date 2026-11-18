"""Zoomed Folium maps for the report: classifier-only route vs HMM route vs reference.

For each trip, picks the 90-point window (vehicle moving at least 300 m) with the most points that the
HMM corrected, and saves the full-trip map zoomed to that window:
    outputs/hmm/maps/zoom_trip_<id>.html
Uses outputs/hmm/student_test_decoded.parquet from scripts/hmm_decode.py.

Run: python scripts/zoom_maps.py [trip ids...]   (default: 00000056 00000091)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from src.data_io import REPO_ROOT, Trip
from src.hmm import RoadNetwork, reconstruct_route
from src.viz import COLORS, trip_map

OUT = os.path.join(REPO_ROOT, "outputs", "hmm", "maps")
WINDOW, MIN_SPAN_M = 90, 300

if __name__ == "__main__":
    trips = sys.argv[1:] or ["00000056", "00000091"]
    dec = pd.read_parquet(os.path.join(REPO_ROOT, "outputs", "hmm", "student_test_decoded.parquet"))
    for tid in trips:
        d = dec[dec.trip == tid].sort_values("point_idx").reset_index(drop=True)
        trip = Trip(tid, "student_test")
        xy = trip.track_xy[d.point_idx]
        fixed = ((d.correct_classifier == 0) & (d.correct_hmm == 1)).to_numpy()
        best, start = -1, 0
        for i in range(0, max(len(d) - WINDOW, 1), 10):
            if np.ptp(xy[i:i + WINDOW], axis=0).max() > MIN_SPAN_M and fixed[i:i + WINDOW].sum() > best:
                best, start = int(fixed[i:i + WINDOW].sum()), i
        w = d.iloc[start:start + WINDOW]
        net = RoadNetwork(trip)
        r_clf = reconstruct_route(net, d.arc_classifier.to_numpy())[0]
        r_hmm = reconstruct_route(net, d.arc_hmm.to_numpy())[0]
        m = trip_map(trip, routes={"Classifier-only route": (r_clf, COLORS["classifier"]),
                                   "HGB + HMM route": (r_hmm, COLORS["predicted"])})
        lat, lon = trip.track.lat.to_numpy()[w.point_idx], trip.track.lon.to_numpy()[w.point_idx]
        m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
        path = f"{OUT}/zoom_trip_{tid}.html"
        m.save(path)
        print(f"{tid}: points {start}-{start + WINDOW}, {best} corrected by the HMM in this window -> {path}")
