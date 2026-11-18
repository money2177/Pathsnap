"""Raw-data loading and geographic helpers for PathSnap.

File formats (no headers, whitespace separated):
  .track  lon, lat, t_seconds
  .nodes  lon, lat                (row index = node_id)
  .arcs   from_node_id, to_node_id (row index = arc_id)
  .route  arc_id                   (chronological ground-truth arcs)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import shapely
import pandas as pd
from pyproj import Transformer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Raw dataset location: $PATHSNAP_DATA if set, else <repo>/data (contains train/ and student_test/).
DATA_ROOT = os.environ.get("PATHSNAP_DATA", os.path.join(REPO_ROOT, "data"))
PARTITIONS = ("train", "student_test")


def list_trips(partition: str, root: str = DATA_ROOT) -> list[str]:
    """Trip IDs in a partition (ignores .DS_Store and other non-trip entries)."""
    d = os.path.join(root, partition)
    return sorted(x for x in os.listdir(d) if x.isdigit() and os.path.isdir(os.path.join(d, x)))


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat).

    Trips are spread across the world, so each trip is projected into its own
    local UTM zone; distances are then in metres with negligible distortion.
    """
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Trip:
    trip_id: str
    partition: str
    root: str = DATA_ROOT
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def dir(self) -> str:
        return os.path.join(self.root, self.partition, self.trip_id)

    def _path(self, ext: str) -> str:
        return os.path.join(self.dir, f"{self.trip_id}.{ext}")

    # ---------- raw tables ----------
    @cached_property
    def track(self) -> pd.DataFrame:
        return pd.read_csv(self._path("track"), sep=r"\s+", header=None, names=["lon", "lat", "t"])

    @cached_property
    def nodes(self) -> pd.DataFrame:
        return pd.read_csv(self._path("nodes"), sep=r"\s+", header=None, names=["lon", "lat"])

    @cached_property
    def arcs(self) -> pd.DataFrame:
        return pd.read_csv(self._path("arcs"), sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)

    @cached_property
    def route(self) -> np.ndarray | None:
        p = self._path("route")
        if not os.path.exists(p):
            return None
        return pd.read_csv(p, header=None, names=["arc_id"], dtype=np.int64)["arc_id"].to_numpy()

    # ---------- projected geometry (metres) ----------
    @cached_property
    def epsg(self) -> int:
        return utm_epsg(self.track.lon.mean(), self.track.lat.mean())

    @cached_property
    def _to_xy(self) -> Transformer:
        return Transformer.from_crs(4326, self.epsg, always_xy=True)

    @cached_property
    def track_xy(self) -> np.ndarray:
        return np.column_stack(self._to_xy.transform(self.track.lon.values, self.track.lat.values))

    @cached_property
    def nodes_xy(self) -> np.ndarray:
        return np.column_stack(self._to_xy.transform(self.nodes.lon.values, self.nodes.lat.values))

    @cached_property
    def arc_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """(A, B): start and end coordinates of every arc, shape (n_arcs, 2) each."""
        a = self.arcs.to_numpy()
        return self.nodes_xy[a[:, 0]], self.nodes_xy[a[:, 1]]

    @cached_property
    def arc_length(self) -> np.ndarray:
        A, B = self.arc_xy
        return np.linalg.norm(B - A, axis=1)


def segments_tree(A: np.ndarray, B: np.ndarray):
    """Shapely STRtree over segments (A[i] -> B[i]). Build once per trip and reuse."""
    return shapely.STRtree(shapely.linestrings(np.stack([A, B], axis=1)))


def point_to_segments(P: np.ndarray, A: np.ndarray, B: np.ndarray, tree=None):
    """Exact distance from each point in P (n,2) to its nearest segment among (A,B) (m,2),
    using an R-tree spatial index (O(n log m) instead of O(n*m)).

    Returns (min_dist (n,), argmin_segment (n,)).
    """
    tree = tree if tree is not None else segments_tree(A, B)
    (pi, si), d = tree.query_nearest(shapely.points(P), return_distance=True, all_matches=False)
    dmin = np.full(len(P), np.inf)
    amin = np.full(len(P), -1, dtype=np.int64)
    dmin[pi], amin[pi] = d, si
    return dmin, amin


def load_all(partitions=PARTITIONS) -> list[Trip]:
    return [Trip(t, p) for p in partitions for t in list_trips(p)]
