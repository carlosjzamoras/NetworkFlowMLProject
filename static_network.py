"""
static_network.py

Builds a single, one-time HDF5 file holding the fixed network structure
(city ordering, commodity ordering, directed edge list, and per-edge
distance). This data never changes across simulations, so it is stored
exactly once here rather than being duplicated inside every simulation
group in the sharded dataset.
"""

from pathlib import Path

import h5py
import numpy as np


def build_static_network(bv, output_path: Path) -> None:
    """
    Write city_order, commodity_order, and the directed edge list
    (with distance) to a single HDF5 file.

    `bv` is a BaseValues instance — imported lazily by the caller so this
    module's read-only StaticNetwork class below has no dependency on
    BaseValues (and therefore no dependency on QuickStats/FAF data files)
    for the densification layer.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    city_order = list(bv.city_list)
    commodity_order = list(bv.commodities)
    edges = list(bv.edges)  # directed, bidirectional list of (city_a, city_b)

    city_index = {city: idx for idx, city in enumerate(city_order)}

    edge_i = np.array([city_index[a] for a, b in edges], dtype=np.int32)
    edge_j = np.array([city_index[b] for a, b in edges], dtype=np.int32)
    edge_distance = np.array(
        [bv.city_dist[(a, b)] for a, b in edges], dtype=np.float32
    )

    # h5py requires fixed-length or variable-length string dtypes for
    # string data — use variable-length utf-8 strings so city/commodity
    # names of differing lengths are stored correctly.
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("city_order", data=np.array(city_order, dtype=object), dtype=str_dtype)
        f.create_dataset("commodity_order", data=np.array(commodity_order, dtype=object), dtype=str_dtype)
        f.create_dataset("edge_i", data=edge_i)
        f.create_dataset("edge_j", data=edge_j)
        f.create_dataset("edge_distance", data=edge_distance)

        f.attrs["num_cities"] = len(city_order)
        f.attrs["num_commodities"] = len(commodity_order)
        f.attrs["num_edges"] = len(edges)


class StaticNetwork:
    """
    Read-only accessor for the static network file. Load once and reuse
    — this is small (city/commodity/edge lists only), so it's fine to
    keep fully in memory for the lifetime of a training run.
    """

    def __init__(self, path: Path):
        path = Path(path)
        with h5py.File(path, "r") as f:
            self.city_order = [c.decode("utf-8") if isinstance(c, bytes) else c
                                for c in f["city_order"][:]]
            self.commodity_order = [c.decode("utf-8") if isinstance(c, bytes) else c
                                     for c in f["commodity_order"][:]]
            self.edge_i = f["edge_i"][:]
            self.edge_j = f["edge_j"][:]
            self.edge_distance = f["edge_distance"][:]
            self.num_cities = int(f.attrs["num_cities"])
            self.num_commodities = int(f.attrs["num_commodities"])
            self.num_edges = int(f.attrs["num_edges"])

        self.city_index = {city: idx for idx, city in enumerate(self.city_order)}
        self.commodity_index = {c: idx for idx, c in enumerate(self.commodity_order)}

    @property
    def N(self) -> int:
        return self.num_cities

    @property
    def K(self) -> int:
        return self.num_commodities