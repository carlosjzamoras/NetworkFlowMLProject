"""
densify_dataset.py

Densification / data-loading layer: turns the sparse, sharded HDF5
representation produced by generation.py into PyTorch-ready dense
tensors, lazily and only for the batch actually requested.

Responsibilities that deliberately live elsewhere:
  - Running simulations / writing HDF5 shards: generation.py
  - The optimization model itself: pyomo_sim.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from static_network import StaticNetwork


def coo_to_dense_2d(rows, cols, values, shape):
    dense = np.zeros(shape, dtype=np.float32)
    if len(values) > 0:
        dense[rows, cols] = values
    return dense


def coo_to_dense_3d(k_idx, i_idx, j_idx, values, shape):
    dense = np.zeros(shape, dtype=np.float32)
    if len(values) > 0:
        dense[k_idx, i_idx, j_idx] = values
    return dense


class TransportationDataset(Dataset):
    """
    PyTorch Dataset over a directory of sharded HDF5 simulation files.

    Lazy by design: constructing this object only scans shard headers
    (each shard's `num_simulations` attribute) to build a global index ->
    (shard_path, local_index) map. No simulation data is read until
    __getitem__ is called for a specific index.

    HDF5 file handles are NOT kept open across the whole dataset's
    lifetime — each __getitem__ opens its shard, reads exactly one
    simulation group, and closes it. This makes the dataset safe to use
    with DataLoader multiprocessing workers, since no h5py.File handle
    is shared across process boundaries (h5py file handles are not
    guaranteed safe to share after a fork).
    """

    def __init__(self, shard_dir: Path, static_network_path: Path):
        self.shard_dir = Path(shard_dir)
        self.static = StaticNetwork(static_network_path)

        shard_paths = sorted(self.shard_dir.glob("shard_*.h5"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard files found in {self.shard_dir}")

        self._index_map = []  # global_idx -> (shard_path, local_idx)
        for shard_path in shard_paths:
            with h5py.File(shard_path, "r") as f:
                n = int(f.attrs["num_simulations"])
            for local_idx in range(n):
                self._index_map.append((shard_path, local_idx))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, global_idx: int) -> dict:
        shard_path, local_idx = self._index_map[global_idx]
        group_name = f"simulations/simulation_{local_idx:06d}"

        with h5py.File(shard_path, "r") as f:
            group = f[group_name]

            flow_grp = group["optimal_flow"]
            flow_k = flow_grp["k"][:]
            flow_i = flow_grp["i"][:]
            flow_j = flow_grp["j"][:]
            flow_v = flow_grp["v"][:]

            cap_grp = group["capacity"]
            cap_i = cap_grp["i"][:]
            cap_j = cap_grp["j"][:]
            cap_v = cap_grp["v"][:]

            supply = group["supply"][:]
            demand = group["demand"][:]
            price = group["price"][:]
            unmet_demand = group["unmet_demand"][:]

            objective = float(group.attrs["objective"])
            shipping_cost = float(group.attrs["shipping_cost"])

            # h5py can return string attributes as either a native Python
            # str or as numpy.bytes_ depending on version/how the
            # attribute was written — a naive str(...) on bytes produces
            # the literal text "b'SimulatedBalanced'" instead of
            # "SimulatedBalanced". Decode explicitly to avoid that.
            raw_simulation_type = group.attrs["simulation_type"]
            if isinstance(raw_simulation_type, bytes):
                simulation_type = raw_simulation_type.decode("utf-8")
            else:
                simulation_type = str(raw_simulation_type)

        N, K = self.static.N, self.static.K

        capacity_dense = coo_to_dense_2d(cap_i, cap_j, cap_v, (N, N))
        flow_dense = coo_to_dense_3d(flow_k, flow_i, flow_j, flow_v, (K, N, N))

        inputs = {
            "capacity": torch.from_numpy(capacity_dense),
            "supply": torch.from_numpy(supply),
            "demand": torch.from_numpy(demand),
            "price": torch.from_numpy(price),
        }
        target = torch.from_numpy(flow_dense)

        return {
            "inputs": inputs,
            "target": target,
            "unmet_demand": torch.from_numpy(unmet_demand),
            "objective": objective,
            "shipping_cost": shipping_cost,
            "simulation_type": simulation_type,
        }


def default_collate_fn(batch: list) -> dict:
    """
    Stacks a list of __getitem__ dicts into batched tensors. Standard
    torch default_collate handles nested dicts fine, but this explicit
    version keeps the batch structure obvious and easy to extend.
    """
    inputs = {
        key: torch.stack([b["inputs"][key] for b in batch])
        for key in batch[0]["inputs"]
    }
    target = torch.stack([b["target"] for b in batch])
    unmet_demand = torch.stack([b["unmet_demand"] for b in batch])
    objective = torch.tensor([b["objective"] for b in batch], dtype=torch.float32)
    shipping_cost = torch.tensor([b["shipping_cost"] for b in batch], dtype=torch.float32)
    simulation_type = [b["simulation_type"] for b in batch]

    return {
        "inputs": inputs,
        "target": target,
        "unmet_demand": unmet_demand,
        "objective": objective,
        "shipping_cost": shipping_cost,
        "simulation_type": simulation_type,
    }