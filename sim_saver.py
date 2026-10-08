"""
generation.py

Generation layer: creates simulation scenarios, solves each with Pyomo,
converts flow/capacity to COO, and writes results into sharded HDF5
files. Never accumulates simulations in memory beyond the current shard.

Responsibilities that deliberately live elsewhere:
  - The optimization model itself: pyomo_sim.py
  - Reading the dataset back for training: densify_dataset.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyomo.environ as pyo

from base_values import BaseValues
from static_network import build_static_network
from pyomo_sim import build_and_solve

# NOTE: this assumes simulations.py exposes each regime as a function with
# signature `func(bv: BaseValues, rng: np.random.Generator) -> (demand_dict,
# supply_dict, capacity_dict, price_dict)`. This is checked defensively at
# call time below (see _call_simulation_function) rather than assumed
# silently, since this file has not been independently verified.
from simulations import (
    AbundantSupply,
    SimulatedBalanced,
    SimulatedUnderSupply,
    SimulatedCapacityConstrained,
)

SIMULATION_FUNCTIONS = {
    "SimulatedBalanced": SimulatedBalanced,
    "AbundantSupply": AbundantSupply,
    "SimulatedUnderSupply": SimulatedUnderSupply,
    "SimulatedCapacityConstrained": SimulatedCapacityConstrained,
}

REGIME_WEIGHTS = {
    "SimulatedBalanced": 0.60,
    "AbundantSupply": 0.40 / 3,
    "SimulatedUnderSupply": 0.40 / 3,
    "SimulatedCapacityConstrained": 0.40 / 3,
}

# Solver output can carry tiny floating-point noise (e.g. 1e-10) even where
# the true optimal value is exactly zero. Treating anything below this as
# zero keeps the COO arrays from filling up with meaningless near-zero
# entries that inflate storage and add no real information.
ZERO_TOLERANCE = 1e-6


def _call_simulation_function(sim_fn, bv, rng):
    """
    Calls a regime function with the expected (bv, rng) signature, and
    raises a clear, specific error (rather than an opaque TypeError deep
    in a stack trace) if the function doesn't match that contract.
    """
    try:
        result = sim_fn(bv, rng)
    except TypeError as e:
        raise TypeError(
            f"simulations.py function '{sim_fn.__name__}' could not be called as "
            f"func(bv, rng). Expected signature: func(bv: BaseValues, "
            f"rng: np.random.Generator) -> (demand_dict, supply_dict, "
            f"capacity_dict, price_dict). Original error: {e}"
        ) from e

    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError(
            f"simulations.py function '{sim_fn.__name__}' must return a 4-tuple "
            f"(demand_dict, supply_dict, capacity_dict, price_dict); "
            f"got {type(result)} with length "
            f"{len(result) if isinstance(result, tuple) else 'N/A'}."
        )
    return result


@dataclass
class GenerationConfig:
    num_simulations: int = 100_000
    shard_size: int = 2_500
    seed: int = 42
    output_dir: Path = Path("training_data")
    static_network_path: Path = Path("static_network.h5")
    require_exact_count: bool = True
    max_extra_attempts_factor: int = 5  # safety cap when require_exact_count retries failures


def create_simulation_plan(num_samples: int, rng: np.random.Generator) -> list:
    """
    60% SimulatedBalanced, remaining 40% split evenly across the other
    three regimes. Order is shuffled so regimes are interleaved, not
    contiguous.
    """
    balanced_count = int(round(num_samples * 0.60))
    remaining = num_samples - balanced_count

    other_types = ["AbundantSupply", "SimulatedUnderSupply", "SimulatedCapacityConstrained"]
    plan = ["SimulatedBalanced"] * balanced_count
    for i in range(remaining):
        plan.append(other_types[i % len(other_types)])

    plan = np.array(plan, dtype=object)
    rng.shuffle(plan)
    return plan.tolist()


def _dense_to_coo_2d(matrix: np.ndarray):
    """(N, N) dense -> (rows, cols, values), nonzero entries only."""
    rows, cols = np.nonzero(matrix)
    values = matrix[rows, cols].astype(np.float32)
    return rows.astype(np.int32), cols.astype(np.int32), values


def _flow_to_coo(model: pyo.ConcreteModel, bv: BaseValues, static) -> tuple:
    """
    Extract nonzero flow entries directly from the solved Pyomo model,
    without ever materializing the dense (K, N, N) tensor. Values below
    FLOW_EPSILON are treated as solver-tolerance noise, not real flow.
    Returns (commodity_idx, row_idx, col_idx, values) as int32/float32 arrays.
    """
    k_list, i_list, j_list, v_list = [], [], [], []
    for (i, j) in model.Arcs:
        for c in model.Commodities:
            value = pyo.value(model.x[i, j, c])
            if value is None or abs(value) < ZERO_TOLERANCE:
                continue
            k_list.append(static.commodity_index[c])
            i_list.append(static.city_index[i])
            j_list.append(static.city_index[j])
            v_list.append(value)

    return (
        np.array(k_list, dtype=np.int32),
        np.array(i_list, dtype=np.int32),
        np.array(j_list, dtype=np.int32),
        np.array(v_list, dtype=np.float32),
    )


def _capacity_to_coo(capacity_dict: dict, static) -> tuple:
    """Nonzero capacity entries as (row_idx, col_idx, values), same tolerance as flow."""
    i_list, j_list, v_list = [], [], []
    for (a, b), value in capacity_dict.items():
        if abs(value) < ZERO_TOLERANCE:
            continue
        i_list.append(static.city_index[a])
        j_list.append(static.city_index[b])
        v_list.append(value)
    return (
        np.array(i_list, dtype=np.int32),
        np.array(j_list, dtype=np.int32),
        np.array(v_list, dtype=np.float32),
    )


def _dict_to_dense_nk(data_dict: dict, static) -> np.ndarray:
    """{(city, commodity): value} -> dense (N, K) float32 array."""
    matrix = np.zeros((static.N, static.K), dtype=np.float32)
    for (city, commodity), value in data_dict.items():
        matrix[static.city_index[city], static.commodity_index[commodity]] = value
    return matrix


def _run_one_simulation(bv: BaseValues, static, regime_name: str, rng: np.random.Generator):
    """
    Generate one scenario and solve it. Returns (sample_dict, termination)
    where sample_dict is None if the solve did not terminate optimally.

    NOTE: build_and_solve's actual return signature (per pyomo_sim.py) is:
        (model, result, shipping_cost, unmet_demand_dict, objective_value)
    with shipping_cost/unmet_demand_dict/objective_value all None when the
    solve is not optimal. This reuses those values directly rather than
    recomputing them, to stay consistent with the exact objective
    formulation in pyomo_sim.py.
    """
    sim_fn = SIMULATION_FUNCTIONS[regime_name]
    demand_dict, supply_dict, capacity_dict, price_dict = _call_simulation_function(
        sim_fn, bv, rng
    )

    model, result, shipping_cost, unmet_demand_dict, objective_value = build_and_solve(
        bv, demand_dict, supply_dict, capacity_dict, price_dict
    )
    termination = result.solver.termination_condition

    if termination != pyo.TerminationCondition.optimal:
        return None, termination

    flow_k, flow_i, flow_j, flow_v = _flow_to_coo(model, bv, static)
    cap_i, cap_j, cap_v = _capacity_to_coo(capacity_dict, static)

    supply_matrix = _dict_to_dense_nk(supply_dict, static)
    demand_matrix = _dict_to_dense_nk(demand_dict, static)
    price_matrix = _dict_to_dense_nk(price_dict, static)

    unmet_matrix = np.zeros((static.N, static.K), dtype=np.float32)
    for (n, c), v in (unmet_demand_dict or {}).items():
        if v is not None and abs(v) >= ZERO_TOLERANCE:
            unmet_matrix[static.city_index[n], static.commodity_index[c]] = v

    sample = {
        "flow_k": flow_k, "flow_i": flow_i, "flow_j": flow_j, "flow_v": flow_v,
        "cap_i": cap_i, "cap_j": cap_j, "cap_v": cap_v,
        "supply": supply_matrix, "demand": demand_matrix,
        "price": price_matrix, "unmet_demand": unmet_matrix,
        "objective": float(objective_value), "shipping_cost": float(shipping_cost),
        "simulation_type": regime_name,
    }
    return sample, termination


def _write_simulation_group(shard_file: h5py.File, local_idx: int, sample: dict) -> None:
    group = shard_file.create_group(f"simulations/simulation_{local_idx:06d}")

    flow_grp = group.create_group("optimal_flow")
    flow_grp.create_dataset("k", data=sample["flow_k"])
    flow_grp.create_dataset("i", data=sample["flow_i"])
    flow_grp.create_dataset("j", data=sample["flow_j"])
    flow_grp.create_dataset("v", data=sample["flow_v"])

    cap_grp = group.create_group("capacity")
    cap_grp.create_dataset("i", data=sample["cap_i"])
    cap_grp.create_dataset("j", data=sample["cap_j"])
    cap_grp.create_dataset("v", data=sample["cap_v"])

    group.create_dataset("supply", data=sample["supply"])
    group.create_dataset("demand", data=sample["demand"])
    group.create_dataset("price", data=sample["price"])
    group.create_dataset("unmet_demand", data=sample["unmet_demand"])

    group.attrs["objective"] = sample["objective"]
    group.attrs["shipping_cost"] = sample["shipping_cost"]
    group.attrs["simulation_type"] = sample["simulation_type"]


def generate_dataset(config: GenerationConfig) -> dict:
    """
    Run the full generation pipeline. Returns a small summary dict
    (counts, failure breakdown) — does not return simulation data itself.

    require_exact_count contract (exactly one of these two behaviors,
    no ambiguity):

      True  -> Keep attempting additional simulations, beyond the
               original shuffled plan if necessary, until EXACTLY
               config.num_simulations valid (optimal) samples have been
               written — or until max_total_attempts is exceeded, in
               which case a RuntimeError is raised rather than silently
               returning a short dataset.

      False -> Attempt EXACTLY config.num_simulations simulations (one
               per entry in the shuffled plan, regardless of whether
               each one succeeds or fails), then stop. The resulting
               valid_simulations count may therefore be LESS than
               config.num_simulations if any attempts failed to solve
               optimally. No retries occur in this mode.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    static_path = Path(config.static_network_path)
    bv = BaseValues()

    if not static_path.exists():
        build_static_network(bv, static_path)

    from static_network import StaticNetwork
    static = StaticNetwork(static_path)

    root_seed_seq = np.random.SeedSequence(config.seed)
    plan_rng = np.random.default_rng(root_seed_seq)

    plan = create_simulation_plan(config.num_simulations, plan_rng)

    def next_rng():
        """
        Lazily spawn one independent, reproducible child RNG per call.
        SeedSequence.spawn() is safe to call repeatedly over the object's
        lifetime — it tracks how many children have been spawned
        internally — so there's no fixed budget to precompute or exhaust,
        unlike a pre-sized pool of child seeds.
        """
        child_seed = root_seed_seq.spawn(1)[0]
        return np.random.default_rng(child_seed)

    valid_count = 0
    failure_counts = {}
    shard_index = 0
    local_idx = 0
    shard_file = None

    def open_new_shard():
        nonlocal shard_file, local_idx
        if shard_file is not None:
            shard_file.close()
        path = output_dir / f"shard_{shard_index:05d}.h5"
        shard_file = h5py.File(path, "w")
        shard_file.attrs["static_network_path"] = str(static_path)
        local_idx = 0
        return shard_file

    shard_file = open_new_shard()

    plan_cursor = 0
    total_attempts = 0
    max_total_attempts = config.num_simulations * config.max_extra_attempts_factor

    while valid_count < config.num_simulations:
        # ── Determine which regime to attempt this iteration ──
        if plan_cursor < len(plan):
            regime_name = plan[plan_cursor]
            plan_cursor += 1
        elif config.require_exact_count:
            # Plan exhausted but still short of the exact target — draw
            # additional regimes according to the same weighting until we
            # reach num_simulations valid samples (or hit the hard cap).
            regime_name = str(plan_rng.choice(
                list(REGIME_WEIGHTS.keys()), p=list(REGIME_WEIGHTS.values())
            ))
        else:
            # Plan exhausted and exact count not required — stop here,
            # deliberately, regardless of success/failure state. This is
            # checked BEFORE attempting another solve, unlike the
            # previous version, which could skip this check entirely on
            # a failed sample and loop indefinitely.
            break

        total_attempts += 1
        if total_attempts > max_total_attempts:
            raise RuntimeError(
                f"Exceeded max_total_attempts ({max_total_attempts}) while trying to "
                f"reach {config.num_simulations} valid simulations. Only {valid_count} "
                f"valid samples were produced. Failure breakdown so far: {failure_counts}. "
                f"This usually means one or more regimes are producing infeasible "
                f"scenarios too often — consider loosening regime parameters or "
                f"increasing max_extra_attempts_factor."
            )

        rng = next_rng()
        sample, termination = _run_one_simulation(bv, static, regime_name, rng)

        if sample is None:
            failure_counts[str(termination)] = failure_counts.get(str(termination), 0) + 1
            continue  # do not write; loop condition / plan-exhaustion check re-evaluated next iteration

        _write_simulation_group(shard_file, local_idx, sample)
        local_idx += 1
        valid_count += 1
        print(f"Saved sample {valid_count}/{config.num_simulations}") 

        if local_idx >= config.shard_size:
            shard_file.attrs["num_simulations"] = local_idx
            shard_index += 1
            print(f"Shard {shard_index - 1} complete: {local_idx} simulations saved")
            shard_file = open_new_shard()

    if shard_file is not None:
        if local_idx > 0:
            shard_file.attrs["num_simulations"] = local_idx
            shard_file.close()
        else:
            # last shard ended up empty (exact multiple of shard_size, or
            # require_exact_count=False stopped right at a boundary) — discard it
            path = Path(shard_file.filename)
            shard_file.close()
            path.unlink(missing_ok=True)

    summary = {
        "valid_simulations": valid_count,
        "requested_simulations": config.num_simulations,
        "total_attempts": total_attempts,
        "num_shards": shard_index + (1 if local_idx > 0 else 0),
        "failures": failure_counts,
    }
    print("\nGENERATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


# if __name__ == "__main__":
#     # Anchor paths to this script's own directory, not the current working
#     # directory — otherwise output location depends on where you happen to
#     # invoke the script from, rather than staying fixed relative to the
#     # project.
#     project_root = Path(__file__).resolve().parent

#     config = GenerationConfig(
#         num_simulations=10,
#         shard_size=4,
#         seed=42,
#         output_dir= Path("~/Desktop/project1/training_data").expanduser(),
#         static_network_path=project_root / "static_network_test.h5",
#     )
#     generate_dataset(config)