import numpy as np
import pyomo.environ as pyo

from base_values import BaseValues


RATE_TON_MILE = 0.05 #Set as a constant rate tonnage per mile


#Actual Pyomo Model Build

def build_and_solve(
    bv: BaseValues,
    demand_dict: dict,
    supply_dict: dict,
    capacity_dict: dict,
    price_dict: dict = None,
):
    """
    Build and solve one transportation optimization problem.
    """

    model = pyo.ConcreteModel()

#Sets

    model.Nodes = pyo.Set(
        initialize=bv.city_list
    )

    model.Commodities = pyo.Set(
        initialize=bv.commodities
    )

    model.Arcs = pyo.Set(
        initialize=list(capacity_dict.keys()),
        dimen=2
    )

#Parameters

    distance = {
        arc: bv.city_dist[arc]
        for arc in capacity_dict.keys()
    }

    model.dist = pyo.Param(
        model.Arcs,
        initialize=distance
    )

    model.cap = pyo.Param(
        model.Arcs,
        initialize=capacity_dict
    )

#Decision Variables

    model.x = pyo.Var(
        model.Arcs,
        model.Commodities,
        domain=pyo.NonNegativeReals
    )

    model.unused_supply = pyo.Var(
        model.Nodes,
        model.Commodities,
        domain=pyo.NonNegativeReals
    )

    model.unmet_demand = pyo.Var(
        model.Nodes,
        model.Commodities,
        domain=pyo.NonNegativeReals
    )

#Price

    price = (
        price_dict
        if price_dict is not None
        else bv.price_dict
    )

#Unmet demand Penalty

    max_arc_cost = max(
        price.get((i, c), 0)
        + RATE_TON_MILE * distance[(i, j)]
        for (i, j) in capacity_dict.keys()
        for c in bv.commodities
    )

    unmet_penalty = 5 * max_arc_cost

#Objective Function

    def obj_rule(m):

        shipping_cost = sum(
            (
                price.get((i, c), 0)
                + RATE_TON_MILE * m.dist[i, j]
            )
            * m.x[i, j, c]

            for (i, j) in m.Arcs
            for c in m.Commodities
        )

        unmet_cost = (
            unmet_penalty
            * sum(
                m.unmet_demand[n, c]
                for n in m.Nodes
                for c in m.Commodities
            )
        )

        return shipping_cost + unmet_cost

    model.cost = pyo.Objective(
        rule=obj_rule,
        sense=pyo.minimize
    )

#Capacity Constraint 
    def cap_rule(m, i, j):

        return sum(
            m.x[i, j, c]
            for c in m.Commodities
        ) <= m.cap[i, j]

    model.cap_con = pyo.Constraint(
        model.Arcs,
        rule=cap_rule
    )

#Flow Conservation

    def conservation_rule(m, n, c):

        inflow = sum(
            m.x[i, j, c]
            for (i, j) in m.Arcs
            if j == n
        )

        outflow = sum(
            m.x[i, j, c]
            for (i, j) in m.Arcs
            if i == n
        )

        net_supply = (
            supply_dict.get((n, c), 0)
            - demand_dict.get((n, c), 0)
        )

        expr = (
            outflow
            - inflow
            + m.unused_supply[n, c]
            - m.unmet_demand[n, c]
            == net_supply
        )

        if isinstance(expr, bool):
            return pyo.Constraint.Skip

        return expr

    model.conservation_con = pyo.Constraint(
        model.Nodes,
        model.Commodities,
        rule=conservation_rule
    )
#Solve

    solver = pyo.SolverFactory("highs")

    result = solver.solve(
        model,
        tee=False
    )

#Results

    if (
        result.solver.termination_condition
        != pyo.TerminationCondition.optimal
    ):
        return (
            model,
            result,
            None,
            None,
            None,
        )

    shipping_cost = pyo.value(
        sum(
            (
                price.get((i, c), 0)
                + RATE_TON_MILE * model.dist[i, j]
            )
            * model.x[i, j, c]

            for (i, j) in model.Arcs
            for c in model.Commodities
        )
    )

    unmet_demand_dict = {
        (n, c): pyo.value(
            model.unmet_demand[n, c]
        )

        for n in model.Nodes
        for c in model.Commodities

        if pyo.value(
            model.unmet_demand[n, c]
        ) > 0
    }

    objective_value = pyo.value(
        model.cost
    )

    return (
        model,
        result,
        shipping_cost,
        unmet_demand_dict,
        objective_value,
    )


#Flow Matrix

def flows_to_matrix(
    model: pyo.ConcreteModel,
    bv: BaseValues,
):
    """
    Convert Pyomo flow variables into:

        (K, N, N)

    K = commodities
    N = cities
    """

    city_order = list(bv.city_list)
    commodity_order = list(bv.commodities)

    city_index = {
        city: idx
        for idx, city in enumerate(city_order)
    }

    commodity_index = {
        c: idx
        for idx, c in enumerate(commodity_order)
    }

    n = len(city_order)
    k = len(commodity_order)

    matrix = np.zeros(
        (k, n, n),
        dtype=np.float32
    )

    for (i, j) in model.Arcs:

        for c in model.Commodities:

            value = pyo.value(
                model.x[i, j, c]
            )

            if value is None:
                continue

            matrix[
                commodity_index[c],
                city_index[i],
                city_index[j]
            ] = value

    return (
        matrix,
        commodity_order,
        city_order,
    )


#Dictionary to Matrix
def dict_to_matrix(
    data_dict,
    bv: BaseValues,
):
    """
    Convert {(city, commodity): value}
    into:

        (N, K)
    """

    city_order = list(bv.city_list)
    commodity_order = list(bv.commodities)

    city_index = {
        city: idx
        for idx, city in enumerate(city_order)
    }

    commodity_index = {
        c: idx
        for idx, c in enumerate(commodity_order)
    }

    matrix = np.zeros(
        (
            len(city_order),
            len(commodity_order)
        ),
        dtype=np.float32
    )

    for (city, commodity), value in data_dict.items():

        i = city_index[city]
        k = commodity_index[commodity]

        matrix[i, k] = value

    return matrix


#Capacity to Matrix
def capacity_to_matrix(
    capacity_dict,
    bv: BaseValues,
):
    """
    Convert {(city_i, city_j): capacity}
    into:

        (N, N)
    """

    city_order = list(bv.city_list)

    city_index = {
        city: idx
        for idx, city in enumerate(city_order)
    }

    n = len(city_order)

    matrix = np.zeros(
        (n, n),
        dtype=np.float32
    )

    for (i, j), capacity in capacity_dict.items():

        a = city_index[i]
        b = city_index[j]

        matrix[a, b] = capacity

    return matrix


#Unmet Demand to Matrix

def unmet_demand_to_matrix(
    unmet_demand_dict,
    bv: BaseValues,
):
    """
    Convert {(city, commodity): unmet}
    into:

        (N, K)
    """

    return dict_to_matrix(
        unmet_demand_dict,
        bv
    )


#Convert Graph to Tensor
def graph_to_tensor(
    bv: BaseValues,
    capacity_dict: dict,
):
    """
    Construct a graph tensor with 3 channels:

        channel 0 = adjacency
        channel 1 = distance
        channel 2 = capacity

    Shape:

        (3, N, N)
    """

    city_order = list(bv.city_list)

    city_index = {
        city: idx
        for idx, city in enumerate(city_order)
    }

    n = len(city_order)

    graph = np.zeros(
        (3, n, n),
        dtype=np.float32
    )

    for (i, j) in capacity_dict.keys():

        a = city_index[i]
        b = city_index[j]

        # Adjacency
        graph[0, a, b] = 1

        # Distance
        graph[1, a, b] = bv.city_dist[i, j]

        # Capacity
        graph[2, a, b] = capacity_dict[i, j]

    return graph


#Sample Run
def run_sample(
    bv: BaseValues,
    demand_dict: dict,
    supply_dict: dict,
    capacity_dict: dict,
    price_dict: dict = None,
):
    """
    Solve one simulation and return all information
    needed for dataset generation.
    """

    (
        model,
        result,
        shipping_cost,
        unmet_demand,
        objective,
    ) = build_and_solve(
        bv,
        demand_dict,
        supply_dict,
        capacity_dict,
        price_dict,
    )

    status = result.solver.status

    termination = (
        result.solver.termination_condition
    )

#Fail Save
    if termination != pyo.TerminationCondition.optimal:

        return {
            "status": status,
            "termination": termination,
            "matrix": None,
            "demand": None,
            "supply": None,
            "capacity": None,
            "graph": None,
            "unmet_demand": None,
            "objective": None,
            "shipping_cost": None,
            "commodity_order": None,
            "city_order": None,
        }

#Converting the results

    matrix, commodity_order, city_order = (
        flows_to_matrix(model, bv)
    )

    demand_matrix = dict_to_matrix(
        demand_dict,
        bv
    )

    supply_matrix = dict_to_matrix(
        supply_dict,
        bv
    )

    capacity_matrix = capacity_to_matrix(
        capacity_dict,
        bv
    )

    graph = graph_to_tensor(
        bv,
        capacity_dict
    )

    unmet_matrix = unmet_demand_to_matrix(
        unmet_demand,
        bv
    )

#Run Everything

    return {
        "status": status,
        "termination": termination,

        # Input
        "graph": graph,
        "demand": demand_matrix,
        "supply": supply_matrix,
        "capacity": capacity_matrix,

        # Output
        "matrix": matrix,

        # Additional information
        "unmet_demand": unmet_matrix,
        "objective": objective,
        "shipping_cost": shipping_cost,

        "commodity_order": commodity_order,
        "city_order": city_order,
    }


#Run Specific Simulations

def run_simulation(
    simulation_type,
    bv=None,
):
    """
    Generate and solve one sample.

    Valid simulation types:

        "AbundantSupply"
        "SimulatedBalanced"
        "SimulatedUnderSupply"
        "SimulatedCapacityConstrained"

    Example:

        result = run_simulation(
            "SimulatedBalanced"
        )
    """

    if bv is None:
        bv = BaseValues()

    # Import here to avoid circular imports.
    from simulations import (
        AbundantSupply,
        SimulatedBalanced,
        SimulatedUnderSupply,
        SimulatedCapacityConstrained,
    )

    simulation_functions = {
        "AbundantSupply": AbundantSupply,
        "SimulatedBalanced": SimulatedBalanced,
        "SimulatedUnderSupply": SimulatedUnderSupply,
        "SimulatedCapacityConstrained":
            SimulatedCapacityConstrained,
    }

    if simulation_type not in simulation_functions:

        raise ValueError(
            f"Unknown simulation type: "
            f"{simulation_type}\n"
            f"Valid types: "
            f"{list(simulation_functions.keys())}"
        )


    #Generation

    simulation_function = (
        simulation_functions[simulation_type]
    )

    (
        demand_dict,
        supply_dict,
        capacity_dict,
        price_dict,
    ) = simulation_function()

   #Solve

    result = run_sample(
        bv=bv,
        demand_dict=demand_dict,
        supply_dict=supply_dict,
        capacity_dict=capacity_dict,
        price_dict=price_dict,
    )

    #original dictionaries
    #for debugging.
    result["simulation_type"] = simulation_type
    result["demand_dict"] = demand_dict
    result["supply_dict"] = supply_dict
    result["capacity_dict"] = capacity_dict
    result["price_dict"] = price_dict

    return result