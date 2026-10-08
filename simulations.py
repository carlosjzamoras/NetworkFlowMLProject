"""
simulations.py

The file does a stochastic simulation for three scenarios 
1) balance supply and demand
2) Unbalances supply and demand
3) Capacity Constrained

"""

import numpy as np
from collections import defaultdict


commodity_source_cities = {

    "CORN": [
        "Des Moines", "Chicago", "Springfield", "Omaha", "Lincoln",
        "Indianapolis", "Fort Wayne", "Cincinnati", "Columbus", "Dayton", "Toledo",
        "St. Louis", "Kansas City", "Wichita",
        "Minneapolis", "Saint Paul", "Milwaukee",
        "Louisville", "Lexington", "Nashville",
        "Sioux Falls",
        "Little Rock", "Memphis", "Jackson",
    ],

    "SOYBEANS": [
        "Chicago", "Springfield", "Des Moines",
        "St. Louis", "Kansas City", "Omaha", "Lincoln",
        "Indianapolis", "Fort Wayne", "Cincinnati", "Columbus", "Dayton", "Toledo",
        "Minneapolis", "Saint Paul", "Milwaukee",
        "Sioux Falls",
        "Little Rock", "Memphis", "Jackson", "Nashville",
        "Louisville",
    ],

    "WHEAT": [
        "Wichita", "Topeka", "Kansas City",
        "Oklahoma City", "Tulsa",
        "Omaha", "Lincoln",
        "Sioux Falls", "Bismarck",
        "Billings",
        "Lubbock", "Dallas", "San Antonio",
        "Denver", "Colorado Springs", "Cheyenne",
        "Boise", "Spokane", "Seattle",
        "Minneapolis", "Saint Paul",
        "St. Louis",
    ],

    "COTTON": [
        "Lubbock", "Dallas", "El Paso", "San Antonio", "Corpus Christi",
        "Laredo", "Houston", "Austin", "Beaumont",
        "Jackson", "Memphis", "Little Rock", "Shreveport",
        "Birmingham", "Montgomery", "Mobile",
        "Atlanta", "Augusta", "Savannah",
        "Columbia", "Greenville", "Charleston",
        "Charlotte", "Greensboro", "Raleigh", "Wilmington",
        "Phoenix", "Tucson",
        "Fresno",
    ],

    "HAY": [
        "Boise", "Salt Lake City", "Provo", "Denver", "Colorado Springs",
        "Cheyenne", "Billings",
        "Spokane", "Seattle", "Eugene", "Portland",
        "Bismarck", "Sioux Falls",
        "Omaha", "Lincoln", "Wichita", "Topeka",
        "Lubbock", "Dallas", "San Antonio", "Austin",
        "Albuquerque", "Phoenix", "Tucson",
        "Fresno", "Sacramento",
        "Des Moines", "Minneapolis", "Saint Paul", "Milwaukee", "Chicago",
        "Atlanta", "Nashville", "Lexington",
    ]
}

commodities = ["CORN", "SOYBEANS", "WHEAT", "COTTON", "HAY"]


def split_into(rng: np.random.Generator, partitions, total_number):
    proportions = rng.dirichlet(np.ones(partitions))
    return proportions * total_number


def generate_capacity(
    rng: np.random.Generator,
    edges,
    supply_dict,
    demand_dict,
    commodities,
    low_prob=0.25,
    medium_prob=0.50,
    high_prob=0.25,
    low_factor=0.20,
    medium_factor=0.80,
    high_factor=2.00,
    sigma=0.50,
):
    total_supply = sum(supply_dict.values())
    total_demand = sum(demand_dict.values())

    network_capacity = np.sqrt(total_supply * total_demand)

    regime_multiplier = {
        "low": low_factor,
        "medium": medium_factor,
        "high": high_factor,
    }

    capacity_dict = {}
    processed_edges = set()

    for city_a, city_b in edges:

        physical_edge = frozenset((city_a, city_b))

        if physical_edge in processed_edges:
            continue

        processed_edges.add(physical_edge)

        regime = rng.choice(
            ["low", "medium", "high"],
            p=[low_prob, medium_prob, high_prob]
        )

        random_factor = rng.lognormal(
            mean=0,
            sigma=sigma
        )

        capacity = (
            network_capacity
            * regime_multiplier[regime]
            * random_factor
        )

        capacity_dict[(city_a, city_b)] = capacity
        capacity_dict[(city_b, city_a)] = capacity

    return capacity_dict


def _generate_price(rng: np.random.Generator, bv, supply_dict):
    """
    Shared price-generation logic used by every regime: perturb each
    commodity's known baseline prices, then assign (without replacement)
    to the city-commodity pairs established by supply_dict.
    """
    old_price = defaultdict(list)
    for (city, commodity), value in bv.price_dict.items():
        old_price[commodity].append(value)

    for commodity in old_price:
        old_price[commodity] = [
            value * rng.uniform(0.7, 1.35)
            for value in old_price[commodity]
        ]
        rng.shuffle(old_price[commodity])

    price_dict = {}
    for commodity in old_price:
        pairs = [
            (city, commodity)
            for (city, comm) in supply_dict
            if comm == commodity
        ]
        for pair, new_price in zip(pairs, old_price[commodity]):
            price_dict[pair] = new_price

    return price_dict


def _assign_supply_and_demand(rng: np.random.Generator, bv, ratio, demand_jitter):
    
    demand_dict = {}
    supply_dict = {}

    baseSupply = {}
    baseDemand = {}
    for comm in commodities:
        for (_, _comm), val in bv.supply_dict.items():
            if comm == _comm:
                baseSupply[comm] = baseSupply.get(comm, 0) + val
    for comm in commodities:
        for (_, _comm), val in bv.demand_dict.items():
            if comm == _comm:
                baseDemand[comm] = baseDemand.get(comm, 0) + val

    scalingFactor = {
        comm: baseSupply[comm] / (ratio * baseDemand[comm]) for comm in commodities
    }

    for comm in commodities:
        parts = split_into(rng, 3, baseSupply[comm])
        for i in range(3):
            error_supply = rng.uniform(0.95, 1.05)
            _city = rng.choice(commodity_source_cities[comm])
            while (_city, comm) in supply_dict:
                _city = rng.choice(commodity_source_cities[comm])
            supply_dict[(_city, comm)] = parts[i] * error_supply / 100000

    for comm in commodities:
        parts = split_into(rng, 3, baseDemand[comm])
        for i in range(3):
            error_demand = rng.uniform(*demand_jitter)
            _city = rng.choice(commodity_source_cities[comm])
            while (_city, comm) in demand_dict or (_city, comm) in supply_dict:
                _city = rng.choice(commodity_source_cities[comm])
            demand_dict[(_city, comm)] = parts[i] * scalingFactor[comm] * error_demand / 100000

    return demand_dict, supply_dict

# scenario where supply greatly exceeds demand
def AbundantSupply(bv, rng: np.random.Generator):
    ratio = rng.uniform(2.25, 5)
    demand_dict, supply_dict = _assign_supply_and_demand(rng, bv, ratio, demand_jitter=(0.95, 1.05))

    capacity_dict = generate_capacity(
        rng=rng, edges=bv.edges, supply_dict=supply_dict,
        demand_dict=demand_dict, commodities=commodities,
    )
    price_dict = _generate_price(rng, bv, supply_dict)

    return demand_dict, supply_dict, capacity_dict, price_dict


# Create a scenario where demand and supply are drawn close together
def SimulatedBalanced(bv, rng: np.random.Generator):
    ratio = rng.uniform(1.05, 1.3)
    demand_dict, supply_dict = _assign_supply_and_demand(rng, bv, ratio, demand_jitter=(0.85, 1.15))

    capacity_dict = generate_capacity(
        rng=rng, edges=bv.edges, supply_dict=supply_dict,
        demand_dict=demand_dict, commodities=commodities,
    )
    price_dict = _generate_price(rng, bv, supply_dict)

    return demand_dict, supply_dict, capacity_dict, price_dict


def SimulatedUnderSupply(bv, rng: np.random.Generator):
    ratio = rng.uniform(0.6, 0.9)
    demand_dict, supply_dict = _assign_supply_and_demand(rng, bv, ratio, demand_jitter=(0.85, 1.15))

    capacity_dict = generate_capacity(
        rng=rng, edges=bv.edges, supply_dict=supply_dict,
        demand_dict=demand_dict, commodities=commodities,
    )
    price_dict = _generate_price(rng, bv, supply_dict)

    return demand_dict, supply_dict, capacity_dict, price_dict


def SimulatedCapacityConstrained(bv, rng: np.random.Generator):
    ratio = rng.uniform(1.05, 1.35)
    demand_dict, supply_dict = _assign_supply_and_demand(rng, bv, ratio, demand_jitter=(0.85, 1.15))

    capacity_dict = generate_capacity(
        rng=rng, edges=bv.edges, supply_dict=supply_dict,
        demand_dict=demand_dict, commodities=commodities,
    )

    # Make 10-20% of physical edges capacity constrained
    physical_edges = list({
        frozenset((city_a, city_b))
        for city_a, city_b in bv.edges
    })

    constrained_percentage = rng.uniform(0.10, 0.20)
    number_constrained = max(1, int(len(physical_edges) * constrained_percentage))

    constrained_indices = rng.choice(
        len(physical_edges), size=number_constrained, replace=False
    )
    constrained_edges = [physical_edges[i] for i in constrained_indices]

    for edge in constrained_edges:
        city_a, city_b = tuple(edge)
        original_capacity = capacity_dict[(city_a, city_b)]
        constraint_factor = rng.uniform(0.10, 0.30)
        constrained_capacity = original_capacity * constraint_factor
        capacity_dict[(city_a, city_b)] = constrained_capacity
        capacity_dict[(city_b, city_a)] = constrained_capacity

    price_dict = _generate_price(rng, bv, supply_dict)

    return demand_dict, supply_dict, capacity_dict, price_dict