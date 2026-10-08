# NetworkFlowMLProject
# Business Problem

A classic problem in Operations Research is the Multi-Commodity Network Flow (MCNF) problem; it appears throughout logistics, telecommunications, and supply chain planning. The goal is to route several distinct commodities through a shared network at minimum cost (or maximum utilization) while respecting constraints such as edge capacities and supply limits.

This project models a synthetic U.S. freight network, grounded in historical data from the U.S. Census Bureau and USDA NASS.

Problem statement: Five agricultural commodities each have a supply ceiling and are sourced from three supplier cities per commodity. Each commodity is demanded by three cities. Every transportation edge has a capacity limit, and each commodity's price per ton varies by supplier city. The objective is to minimize the total cost of meeting demand while respecting edge capacities and supplier limits.

## Approach

MCNF is a well-understood linear program, and it is normally solved with a mathematical optimization solver. Here the model is written in Pyomo (a Python optimization modeling language) and solved with HiGHS (an open-source LP solver).

The goal of this project is to attack the same problem with a Graph Neural Network. Solvers already return exact optimal solutions, so the question is whether a modern neural architecture can learn to approximate them, and what the tradeoffs would be: inference speed, accuracy, feasibility, and generalization.

## Formulation

### Linear Program

The problem uses a standard MCNF linear program with two slack variables:

- Unmet demand is penalized heavily in the objective, so the solver ships whenever it is feasible to do so.
- Unused supply carries no penalty, so suppliers do not have to ship their full capacity.

The slack variables keep every instance feasible, even when aggregate supply and demand don't balance.

*[Full mathematical formulation to be added.]*

### Training Data: Regime-Based Stochastic Simulation

A regime-based stochastic simulator produced 100,000 LP instances for training. Each sample is first assigned one of four market regimes:

| Regime | Expected LP behavior |
|---|---|
| Abundant supply | Large unused supply, demand fully met |
| Balanced | Tight but feasible network |
| Undersupply | Some demand unmet |
| Capacity-constrained | Edge capacities bind before supply runs out |

Supply, demand, capacity, and price are then drawn conditionally on the assigned regime. This guarantees the dataset covers a deliberate range of optimization outcomes rather than clustering around one baseline.

Each instance is solved with Pyomo/HiGHS, and the optimal flows serve as training targets. Results are stored in sharded, sparse (COO-encoded) HDF5 files.

## Network

Full image of the network included below.
