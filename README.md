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
<img width="926" height="661" alt="network" src="https://github.com/user-attachments/assets/0669a997-f720-44c5-afbb-cfcecfcc97f4" />

# Machine Learning Formulation

The problem is a graph by nature (cities as nodes, routes as edges), which makes a Graph Neural Network the natural choice. A convolutional neural network assumes a regular grid of data, like pixels in an image. A graph has no inherent node ordering, so standard GNN layers are built to be **permutation equivariant**: relabeling the nodes relabels the outputs the same way and changes nothing else.

**Per-Edge Weights (PEW).** The architecture follows *Graph Neural Modeling of Network Flows* (Darvariu, Hailes & Musolesi, University College London, arXiv:2209.05208). PEW extends a graph attention network by giving **each edge its own trainable parameters** instead of sharing one weight matrix across all edges.  PEW is designed for a fixed network topology, so this project keeps a single canonical edge ordering across all samples. That way, edge *e* always refers to the same physical route.

**Physics-informed capacity penalty.** Drawing on Physics-Informed Neural Networks, the training loss adds a soft penalty for predicted flows that exceed edge capacity. This pushes the model toward feasible solutions without hard-coding the constraint.

**Virtual node for global context.** Message passing only spreads information a few hops per layer. As a result, an intermediate node cannot tell how much demand is still unmet elsewhere or how much supply an origin has already used. Following *Neural Message Passing for Quantum Chemistry* (Gilmer et al., 2017), the model adds a **virtual node** connected to every city. The virtual node aggregates and rebroadcasts a summary of the whole graph's state, which gives every node global context.

**Residual connections.** Following ResNet (He et al., 2016), the model uses skip connections between layers to stabilize training as depth increases.

### Results
On a held-out test set, adding an LP-objective term to the training loss cut the model's total cost from 4.2× to 2.9× the LP optimum and its unmet demand from 84% to 58% (the LP leaves 17%), while keeping capacity violations under 0.01%. These results come from a preliminary run on 40k training samples for 15 epochs with no tuning. Next we'll train on the full 82k samples for 30 epochs and tune the loss weights, learning rate and architecture to close the remaining gap to the LP.
*[Pending]*
