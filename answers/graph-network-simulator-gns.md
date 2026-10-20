# The Graph Network-based Simulator (GNS)

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** The Graph Network-based Simulator (Sanchez-Gonzalez et al., ICML 2020, "Learning to Simulate Complex Physics with Graph Networks") learns to simulate particle systems: fluids, sand, viscous "goop" and rigid or deformable solids interacting with each other. Each particle is a node of a graph that is **rebuilt at every time step** by connecting particles closer than a connectivity radius. An **encoder** embeds each particle's recent velocity history, its material type and its distance to the walls, plus relative displacements on the edges; a **processor** of several message-passing steps lets particles exchange information; a **decoder** predicts each particle's **acceleration**, which a simple semi-implicit Euler step turns into new velocities and positions. Trained only on single-step predictions with **noise added to the input states**, GNS generalized to thousands of steps, new initial conditions and at least an order of magnitude more particles than in training. The paper identified two main determinants of long-term accuracy: the number of message-passing steps and the noise injection.

**On this page**

- [What GNS simulates](#what-gns-simulates)
- [Building the graph](#building-the-graph)
- [Encoder, processor, decoder](#encoder-processor-decoder)
- [Integrating the prediction](#integrating-the-prediction)
- [Training with noise](#training-with-noise)
- [Code: GNS inputs and the integrator](#code-gns-inputs-and-the-integrator)
- [What GNS showed, and its limits](#what-gns-showed-and-its-limits)
- [In an interview](#in-an-interview)

## What GNS simulates

Lagrangian particle methods represent a material by moving particles that carry mass, velocity and other properties: smoothed particle hydrodynamics (SPH) for fluids with free surfaces and splashes, the material point method (MPM) for sand and snow, position-based dynamics for games and graphics, the discrete element method for granular media. They handle large deformations, free surfaces, fragmentation and mixing naturally, but they are expensive: many particles, many neighbours, small time steps.

GNS learns the time-stepping of such simulators from their trajectories. The paper trained it on water, sand, goop and rigid-body data from several simulators (including SPH and MPM-based ones), in 2D and 3D, with up to tens of thousands of particles.

## Building the graph

At every time step:

1. **Nodes** are the particles (plus, optionally, particles representing static walls or obstacles).
2. **Edges** connect every pair of particles closer than a **connectivity radius** R, in both directions. Because particles move, the graph changes every step and is rebuilt with a spatial data structure (a k-d tree or a cell list). See [k-nearest-neighbour and radius graphs](mesh-data-processing.md#k-nearest-neighbour-and-radius-graphs).
3. **The radius is a key hyperparameter:** large enough that each particle has a reasonable number of neighbours (tens), small enough to keep the graph sparse.

This is the difference from MeshGraphNets, where the graph comes from a fixed simulation mesh (plus world edges for contact). See [MeshGraphNets](gnns-and-meshgraphnets.md#meshgraphnets-encoder-processor-decoder).

## Encoder, processor, decoder

**Encoder.** MLPs embed:

- **node features:** the particle's last C velocities (C = 5 in the paper; velocities are finite differences of positions), its material type (a learned embedding), and its distance to the domain boundaries, clipped at the radius;
- **edge features:** the relative displacement between the two particles, x_i − x_j, and its length (both normalized by the radius).

Absolute positions are deliberately left out (except through wall distances), which makes the model translation-invariant: shifting the whole scene changes nothing.

**Processor.** M rounds of message passing with graph-network blocks, each with its own weights and residual connections (the paper used M = 10 as a default and studied the effect of M). Through these rounds, a particle "sees" particles up to M hops away, which lets pressure-like interactions propagate through a fluid.

**Decoder.** An MLP maps each particle's final latent vector to its **acceleration** (normalized with training statistics).

## Integrating the prediction

The predicted acceleration is integrated with a **semi-implicit Euler** step (with the simulator's output interval taken as Δt = 1 in normalized units):

```math
v_{t+1} = v_t + a_t\,\Delta t, \qquad x_{t+1} = x_t + v_{t+1}\,\Delta t
```

Predicting acceleration rather than the next position directly builds in the structure of Newtonian dynamics: inertia is handled exactly by the integrator, and the network only has to learn the forces. The rollout repeats: rebuild the graph, encode, process, decode, integrate.

## Training with noise

**The problem:** the model is trained to predict one step from *true* states, but during a rollout it receives its *own* predictions, which contain small errors. Those inputs are slightly off the training distribution, the next prediction is worse, and errors compound.

**The GNS remedy:** corrupt the input velocities (and positions) during training with **random-walk noise**, accumulated over the input history, and train the model to predict the acceleration that brings the noisy state back to the true next state. The model learns to correct small errors instead of amplifying them. The paper found this, together with the number of message-passing steps, to be the main determinant of long-term rollout quality, and the model was otherwise robust to hyperparameter choices. See [Noise injection](autoregressive-rollouts.md#noise-injection).

## Code: GNS inputs and the integrator

Building the per-step inputs (velocity history, wall distances, radius graph with relative edge features) and the integration step, with a stand-in for the network's predicted acceleration. The translation check confirms the inputs carry no absolute position.

```python
import numpy as np
from scipy.spatial import cKDTree

def gns_inputs(positions, radius, bounds, n_hist=5):
    """Build GNS-style inputs from a short trajectory of particle positions (T, N, dim).
    Node features: the last n_hist velocities and distances to the walls; edges: all pairs within radius,
    with relative displacement and its length. Nothing depends on absolute position except the walls."""
    vel = np.diff(positions[-(n_hist + 1):], axis=0)                 # (n_hist, N, dim) finite-difference velocities
    x = positions[-1]
    node = np.concatenate([vel.transpose(1, 0, 2).reshape(len(x), -1),
                           np.clip(x - bounds[0], 0, radius), np.clip(bounds[1] - x, 0, radius)], axis=1)
    pairs = cKDTree(x).query_pairs(radius, output_type="ndarray")    # undirected pairs within the radius
    senders = np.concatenate([pairs[:, 0], pairs[:, 1]])
    receivers = np.concatenate([pairs[:, 1], pairs[:, 0]])           # both directions
    rel = (x[senders] - x[receivers]) / radius
    edge = np.concatenate([rel, np.linalg.norm(rel, axis=1, keepdims=True)], axis=1)
    return node, edge, np.stack([senders, receivers])

def semi_implicit_euler(x, v, acceleration, dt=1.0):
    """GNS integrates the predicted acceleration: v_{t+1} = v_t + a dt, x_{t+1} = x_t + v_{t+1} dt."""
    v_next = v + acceleration * dt
    return x + v_next * dt, v_next

rng = np.random.default_rng(0)
traj = np.cumsum(rng.normal(0, 0.001, (6, 500, 2)), axis=0) + rng.random((1, 500, 2))   # 6 frames, 500 particles
bounds = (np.zeros(2), np.ones(2))
node, edge, edge_index = gns_inputs(traj, radius=0.05, bounds=bounds)
print("node features", node.shape, "edge features", edge.shape, "edges", edge_index.shape[1])

shift = np.array([0.2, -0.1])                                        # translate particles AND walls
node_s, edge_s, _ = gns_inputs(traj + shift, radius=0.05, bounds=(bounds[0] + shift, bounds[1] + shift))
translation_invariant = np.allclose(node, node_s) and np.allclose(edge, edge_s)
print("translation invariant:", translation_invariant)

acc = np.zeros((500, 2)); acc[:, 1] = -1e-4                          # stand-in for the network's output (gravity-like)
x_next, v_next = semi_implicit_euler(traj[-1], traj[-1] - traj[-2], acc)
print("next positions", x_next.shape)
```

Output:

```
node features (500, 14) edge features (1910, 3) edges 1910
translation invariant: True
next positions (500, 2)
```

The network between `gns_inputs` and `semi_implicit_euler` is an encoder, a stack of message-passing layers such as the one in [Code: one message-passing layer](gnns-and-meshgraphnets.md#code-one-message-passing-layer), and an MLP decoder to two acceleration components per particle.

## What GNS showed, and its limits

**Results reported in the paper:**

- one model architecture worked across fluids, rigid solids and deformable materials interacting with each other;
- trained on single-step predictions with thousands of particles, it generalized to **thousands of time steps**, new initial conditions and **at least an order of magnitude more particles** at test time;
- robust to most hyperparameters, with the number of message-passing steps and the training noise mattering most.

**Limits to keep in mind:**

- **Cost of graph building and message passing** grows with the number of particles and neighbours; for millions of particles, efficient neighbour search and multi-GPU partitioning are needed.
- **Conservation is not guaranteed:** momentum and energy are not built in (later work adds symmetries and conservation, for example equivariant networks such as SEGNN, benchmarked in LagrangeBench).
- **Long rollouts of chaotic flows** still drift from the reference; the model reproduces plausible behaviour and statistics more than exact trajectories. See [What a surrogate can predict for a chaotic flow](autoregressive-rollouts.md#what-a-surrogate-can-predict-for-a-chaotic-flow).
- **Generalization has limits:** new materials, much larger domains or very different boundary configurations are out of distribution.

**Benchmarks:** LagrangeBench (Toshev et al., NeurIPS 2023) provides seven SPH datasets with JAX implementations of GNS and SEGNN and physical metrics (kinetic-energy error, Sinkhorn distance between particle distributions). See [Other mesh and particle datasets](public-simulation-datasets.md#other-mesh-and-particle-datasets).

## In an interview

**Thirty-second answer to "How does the Graph Network-based Simulator model particle systems?"** "Each particle is a node; at every step the graph is rebuilt by connecting particles within a radius. The encoder embeds each particle's last few velocities, its material type and its distance to the walls, plus relative displacements on edges, so the model is translation-invariant. About ten message-passing steps exchange information between neighbours, a decoder predicts each particle's acceleration, and a semi-implicit Euler step updates velocity and position. It is trained on one-step targets with random-walk noise added to the inputs, which is what keeps thousands-of-steps rollouts stable, and it generalized to many more particles than it was trained on."

**Likely follow-ups**

- *Why predict acceleration rather than position?* The integrator handles inertia exactly; the network only learns forces, a simpler and better-normalized target.
- *GNS vs MeshGraphNets?* GNS rebuilds a radius graph of particles every step; MeshGraphNets uses the simulation mesh (plus world edges for contact).
- *What mattered most for long rollouts?* The number of message-passing steps and the noise injected during training.

## References

- [Learning to Simulate Complex Physics with Graph Networks (Sanchez-Gonzalez et al., 2020)](https://arxiv.org/abs/2002.09405) and [code](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate)
- [Relational inductive biases, deep learning, and graph networks (Battaglia et al.)](https://arxiv.org/abs/1806.01261)
- [LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite (Toshev et al.)](https://arxiv.org/abs/2309.16342)
- [Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets)](https://arxiv.org/abs/2010.03409)
