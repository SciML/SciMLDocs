# Modeling Languages

While in theory one can build perfect code for all models from scratch, in practice
many scientists and engineers need or want some help! The SciML modeling tools
provide a higher level interface over the equation solver, which helps the translation
from good models to good simulations in a way that abstracts away the mathematical
and computational details without giving up performance.

## ModelingToolkit.jl: Acausal Symbolic Modeling

[Acausal modeling is an extension of causal modeling](https://arxiv.org/pdf/1909.00484.pdf)
that is more composable and allows for more code reuse. Build a model of an electric engine,
then build a model of a battery, and now declare connections by stating "the voltage at the
engine equals the voltage at the connector of the battery", and generate the composed model.
The tool for this is [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/).
ModelingToolkit.jl is a sophisticated symbolic modeling library which allows for specifying
these types of large-scale differential equation models in a simple way, abstracting away
the computational details. However, its symbolic analysis allows for generating much more
performant code for differential-algebraic equations than most users could ever write by hand,
with its `structural_simplify` automatically correcting the model to improve parallelism,
numerical stability, and automatically remove variables which it can show are redundant.

ModelingToolkit.jl is the base of the SciML symbolic modeling ecosystem, defining the `AbstractSystem`
types, such as `ODESystem`, `SDESystem`, `OptimizationSystem`, `PDESystem`, and more, which are
then used by all the other modeling tools. As such, when using other modeling tools like Catalyst.jl,
the reference for all the things that can be done with the symbolic representation is simply
ModelingToolkit.jl.

## Catalyst.jl: Chemical Reaction Networks (CRN), Systems Biology, and Quantitative Systems Pharmacology (QSP) Modeling

[Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/) is a modeling interface for efficient simulation
of mass action ODE, chemical Langevin SDE, and stochastic chemical kinetics jump process (i.e. chemical
master equation) models for chemical reaction networks and population processes. It uses a
highly intuitive chemical reaction syntax interface, which generates all the extra functionality
necessary for the fastest use with JumpProcesses.jl, DifferentialEquations.jl, and higher level SciML
libraries. Its `ReactionSystem` type is a programmable extension of the ModelingToolkit `AbstractSystem`
interface, meaning that complex reaction systems are represented symbolically, and then compiled to
optimized representations automatically when converting `ReactionSystem`s to concrete ODE/SDE/jump process
representations. Catalyst also provides functionality to support chemical reaction network and steady-state analysis.

For an overview of the library, see
[Modeling Biochemical Systems with Catalyst.jl - Samuel Isaacson](https://www.youtube.com/watch?v=5p1PJE5A5Jw)

## NBodySimulator.jl: A differentiable simulator for N-body problems, including astrophysical and molecular dynamics

[NBodySimulator.jl](https://docs.sciml.ai/NBodySimulator/stable/) is a differentiable simulator for N-body problems,
including astrophysical and molecular dynamics. It uses the DifferentialEquations.jl solvers, allowing for one to
choose between a large variety of symplectic integration schemes. It implements many of the thermostats required for
doing standard molecular dynamics approximations.

## DiffEqFinancial.jl: Financial models for use in the DifferentialEquations ecosystem

The goal of [DiffEqFinancial.jl](https://github.com/SciML/DiffEqFinancial.jl/commits/master) is to be a feature-complete set
of solvers for the types of problems found in libraries like QuantLib, such as the Heston process or the
Black-Scholes model.

## ParameterizedFunctions.jl: Simple Differential Equation Definitions Made Easy

![](https://user-images.githubusercontent.com/1814174/172001045-b9e35b8d-0d40-41af-b606-95b81bb1194d.png)

This image that went viral is actually runnable code from [ParameterizedFunctions.jl](https://docs.sciml.ai/ParameterizedFunctions/stable/). Define equations and models using a very simple high-level syntax and let the code generation tools build symbolic fast Jacobian, gradient, etc. functions for you.

# Third-Party Tools of Note

## MomentClosure.jl: Automated Generation of Moment Closure Equations

[MomentClosure.jl](https://github.com/augustinas1/MomentClosure.jl) is a library for generating the moment
closure equations for a given chemical master equation or stochastic differential equation. Thus instead of
solving a stochastic model thousands of times to find the mean and variance, this library can generate the
deterministic equations for how the mean and variance evolve in order to be solved in a single run. MomentClosure.jl
uses Catalyst `ReactionSystem` and ModelingToolkit `SDESystem` types as the input for its symbolic generation
processes.

## Agents.jl: Agent-Based Modeling Framework in Julia

If one wants to do agent-based modeling in Julia,
[Agents.jl](https://github.com/JuliaDynamics/Agents.jl) is the go-to library. It's fast and flexible,
making it a solid foundation for any agent-based model.

## Unitful.jl: A Julia package for physical units

Supports not only SI units, but also any other unit system.
[Unitful.jl](https://painterqubits.github.io/Unitful.jl/stable/) has minimal run-time penalty of units.
Includes facilities for dimensional analysis, and integrates easily with the usual mathematical operations and collections that are defined in Julia.

## ReactionMechanismSimulator.jl: Simulation and Analysis of Large Chemical Reaction Systems

[ReactionMechanismSimulator.jl](https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl)
is a tool for simulating and analyzing large chemical reaction mechanisms. It
interfaces with the ReactionMechanismGenerator suite for automatically constructing reaction pathways
from chemical components to quickly build realistic models of chemical systems.

## FiniteStateProjection.jl: Direct Solution of Chemical Master Equations

[FiniteStateProjection.jl](https://docs.sciml.ai/FiniteStateProjection/dev/) is a library for finite state
projection direct solving of the chemical master equation. It automatically converts the Catalyst `ReactionSystem`
definitions into ModelingToolkit `ODESystem` representations for the evolution of probability distributions to
allow for directly solving the weak form of the stochastic model.

## AlgebraicPetri.jl: Applied Category Theory of Modeling

[AlgebraicPetri.jl](https://github.com/AlgebraicJulia/AlgebraicPetri.jl) is a library for automating the intuitive
generation of dynamical models using a Category theory-based approach.

## QuantumOptics.jl: Simulating quantum systems.

[QuantumOptics.jl](https://docs.qojulia.org/) makes it easy to simulate various kinds of quantum systems.
It is inspired by the Quantum Optics Toolbox for MATLAB and the Python framework QuTiP.
