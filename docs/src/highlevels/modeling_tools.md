# SciML Modeling Libraries

While in theory one can build perfect code for all models from scratch, in practice 
many scientists and engineers need or want some help! The SciML modeling tools
provide a higher level interface over the equation solvers which help the translation
from good models to good simulations in a way that abstracts away the mathematical
and computational details without giving up performance.

## ModelingToolkit.jl: Acausal Symbolic Modeling

[Acausal modeling is an extension of causal modeling](https://arxiv.org/pdf/1909.00484.pdf)
that is more composable and allows for more code reuse. Build a model of an electric engine,
then build a model of a battery, and now declare connections by stating "the voltage at the
engine equals the voltage at the connector of the battery", and generate the composed model.
The tool for this is ModelingToolkit.jl. ModelingToolkit.jl is a sophisticated symbolic
modeling library which allows for specifying these types of large-scale differential equation
models in a simple way, abstracting away the computational details. However, its symbolic
analysis allows for generating much more performant code for differential-algebraic equations
than most users could ever write by hand, with its `structural_simplify` automatically correcting
the model to improve parallelism, numerical stability, and automatically remove variables
which it can show are redundant.

ModelingToolkit.jl is the base of the SciML symbolic modeling ecosystem, defining the `AbstractSystem`
types, such as `ODESystem`, `SDESystem`, `OptimizationSystem`, `PDESystem`, and more, which are
then used by all of the other modeling tools. As such, when using other modeling tools like Catalyst.jl,
the reference for all of the things that can be done with the symbolic representation is simply
ModelingToolkit.jl.

## ModelingToolkitStandardLibrary.jl: A Standard Library for ModelingToolkit

Given the composable nature of acausal modeling systems, it's helpful to not have to define
every component from scratch and instead build off a common base of standard components.
ModelingToolkitStandardLibrary.jl is that library. It provides components for standard models
to start building everything from circuits and engines to robots.

![](https://user-images.githubusercontent.com/1814174/172000112-3579f5cf-c370-48c2-8047-558fbc46aeb6.png)

## Catalyst.jl: Chemical Reaction Networks (CRN), Systems Biology, and Quantiative Systems Pharmacology (QSP) Modeling

[Catalyst.jl](https://github.com/SciML/Catalyst.jl) is a modeling interface for efficient simulation 
of chemical master equation representations chemical reaction networks and other systems models. 
It uses a highly intuitive chemical reaction syntax interface which generates all of the extra 
functionality necessary for the fastest use with DiffEqJump.jl and DifferentialEquations.jl. Its 
`ReactionSystem` type is a programmable extension of the ModelingToolkit `AbstractSystem` interface, 
meaning that complex reaction systems can be generated through code.

For an overview of the library, see 
[Modeling Biochemical Systems with Catalyst.jl - Samuel Isaacson](https://www.youtube.com/watch?v=5p1PJE5A5Jw)

## NBodySimulator.jl: A differentiable simulator for N-body problems, including astrophysical and molecular dynamics

[NBodySimulator.jl](https://github.com/SciML/NBodySimulator.jl) is differentiable simulator for N-body problems, 
including astrophysical and molecular dynamics. It uses the DifferentialEquations.jl solvers, allowing for one to
choose between a large variety of symplectic integration schemes. It implements many of the thermostats required for
doing standard molecular dynamics approximations.

## ParameterizedFunctions.jl: Simple Differential Equation Definitions Made Easy

![](https://user-images.githubusercontent.com/1814174/172001045-b9e35b8d-0d40-41af-b606-95b81bb1194d.png)

This image that went viral is actually runnable code from [ParameterizedFunctions.jl](https://github.com/SciML/ParameterizedFunctions.jl).
Define equations and models using a very simple high level syntax and let the code generation tools build
symbolic fast Jacobian, gradient, etc. functions for you.

# Model Import Libraries

## SBMLToolbox.jl: SBML Import

[SBMLToolbox.jl](https://github.com/SciML/SBMLToolkit.jl) is a library for reading 
[SBML files](https://synonym.caltech.edu/#:~:text=What%20is%20SBML%3F,field%20of%20the%20life%20sciences.)
into the standard formats for Catalyst.jl and ModelingToolkit.jl.

## CellMLToolbox.jl: CellML Import

[CellMLToolbox.jl](https://github.com/SciML/CellMLToolbox.jl) is a library for reading 
[CellML files](https://www.cellml.org/) into the standard formats for ModelingToolkit.jl.

## ReactionNetworkImporters.jl: BioNetGen Import

[ReactionNetworkImporters.jl](https://github.com/SciML/ReactionNetworkImporters.jl) is a library 
for reading [BioNetGen files](https://bionetgen.org/) into the standard formats for Catalyst.jl
and ModelingToolkit.jl.

# Third Party Tools of Note

## ReactionMechanismSimulator.jl: Simulation and Analysis of Large Chemical Reaction Systems

[ReactionMechanismSimulator.jl](https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl)
is a tool for simulating and analyzing large chemical reaction mechanisms. It
interfaces with the ReactionMechanismGenerator suite for automatically constructing reaction pathways
from chemical components to quickly build realistic models of chemical systems.

## MomentClosure.jl: Automated Generation of Moment Closure Equations

[MomentClosure.jl](https://github.com/augustinas1/MomentClosure.jl) is a library for generating the moment
closure equations for a given chemical master equation or stochastic differential equation. Thus instead of
solving a stochastic model thousands of times to find the mean and variance, this library can generate the
deterministic equations for how the mean and variance evolve in order to be solved in a single run. MomentClosure.jl
uses Catalyst `ReactionSystem` and ModelingToolkit `SDESystem` types as the input for its symbolic generation
processes.

## FiniteStateProjection.jl: Direct Solution of Chemical Master Equations

[FiniteStateProjection.jl](https://github.com/kaandocal/FiniteStateProjection.jl) is a library for finite state
projection direct solving of the chemical master equation. It automatically converts the Catayst `ReactionSystem`
definitions into ModelingToolkit `ODESystem` representations for the evolution of probability distributions to
allow for directly solving the weak form of the stochastic model.

## AlgebraicPetri.jl: Applied Category Theory of Modeling

[AlgebraicPetri.jl](https://github.com/AlgebraicJulia/AlgebraicPetri.jl) is a library for automating the intuitive
generation of dynamical models using a Category theory based approach.

## Agents.jl: Agent-Based Modeling Framework in Julia

If one wants to do agent-based modeling in Julia, 
[Agents.jl](https://github.com/JuliaDynamics/Agents.jl) is the go-to library. It's fast and flexible,
making it a solid foundation for any agent-based model.