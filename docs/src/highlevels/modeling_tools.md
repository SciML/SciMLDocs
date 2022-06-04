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

## Catalyst.jl: 

## NBodySimulator.jl:

## ParameterizedFunctions.jl: Simple Differential Equation Definitions Made Easy

![](https://user-images.githubusercontent.com/1814174/172001045-b9e35b8d-0d40-41af-b606-95b81bb1194d.png)

This image that went viral is actually runnable code from [ParameterizedFunctions.jl](https://github.com/SciML/ParameterizedFunctions.jl).
Define equations and models using a very simple high level syntax and let the code generation tools build
symbolic fast Jacobian, gradient, etc. functions for you.

# Third Party Tools of Note

## [ReactionMechanismSimulator.jl](https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl)

ReactionMechanismSimulator.jl is a tool for simulating and analyzing large chemical reaction mechanisms. It
interfaces with the ReactionMechanismGenerator suite for automatically constructing reaction pathways
from chemical components to quickly build realistic models of chemical systems.

## [MomentClosure.jl](https://github.com/augustinas1/MomentClosure.jl)

## [FiniteStateProjection.jl](https://github.com/kaandocal/FiniteStateProjection.jl)