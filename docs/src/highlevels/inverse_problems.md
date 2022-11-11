# [Parameter Estimation, Bayesian Analysis, and Inverse Problems](@id parameter_estimation)

Parameter estimation for models and equations, also known as dynamic data analysis,
solving the inverse problem, or Bayesian posterior estimation (when done probabilistically),
is provided by the SciML tools for the equations in its set. In this introduction, we briefly
present the relevant packages that facilitate parameter estimation, namely:

- [SciMLSensitivity.jl](https://sensitivity.sciml.ai/)
- [DiffEqFlux.jl](https://diffeqflux.sciml.ai/)
- [Turing.jl](https://turing.ml/)
- [DataDrivenDiffEq.jl](https://datadriven.sciml.ai/dev/)
- [DiffEqParamEstim.jl](https://diffeqparamestim.sciml.ai/dev/)
- [DiffEqBayes.jl](https://diffeqbayes.sciml.ai/dev/)

We also provide information regarding the respective strengths of these packages
so that you can easily decide which one suits your needs best.

## SciMLSensitivity.jl: Local Sensitivity Analysis and Automatic Differentiation Support for Solvers

SciMLSensitivity.jl is the system for local sensitivity analysis which all other inverse problem
methods rely on. This package defines the interactions between the equation solvers and automatic
differentiation, defining fast overloads for forward and adjoint (reverse) sensitivity analysis
for fast gradient and Jacobian calculations with respect to model inputs. Its documentation covers
how to use direct differentiation of equation solvers in conjunction with tools like Optimization.jl
to perform model calibration of ODEs against data, PDE-constrained optimization, nonlinear optimal
controls analysis, and much more. As a lower level tool, this library is very versatile, feature-rich,
and high-performance, giving all of the tools required but not directly providing a higher level
interface.

!!! note

    Sensitivity analysis is kept in a separate library from the solvers (SciMLSensitivity.jl), in
    order to not require all equation solvers to have a dependency on all automatic differentiation
    libraries. If automatic differentiation is applied to a solver library without importing
    SciMLSensitivity.jl, an error is thrown letting the user know to import SciMLSensitivity.jl
    for the functionality to exist.

## DataDrivenDiffEq.jl: Data-Driven Modeling and Equation Discovery

The distinguishing feature of this package is that its ultimate goal is to
identify the differential equation model that generated the input data.
Depending on the user's needs, the package can provide structural identification
of a given differential equation (output in a symbolic form) or structural
estimation (output as a function for prediction purposes).

## DiffEqParamEstim.jl: Simplified Parameter Estimation Interface

This package is for simplified parameter estimation. While not as flexible of a
system like DiffEqFlux.jl, it provides ready-made functions for doing standard
optimization procedures like L2 fitting and MAP estimates. Among other features,
it allows for the optimization of parameters in ODEs, stochastic problems, and
delay differential equations.

## DiffEqBayes.jl: Simplified Bayesian Estimation Interface

As the name suggests, this package has been designed to provide the estimation
of differential equations parameters by means of Bayesian methods. It works in
conjunction with [Turing.jl](https://turing.ml/),
[CmdStan.jl](https://github.com/StanJulia/CmdStan.jl),
[DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl), and
[ApproxBayes.jl](https://github.com/marcjwilliams1/ApproxBayes.jl). While not
as flexible as direct usage of DiffEqFlux.jl or Turing.jl, DiffEqBayes.jl can
be an approachable interface for those not familiar with Bayesian estimation,
and provides a nice way to use Stan from pure Julia.

# Third Party Tools of Note

## Turing.jl: A Flexible Probabilistic Programming Language for Bayesian Analysis

In the context of differential equations and parameter estimation, Turing.jl
allows for a Bayesian estimation of differential equations (used in conjunction
with the high-level package DiffEqBayes.jl). For more examples on combining
Turing.jl with DiffEqBayes.jl, see the documentation below. It is important
to note that Turing.jl can also perform Bayesian estimation without relying on
DiffEqBayes.jl (for an example, consult
[this](https://turing.ml/stable/tutorials/10-bayesian-differential-equations/) tutorial).

## Topopt.jl: Topology Optimization in Julia

[Topopt.jl](https://github.com/JuliaTopOpt/TopOpt.jl) solves topology optimization problems
which are inverse problems on partial differential equations, solving for an optimal domain.

# Recommended Automatic Differentiation Libraries

Solving inverse problems commonly requires using automatic differentiation (AD). SciML includes
extensive support for automatic differentiation throughout its solvers, though some AD libraries
are more tested than others. The following libraries are the current recommendations of the
SciML developers.

## ForwardDiff.jl: Operator-Overloading Forward Mode Automatic Differentiation

[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is a library for operator-overloading
based forward-mode automatic differentiation. It's commonly used as the default method for generating
Jacobians throughout the SciML solver libraries.

!!! note

    Because ForwardDiff.jl uses an operator overloading approach, uses of ForwardDiff.jl require that
    any caches for non-allocating mutating code allows for `Dual` numbers. To allow such code to be
    ForwardDiff.jl-compatible, see [PreallocationTools.jl](https://github.com/SciML/PreallocationTools.jl).

## Enzyme.jl: LLVM-Level Forward and Reverse Mode Automatic Differentiation

[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is an LLVM-level AD library for forward and reverse
automatic differentiation. It supports many features required for high performance, such as being able to
differentiate mutating and interleave compiler optimization with the AD passes. However, it does not support
all of the Julia runtime, and thus some code with many dynamic behaviors and garbage collection (GC) invocations
can be incompatible with Enzyme. Enzyme.jl is quickly becoming the new standard AD for SciML.

## Zygote.jl: Julia-Level Source-to-Source Reverse Mode Automatic Differentiation

[Zygote.jl](https://github.com/FluxML/Zygote.jl) is the current standard user-level reverse-mode automatic
differentiation library for the SciML solvers. User-level means that many library tutorials, like in
SciMLSensitivity.jl and DiffEqFlux.jl, are written showcase user code using Zygote.jl. This is because
Zygote.jl is the AD engine associated with the Flux machine learning library. However, Zygote.jl has many
limitations which limits its performance in equation solver contexts, such as an inability to handle mutation
and introducing many small allocations and type-instabilities. For this reason, the SciML equation
solvers include define differentiation overloads using [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl),
meaning that the equation solvers tend to not use Zygote.jl internally even if the user code uses `Zygote.gradient`.
In this manner, the speed and performance of more advanced techniques can be preserved while using the Julia standard.

## FiniteDiff.jl: Fast Finite Difference Approximations

[FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) is the preferred fallback library for numerical
differentiation and is commonly used by SciML solver libraries when automatic differentiation is disabled.

## SparseDiffTools.jl: Tools for Fast Automatic Differentiation with Sparse Operators

[SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl) is a library for sparse automatic
differentiation. It's used internally by many of the SciML equation solver libraries, which explicitly
expose interfaces for `colorvec` color vectors generated by SparseDiffTools.jl's methods. SparseDiffTools.jl
also includes many features useful to users, such as operators for matrix-free Jacobian-vector and Hessian-vector
products.
