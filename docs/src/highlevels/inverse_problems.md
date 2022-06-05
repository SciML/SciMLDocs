# [Parameter Estimation, Bayesian Analysis, and Inverse Problems Overview](@id parameter_estimation)

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
to perform model callibration of ODEs against data, PDE-constrained optimization, nonlinear optimal
controls analysis, and much more. As a lower level tool, this library is very versitile, feature-rich,
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
optmization procedures like L2 fitting and MAP estimates. Among other features,
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

### Turing.jl: A Flexible Probabilistic Programming Language for Bayesian Analysis

In the context of differential equations and parameter estimation, Turing.jl
allows for a Bayesian estimation of differential equations (used in conjunction
with the high-level package DiffEqBayes.jl). For more examples on combining
Turing.jl with DiffEqBayes.jl, see the documentation below. It is important
to note that Turing.jl can also perform Bayesian estimation without relying on
DiffEqBayes.jl (for an example, consult 
[this](https://turing.ml/stable/tutorials/10-bayesian-differential-equations/) tutorial).
