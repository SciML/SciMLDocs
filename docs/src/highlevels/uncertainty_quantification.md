# Uncertainty Quantification Overview

There's always uncertainty in our models. Whether it's in the form of the model's equations
or in the model's parameters, the uncertainty in our simulation's output often needs to be
quantified. The following tools automate this process.

For Measurements.jl vs MonteCarloMeasurements.jl vs Intervals.jl, and the relation to other
methods, see [the Uncertainty Programming chapter of the SciML Book](https://book.sciml.ai/notes/19/).

## PolyChaos.jl: Intrusive Polynomial Chaos Expansions Made Unintrusive

[PolyChaos.jl](https://github.com/SciML/PolyChaos.jl) is a library for calculating
intrusive polynomial chaos expansions (PCE) on arbitrary Julia functions. This allows
for inputing representations of probability distributions into functions to compute
the output distribution in an expansion representation. While normally this would require
deriving the PCE-expanded equations by hand, PolyChaos.jl does this at the compiler level
using Julia's multiple dispatch, giving a high-performance implementation to a normally
complex and tedious mathematical transformation.

## DiffEqUncertainty.jl: Fast Calculations of Expectations of Equation Solutions

[DiffEqUncertainty.jl](https://github.com/SciML/DiffEqUncertainty.jl) is a library for
accelerating the calculation of expectations of equation solutions with respect to
input probability distributions, allowing for applications like robust optimization
with respect to uncertainty. It uses Koopman operator techniques to calculate these
expectations without requiring the propagation of uncertainties through a solver,
effectively performing the adjoint of uncertainty quantification and being much more
efficient in the process.

Additionally, DiffEqUncertainty.jl has the ProbInts method for generating stochastic
equations to mimic the error distribution of an ODE solver in order to quantify with 
respect to numerical error.

# Third Party Libraries to Note

## Measurements.jl: Automated Linear Error Propagation

[Measurements.jl](https://github.com/JuliaPhysics/Measurements.jl) is a library for
automating linear error propagation. Uncertain numbers are defined as `x = 3.8 ± 0.4`
and are pushed through calculations using a normal distribution approximation in order
to compute an approximate uncertain output. Measurements.jl uses a dictionary-based
approach to keep track of correlations to improve the accuracy over naive implementations,
though note that linear error propagation theory still has some major issues handling
some types of equations 
[as described in detail in the MonteCarloMeasurements.jl documentation](https://baggepinnen.github.io/MonteCarloMeasurements.jl/v1.0/comparison/).

## MonteCarloMeasurements.jl: Automated Monte Carlo Error Propogation

[MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl)
is a library for automating the uncertainty quantification of equation solution using
Monte Carlo methods. It defines number types which sample from an input distribution
to receive a representative set of parameters that propagate through the solver to
calculate a representative set of possible solutions. Note that Monte Carlo techniques
can be expensive but are exact, in the sense that as the number of sample points increases
to infinity it will compute a correct approximation of the output uncertainty.

## ProbNumDiffEq.jl: Probabilstic Numerics Based Differential Equation Solvers

[ProbNumDiffEq.jl](https://github.com/nathanaelbosch/ProbNumDiffEq.jl) is a a set of
probabilistic numerical ODE solvers which compute the solution of a differential
equation along with a posterior distribution to estimate its numerical approximation
error. Thus these specialized integrators compute an uncertainty output similar to
the ProbInts technique of DiffEqUncertainty, but use specialized integration techniques
in order to do it much faster for specific kinds of equations.

## TaylorIntegration.jl: Taylor Series Integration for Rigorous Numerical Bounds

[TaylorIntegration.jl](https://github.com/PerezHz/TaylorIntegration.jl) is a library
for Taylor series integrators which has special functionality for computing the
interval bound of possible solutions with respect to numerical approximation error.

## IntervalArithmetic.jl: Rigorous Numerical Intervals

[IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl) is a
library for performing interval arithmetic calculations on arbitrary Julia code. Interval
arithmetic computes rigorous computations with respect to finite-precision floating point
arithmetic, i.e. its intervals are guarenteed to include the true solution. However,
interval arithmetic intervals can grow at exponential rates in many problems, thus being
unsuitable for analyses in many equation solver contexts.