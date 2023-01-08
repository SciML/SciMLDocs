# Parameter Analysis Utilities

## GlobalSensitivity.jl: Global Sensitivity Analysis

Derivatives calculate the local sensitivity of a model, i.e. the change in the simulation's outcome if
one were to change the parameter with respect to some chosen part of the parameter space. But how does a
simulation's output change “in general” with respect to a given parameter? That is what global sensitivity
analysis (GSA) computes, and thus [GlobalSensitivity.jl](https://github.com/SciML/GlobalSensitivity.jl) is
the way to answer that question. GlobalSensitivity.jl includes a wide array of methods, including:

- Morris's method
- Sobol's method
- Regression methods (PCC, SRC, Pearson)
- eFAST
- Delta Moment-Independent method
- Derivative-based Global Sensitivity Measures (DGSM)
- EASI
- Fractional Factorial method
- Random Balance Design FAST method

## StructuralIdentifiability.jl: Identifiability Analysis Made Simple

Performing parameter estimation from a data set means attempting to recover parameters
like reaction rates by fitting some model to the data. But how do you know whether you
have enough data to even consider getting the “correct” parameters back?
[StructuralIdentifiability.jl](https://github.com/SciML/StructuralIdentifiability.jl)
allows for running a structural identifiability analysis on a given model to determine
whether it's theoretically possible to recover the correct parameters. It can state whether
a given type of output data can be used to globally recover the parameters (i.e. only a
unique parameter set for the model produces a given output), whether the parameters are
only locally identifiable (i.e. there are finitely many parameter sets which could generate
the seen data), or whether it's unidentifiable (there are infinitely many parameters which
generate the same output data).

For more information on what StructuralIdentifiability.jl is all about, see the
[SciMLCon 2022 tutorial video](https://www.youtube.com/watch?v=jg1DME3cwjg).

## MinimallyDisruptiveCurves.jl

[MinimallyDisruptiveCurves.jl](https://github.com/SciML/MinimallyDisruptiveCurves.jl) is a library for
finding relationships between parameters of models, finding the curves on which the solution is constant.

# Third-Party Libraries to Note

## SIAN.jl: Structural Identifiability Analyzer

[SIAN.jl](https://github.com/alexeyovchinnikov/SIAN-Julia) is a structural identifiability analysis
package which uses an entirely different algorithm from StructuralIdentifiability.jl. For information
on the differences between the two approaches, see
[the Structural Identifiability Tools in Julia tutorial](https://www.youtube.com/watch?v=jg1DME3cwjg).

## DynamicalSystems.jl: A Suite of Dynamical Systems Analysis

[DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/latest/) is an entire ecosystem
of dynamical systems analysis methods, for computing measures of chaos (dimension estimation, Lyapunov coefficients),
generating delay embeddings, and much more. It uses the SciML tools for its internal equation solving
and thus shares much of its API, adding a layer of new tools for extended analyses.

For more information, watch the [tutorial Introduction to DynamicalSystems.jl](https://www.youtube.com/watch?v=A8g9rdEfdNg).

## BifurcationKit.jl

[BifurcationKit.jl](https://github.com/rveltz/BifurcationKit.jl) is a tool for performing bifurcation analysis.
It uses and composes with many SciML equation solvers.

## ReachabilityAnalysis.jl

[ReachabilityAnalysis.jl](https://github.com/JuliaReach/ReachabilityAnalysis.jl) is a library for performing
reachability analysis of dynamical systems, determining for a given uncertainty interval the full set of
possible outcomes from a dynamical system.

## ControlSystems.jl

[ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl) is a library for building and analyzing
control systems.
