# SciML Simulation Analysis Utilities

## GlobalSensitivity.jl: Global Sensitivity Analysis

Derivatives calculate the local sensitivity of a model, i.e. the change in the simulation's outcome if
one was to change the parameter with respect to some chosen part of the parameter space. But how does a
simulation's output change "in general" with respect to a given parameter? That is what global sensitivity
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

## MinimallyDisruptiveCurves.jl

[MinimallyDisruptiveCurves.jl](https://github.com/SciML/MinimallyDisruptiveCurves.jl) is a library for
finding relationships between parameters of models, finding the curves on which the solution is constant.

# Third Party Libraries to Note

## DynamicalSystems.jl: A Suite of Dynamical Systems Analysis

[DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/latest/) is an entire ecosystem
of dynamical systems analysis methods, for computing measures of chaos (dimension estimation, lyopunov coefficients),
generating delay embeddings, and much more. It uses the SciML tools for its internal equation solving
and thus shares much of its API, adding a layer of new tools for extended analyses.

For more information, watch the [tutorial Introduction to DynamicalSystems.jl](https://www.youtube.com/watch?v=A8g9rdEfdNg).

## BifurcationKit.jl

[BifucationKit.jl](https://github.com/rveltz/BifurcationKit.jl) is a tool for performing bifurcation analysis.
It uses and composes with many SciML equation solvers.

## ReachabilityAnalysis.jl

[ReachabilityAnalysis.jl](https://github.com/JuliaReach/ReachabilityAnalysis.jl) is a library for performing
reachability analysis of dynamical systems, determining for a given uncertainty interval the full set of
possible outcomes from a dynamical system.

## ControlSystems.jl

[ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl) is a library for building and analyzing
control systems.

# Recommended Plotting and Visualization Libraries

## Plots.jl

[Plots.jl](https://github.com/JuliaPlots/Plots.jl) is the current standard plotting system for the SciML ecosystem.
SciML types attempt to include plot recipes for as many types as possible, allowing for automatic visualization with
the Plots.jl system. All current tutorials and documentation default to using Plots.jl

## Makie.jl

[Makie.jl](https://makie.juliaplots.org/stable/) is a high-performance interactive plotting system for the Julia
programming language. It's planned to be the default plotting system used by the SciML organization in the near future.