# Equation Solvers Overview

# Third Party Libraries to Note

## JuMP.jl: Julia for Mathematical Programming

While Optimization.jl is the preferred library for nonlinear optimization, for all
other forms of optimization 
[Julia for Mathematical Programming (JuMP)](https://github.com/jump-dev/JuMP.jl) is
the star. JuMP is the leading choice in Julia for doing:

- Linear programming
- Convex programming
- Conic programming

JuMP can also be used for some nonlinear programming, though the Optimization.jl bindings
to the JuMP solvers (via MathOptInterface.jl) is generally preferred.

## FractionalDiffEq.jl: Fractional Differential Equation Solvers

[FractionalDiffEq.jl](https://github.com/SciFracX/FractionalDiffEq.jl) is a set of
high-performance solvers for fractional differential equations.

## ManifoldDiffEq.jl: Solvers for Differential Equations on Manifolds

[ManifoldDiffEq.jl](https://github.com/JuliaManifolds/ManifoldDiffEq.jl) is a set of
high-performance solvers for differential equations on manifolds using methods such
as Lie Group actions and frozen coefficients (Crouch-Grossman methods). These solvers
can in many cases out-perform the OrdinaryDiffEq.jl nonautonomous operator ODE solvers
by using methods specialized on manifold definitions of ManifoldsBase.

## ApproxFun.jl: Automated Spectral Discretizations

[ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl) is a package for
approximating functions in basis sets. One particular use case is with spectral
basis sets, such as Chebyshev functions and Fourier decompositions, making it easy
to represent spectral and pseudospectral discretizations of partial differential equations
as ordinary differential equations for the SciML equation solvers.

## Gridap.jl: Automated Finite Element Discretizations

[Gridap.jl](https://github.com/gridap/Gridap.jl) is a package for grid-based approximation
of partial differential equations, particularly notable for its use of conforming and
nonconforming finite element (FEM) discretizations.

## Trixi.jl: Automated Finite Volume Discretizations for Hyperbolic Equations

[Trixi.jl](https://github.com/trixi-framework/Trixi.jl) is a package for numerical simulation
of hyperbolic conservation laws, i.e. a large set of hyperbolic partial differential equations,
which interfaces and uses the SciML equation solvers for many internal steps.