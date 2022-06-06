# Equation Solvers Overview

The SciML Equation Solvers cover a large set of `SciMLProblem`s with `SciMLAlgorhtm`s
that are efficient, numerically stable, and flexible. These methods tie into libraries
like SciMLSensitivity.jl to be fully differentiable and compatible with machine
learning pipelines, and are designed for integration with applications like parameter
estimation, global sensitivity analysis, and more.

## LinearSolve.jl: Unified Interface for Linear Solvers

[LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) is the canonical library
for solving `LinearProblem`s. It includes:

- Fast pure Julia LU factorizations which outperform standard BLAS
- KLU for faster sparse LU factorization on unstructured matrices
- UMFPACK for faster sparse LU factorization on matrices with some repeated structure
- MKLPardiso wrappers for handling many sparse matrices faster than SuiteSparse (KLU, UMFPACK) methods
- GPU-offloading for large dense matrices
- Wrappers to all of the Krylov implementations (Krylov.jl, IterativeSolvers.jl, KrylovKit.jl) for easy
  testing of all of them. LinearSolve.jl handles the API differences, especially with the preconditioner
  definitions
- A polyalgorithm that smartly chooses between these methods
- A caching interface which automates caching of symbolic factorizations and numerical factorizations
  as optimally as possible
- Compatible with arbitrary AbstractArray and Number types, such as GPU-based arrays, uncertainty
  quantification number types, and more.

## NonlinearSolve.jl: Unified Interface for Nonlinear Solvers

[NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) is the canonical library
for solving `NonlinearProblem`s. It includes:

- Fast non-allocating implementations on static arrays of common methdos (Newton-Rhapson)
- Bracketing methods (Bisection, Falsi) for methods with known upper and lower bounds
- Wrappers to common other solvers (NLsolve.jl, MINPACK, KINSOL from Sundials) for trust
  region methods, line search based approaches, etc.
- Built over the LinearSolve.jl API for maximum flexibility and performance in the solving
  approach
- Compatible with arbitrary AbstractArray and Number types, such as GPU-based arrays, uncertainty
  quantification number types, and more.

## DifferentialEquations.jl: Unified Interface for Differential Equation Solvers

[DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) is the canonical library
for solving `DEProblem`s. This includes:

- Discrete equations (function maps, discrete stochastic (Gillespie/Markov)
simulations) (`DiscreteProblem`)
- Ordinary differential equations (ODEs) (`ODEProblem`)
- Split and Partitioned ODEs (Symplectic integrators, IMEX Methods) (`SplitODEProblem`)
- Stochastic ordinary differential equations (SODEs or SDEs) (`SDEProblem`)
- Stochastic differential-algebraic equations (SDAEs) (`SDEProblem` with mass matrices)
- Random differential equations (RODEs or RDEs) (`RODEProblem`)
- Differential algebraic equations (DAEs) (`DAEProblem` and `ODEProblem` with mass matrices)
- Delay differential equations (DDEs) (`DDEProblem`)
- Neutral, retarded, and algebraic delay differential equations (NDDEs, RDDEs, and DDAEs)
- Stochastic delay differential equations (SDDEs) (`SDDEProblem`)
- Experimental support for stochastic neutral, retarded, and algebraic delay differential equations (SNDDEs, SRDDEs, and SDDAEs)
- Mixed discrete and continuous equations (Hybrid Equations, Jump Diffusions) (`DEProblem`s with callbacks and `JumpProblem`)

The well-optimized DifferentialEquations solvers benchmark as some of the fastest
implementations of classic algorithms. It also includes algorithms from recent
research which routinely outperform the "standard" C/Fortran methods, and algorithms
optimized for high-precision and HPC applications. Simultaneously, it wraps
the classic C/Fortran methods, making it easy to switch over to them whenever
necessary. Solving differential equations with different methods from
different languages and packages can be done by changing one line of code,
allowing for easy benchmarking to ensure you are using the fastest method possible.

DifferentialEquations.jl integrates with the Julia package sphere with:

- GPU acceleration through CUDAnative.jl and CuArrays.jl
- Automated sparsity detection with [SparsityDetection.jl](https://github.com/JuliaDiffEq/SparsityDetection.jl)
- Automatic Jacobian coloring with [SparseDiffTools.jl](https://github.com/JuliaDiffEq/SparseDiffTools.jl), allowing for fast solutions
  to problems with sparse or structured (Tridiagonal, Banded, BlockBanded, etc.) Jacobians
- Allowing the specification of linear solvers for maximal efficiency
- Progress meter integration with the Juno IDE for estimated time to solution
- Automatic plotting of time series and phase plots
- Built-in interpolations
- Wraps for common C/Fortran methods, like Sundials and Hairer's radau
- Arbitrary precision with BigFloats and Arbfloats
- Arbitrary array types, allowing the definition of differential equations on
  matrices and distributed arrays
- Unit-checked arithmetic with Unitful

## Optimization.jl: Unified Interface for Optimization

[Optimization.jl](https://github.com/SciML/Optimization.jl) is the canonical library
for solving `OptimizationProblem`s. It includes wrappers of most of the Julia nonlinear
optimization ecosystem, allowing one syntax to use all packages in a uniform manner. This
covers:

- OptimizationBBO for [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl)
- OptimizationEvolutionary for [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl) (see also [this documentation](https://wildart.github.io/Evolutionary.jl/dev/))
- OptimizationGCMAES for [GCMAES.jl](https://github.com/AStupidBear/GCMAES.jl)
- OptimizationMOI for [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) (usage of algorithm via MathOptInterface API; see also the API [documentation](https://jump.dev/MathOptInterface.jl/stable/))
- OptimizationMetaheuristics for [Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl) (see also [this documentation](https://jmejia8.github.io/Metaheuristics.jl/stable/))
- OptimizationMultistartOptimization for [MultistartOptimization.jl](https://github.com/tpapp/MultistartOptimization.jl) (see also [this documentation](https://juliahub.com/docs/MultistartOptimization/cVZvi/0.1.0/))
- OptimizationNLopt for [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) (usage via the NLopt API; see also the available [algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/))
- OptimizationNOMAD for [NOMAD.jl](https://github.com/bbopt/NOMAD.jl) (see also [this documentation](https://bbopt.github.io/NOMAD.jl/stable/))
- OptimizationNonconvex for [Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl) (see also [this documentation](https://julianonconvex.github.io/Nonconvex.jl/stable/))
- OptimizationQuadDIRECT for [QuadDIRECT.jl](https://github.com/timholy/QuadDIRECT.jl)
- OptimizationSpeedMapping for [SpeedMapping.jl](https://github.com/NicolasL-S/SpeedMapping.jl) (see also [this documentation](https://nicolasl-s.github.io/SpeedMapping.jl/stable/))

## Integrals.jl: Unified Interface for Numerical Integration

[Integrals.jl](https://github.com/SciML/Integrals.jl) is the canonical library
for solving `IntegralsProblem`s. It includes wrappers of most of the Julia quadrature 
ecosystem, allowing one syntax to use all packages in a uniform manner. This
covers:

- Gauss-Kronrod quadrature
- Cubature methods (both `h` and `p` cubature)
- Adaptive Monte Carlo methods

## DiffEqJump.jl: Unified Interface for Jump Processes

[DiffEqJump.jl](https://github.com/SciML/DiffEqJump.jl) is the library for Poisson jump
processes, also known as chemical master equations or Gillespie simulations, for simulating
chemical reaction networks and other applications. It allows for solving with many methods,
including:

- `Direct`: the Gillespie Direct method SSA.
- `RDirect`: A variant of Gillespie's Direct method that uses rejection to
  sample the next reaction.
- *`DirectCR`*: The Composition-Rejection Direct method of Slepoy et al. For
  large networks and linear chain-type networks it will often give better
  performance than `Direct`. (Requires dependency graph, see below.)
- `DirectFW`: the Gillespie Direct method SSA with `FunctionWrappers`. This
  aggregator uses a different internal storage format for collections of
  `ConstantRateJumps`. 
- `FRM`: the Gillespie first reaction method SSA. `Direct` should generally
  offer better performance and be preferred to `FRM`.
- `FRMFW`: the Gillespie first reaction method SSA with `FunctionWrappers`.
- *`NRM`*: The Gibson-Bruck Next Reaction Method. For some reaction network
   structures this may offer better performance than `Direct` (for example,
   large, linear chains of reactions). (Requires dependency graph, see below.) 
- *`RSSA`*: The Rejection SSA (RSSA) method of Thanh et al. With `RSSACR`, for
  very large reaction networks it often offers the best performance of all
  methods. (Requires dependency graph, see below.)
- *`RSSACR`*: The Rejection SSA (RSSA) with Composition-Rejection method of
  Thanh et al. With `RSSA`, for very large reaction networks it often offers the
  best performance of all methods. (Requires dependency graph, see below.)
- *`SortingDirect`*: The Sorting Direct Method of McCollum et al. It will
  usually offer performance as good as `Direct`, and for some systems can offer
  substantially better performance. (Requires dependency graph, see below.)

The design of DiffEqJump.jl composes with DifferentialEquations.jl, allowing for
discrete stochastic chemical reactions to be easily mixed with differential
equation models, allowing for simulation of hybrid systems, jump diffusions,
and differential equations driven by Levy processes.

In addition, DiffEqJump's interfaces allow for solving with regular jump methods, 
such as adaptive Tau-Leaping. 

# Third Party Libraries to Note

## JuMP.jl: Julia for Mathematical Programming

While Optimization.jl is the preferred library for nonlinear optimization, for all
other forms of optimization 
[Julia for Mathematical Programming (JuMP)](https://github.com/jump-dev/JuMP.jl) is
the star. JuMP is the leading choice in Julia for doing:

- Linear Programming
- Quadratic Programming
- Convex Programming
- Conic Programming
- Semidefinite Programming
- Mixed-Complementarity Programming
- Integer Programming
- Mixed Integer (nonlinear/linear) Programming
- (Mixed Integer) Second Order Conic Programming

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

## Manopt.jl: Optimization on Manifolds

[ManOpt.jl](https://github.com/JuliaManifolds/Manopt.jl) allows for easy and efficient
solving of nonlinear optimization problems on manifolds.