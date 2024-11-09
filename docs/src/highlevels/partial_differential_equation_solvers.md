# Partial Differential Equations (PDE)

## NeuralPDE.jl: Physics-Informed Neural Network (PINN) PDE Solvers

[NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) is a partial differential equation
solver library which uses physics-informed neural networks (PINNs) to solve the equations.
It uses the ModelingToolkit.jl symbolic `PDESystem` as its input and can handle a wide
variety of equation types, including systems of partial differential equations,
partial differential-algebraic equations, and integro-differential equations. Its benefit
is its flexibility, and it can be used to easily generate surrogate solutions over entire
parameter ranges. However, its downside is solver speed: PINN solvers tend to be a lot
slower than other methods for solving PDEs.

## MethodOflines.jl: Automated Finite Difference Method (FDM)

[MethodOflines.jl](https://github.com/SciML/MethodOfLines.jl) is a partial differential
equation solver library which automates the discretization of PDEs via the finite
difference method. It uses the ModelingToolkit.jl symbolic `PDESystem` as its input,
and generates `AbstractSystem`s and `SciMLProblem`s whose numerical solution gives
the solution to the PDE.

## FEniCS.jl: Wrappers for the Finite Element Method (FEM)

[FEniCS.jl](https://github.com/SciML/FEniCS.jl) is a wrapper for the popular FEniCS
finite element method library.

## HighDimPDE.jl:  High-dimensional PDE Solvers

[HighDimPDE.jl](https://github.com/SciML/HighDimPDE.jl) is a partial differential equation
solver library which implements algorithms that break down the curse of dimensionality
to solve the equations. It implements deep-learning based and Picard-iteration based methods
to approximately solve high-dimensional, nonlinear, non-local PDEs in up to 10,000 dimensions.
Its cons are accuracy: high-dimensional solvers are stochastic, and might result in wrong solutions
if the solver meta-parameters are not appropriate.

## NeuralOperators.jl: (Fourier) Neural Operators and DeepONets for PDE Solving

[NeuralOperators.jl](https://github.com/SciML/NeuralOperators.jl) is a library for
operator learning based PDE solvers. This includes techniques like:

  - Fourier Neural Operators (FNO)
  - Deep Operator Networks (DeepONets)
  - Markov Neural Operators (MNO)

Currently, its connection to PDE solving must be specified manually, though an interface
for ModelingToolkit `PDESystem`s is in progress.

## DiffEqOperators.jl: Operators for Finite Difference Method (FDM) Discretizations

[DiffEqOperators.jl](https://github.com/SciML/DiffEqOperators.jl) is a library for
defining finite difference operators to easily perform manual FDM semi-discretizations
of partial differential equations. This library is fairly incomplete and most cases
should receive better performance using MethodOflines.jl.

# Third-Party Libraries to Note

A more exhaustive list of Julia PDE packages can be found here: https://github.com/JuliaPDE/SurveyofPDEPackages

## ApproxFun.jl: Automated Spectral Discretizations

[ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl) is a package for
approximating functions in basis sets. One particular use case is with spectral
basis sets, such as Chebyshev functions and Fourier decompositions, making it easy
to represent spectral and pseudospectral discretizations of partial differential equations
as ordinary differential equations for the SciML equation solvers.
## Decapodes.jl: Discrete Exterior Calculus Applied to Partial and Ordinary Differential Equation Systems

[Decapodes.jl](https://github.com/AlgebraicJulia/Decapodes.jl) is a computational physics framework based
on the Discrete Exterior Calculus (DEC). It uses structure preserving discretization methods from the DEC
to simulate multiphysics problems. Feature include solving PDEs on triangulated manifolds, a strongly
typed equation representation that can help ensure correctness of simulations, and compositional methods
for specifying multi-physics problems.

## Ferrite.jl: Finite Element Toolbox for Julia

[Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) is a performant and extensible
library which provides algorithms and data structures to develop finite element software.
This library aims at users which need fine grained control over all algorithmic details,
as for example often necessary in research when developing new grid-based PDE discretizations
or other more advanced problem formulations for example found in continuum mechanics.

## Gridap.jl: Julia-Based Tools for Finite Element Discretizations

[Gridap.jl](https://github.com/gridap/Gridap.jl) is a package for grid-based approximation
of partial differential equations, particularly notable for its use of conforming and
nonconforming finite element (FEM) discretizations.

## Trixi.jl: Adaptive High-Order Numerical Simulations of Hyperbolic Equations

[Trixi.jl](https://github.com/trixi-framework/Trixi.jl) is a package for numerical simulation
of hyperbolic conservation laws, i.e. a large set of hyperbolic partial differential equations,
which interfaces and uses the SciML ordinary differential equation solvers.

## VoronoiFVM.jl: Tools for the Voronoi Finite Volume Discretizations

[VoronoiFVM.jl](https://github.com/j-fu/VoronoiFVM.jl) is a library for generating FVM discretizations
of systems of PDEs. It interfaces with many of the SciML equation solver libraries to allow
for ease of discretization and flexibility in the solver choice.
