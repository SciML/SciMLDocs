# The SciML Open Source Software Ecosystem

The SciML organization is a collection of tools for solving equations and modeling systems developed in the Julia
programming language with bindings to other languages such as R and Python. The organization provides well-maintained 
tools which compose together as a coherent ecosystem. It has a coherent development principle, unified APIs over
large collections of equation solvers, pervasive differentiability and sensitivity analysis, and features many
of the highest performance and parallel implementations one can find.

This documentation is made to pool together the docs of the various SciML libraries
to paint the overarching picture, establish development norms, and document the
shared/common functionality.

## SciML: Combining Scientific Computing and Machine Learning

**SciML is not standard machine learning**, 
[SciML is the combination of scientific computing techniques with machine learning](https://arxiv.org/abs/2001.04385).
Thus the SciML organization is not an organization for machine learning libraries (see 
[FluxML for machine learning in Julia](https://fluxml.ai/)), rather SciML is an organization dedicated to the
development of scientific computing tools which work seamlessly in conjunction with next-generation machine
learning workflows. This includes:

- High performance and accurate tools for standard scientific computing modeling and simulation
- Compatibility with differentiable programming and automatic differentiation
- Tools for building complex multiscale models
- Methods for handling inverse problems, model calibration, controls, and Bayesian analysis
- Symbolic modeling tools for generating efficient code for numerical equation solvers
- Methods for automatic discovery of (bio)physical equations

and much more. For an overview of the broad goals of the SciML organization, watch:

- [The Use and Practice of Scientific Machine Learning](https://www.youtube.com/watch?v=FihLyzdjN_8)
- [State of SciML Scientific Machine Learning](https://www.youtube.com/watch?v=eSeY4K4bITI)

## Overview of Scientific Computing in Julia with SciML

Below is a simplification of the user-facing packages for use in scientific computing and SciML workflows.

|Workflow Element|Associated Julia packages|
| --- | --- |
|Plotting|[Plots](https://docs.juliaplots.org/stable/)|
|Sparse matrix|[SparseArrays](https://docs.julialang.org/en/v1/stdlib/SparseArrays/#Sparse-Arrays)|
|Interpolation/approximation|[DataInterpolations](https://github.com/PumasAI/DataInterpolations.jl), [ApproxFun](https://juliaapproximation.github.io/ApproxFun.jl/stable/)|
|Linear system / least squares|[LinearSolve](http://linearsolve.sciml.ai/dev/)|
|Nonlinear system / rootfinding|[NonlinearSolve](https://nonlinearsolve.sciml.ai/)|
|Polynomial roots|[Polynomials](https://juliamath.github.io/Polynomials.jl/stable/#Root-finding-1)|
|Integration|[Integrals](https://integrals.sciml.ai/)|
|Nonlinear Optimization|[Optimization](https://optimization.sciml.ai/)|
|Other Optimization (linear, quadratic, convex, etc.)|[JuMP](https://github.com/jump-dev/JuMP.jl)|
|[Initial-value problem](https://diffeq.sciml.ai/latest/tutorials/ode_example/#ode_example)|[DifferentialEquations](https://diffeq.sciml.ai/latest/)|
|[Boundary-value problem](https://diffeq.sciml.ai/latest/tutorials/bvp_example/#Boundary-Value-Problems)|[DifferentialEquations](https://diffeq.sciml.ai/latest/)|
|Continuous-Time Markov Chains (Poisson Jumps), Jump Diffusions|[DiffEqJump](https://github.com/SciML/DiffEqJump.jl)|
|Finite differences|[FiniteDifferences](https://juliadiff.org/FiniteDifferences.jl/latest/), [FiniteDiff](https://github.com/JuliaDiff/FiniteDiff.jl)|
|Automatic Differentiation|[ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl), [Enzyme](https://github.com/EnzymeAD/Enzyme.jl), [DiffEqSensitivity](https://sensitivity.sciml.ai/dev/)
|Bayesian Inference|[Turing](https://turing.ml/stable/)|
|Deep Learning|[Flux](https://fluxml.ai/)|
|Acausal Modeling / DAEs|[ModelingToolkit](https://mtk.sciml.ai/dev/)|
|Chemical Reaction Networks|[Catalyst](https://catalyst.sciml.ai/dev/)|
|Symbolic Computing|[Symbolics](https://symbolics.juliasymbolics.org/dev/)|
|Fast Fourier Transform|[FFTW](https://github.com/JuliaMath/FFTW.jl)|

|Partial Differential Equation Discretizations|Associated Julia packages|
|Finite Differences|[MethodOfLines](https://methodoflines.sciml.ai/dev/)|
|Finite Volume|[Trixi](https://github.com/trixi-framework/Trixi.jl)|
|Finite Element|[Gridap](https://github.com/gridap/Gridap.jl)|
|Physics-Informed Neural Networks|[NeuralPDE](https://neuralpde.sciml.ai/dev/)|
|Neural Operators|[NeuralOperators](https://github.com/SciML/NeuralOperators.jl)|

Note that not all of the mentioned packages are SciML packages, but all are heavily tested against as part of SciML
workflows and collaborate with the SciML developers.

## Domains of SciML

The SciML common interface covers the following domains:

- Linear systems (`LinearProblem`)
  - Direct methods for dense and sparse
  - Iterative solvers with preconditioning
- Nonlinear Systems (`NonlinearProblem`)
  - Systems of nonlinear equations
  - Scalar bracketing systems
- Integrals (quadrature) (`QuadratureProblem`)
- Differential Equations
  - Discrete equations (function maps, discrete stochastic (Gillespie/Markov)
    simulations) (`DiscreteProblem` and `JumpProblem`)
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
- Optimization (`OptimizationProblem`)
  - Nonlinear (constrained) optimization
- (Stochastic/Delay/Differential-Algebraic) Partial Differential Equations (`PDESystem`)
  - Finite difference and finite volume methods
  - Interfaces to finite element methods
  - Physics-Informed Neural Networks (PINNs)
  - Integro-Differential Equations
  - Fractional Differential Equations
- Data-driven modeling
  - Discrete-time data-driven dynamical systems (`DiscreteDataDrivenProblem`)
  - Continuous-time data-driven dynamical systems (`ContinuousDataDrivenProblem`)
  - Symbolic regression (`DirectDataDrivenProblem`)
- Uncertainty quantification and expected values (`ExpectationProblem`)

The SciML common interface also includes
[ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
for defining such systems symbolically, allowing for optimizations like automated
generation of parallel code, symbolic simplification, and generation of sparsity
patterns.

## Inverse Problems, Parameter Estimation, and Structural Identification

We note that parameter estimation and inverse problems are solved directly on their
constituent problem types using tools like [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl).
Thus for example, there is no `ODEInverseProblem`, and instead `ODEProblem` is used to
find the parameters `p` that solve the inverse problem.

## Common Interface High Level

The SciML interface is common as the usage of arguments is standardized across
all of the problem domains. Underlying high level ideas include:

- All domains use the same interface of defining a `SciMLProblem` which is then
  solved via `solve(prob,alg;kwargs)`, where `alg` is a `SciMLAlgorithm`. The
  keyword argument namings are standardized across the organization.
- `SciMLProblem`s are generally defined by a `SciMLFunction` which can define
  extra details about a model function, such as its analytical Jacobian, its
  sparsity patterns and so on.
- There is an organization-wide method for defining linear and nonlinear solvers
  used within other solvers, giving maximum control of performance to the user.
- Types used within the packages are defined by the input types. For example,
  packages attempt to internally use the type of the initial condition as the
  type for the state within differential equation solvers.
- `solve` calls should be thread-safe and parallel-safe.
- `init(prob,alg;kwargs)` returns an iterator which allows for directly iterating
  over the solution process
- High performance is key. Any performance that is not at the top level is considered
  a bug and should be reported as such.
- All functions have an in-place and out-of-place form, where the in-place form
  is made to utilize mutation for high performance on large-scale problems and
  the out-of-place form is for compatibility with tooling like static arrays and
  some reverse-mode automatic differentiation systems.

## Flowchart Example for PDE-Constrained Optimal Control

The following example showcases how the pieces of the common interface connect to solve a problem
that mixes inference, symbolics, and numerics.

![](https://user-images.githubusercontent.com/1814174/126318252-1e4152df-e6e2-42a3-8669-f8608f81a095.png)

## External Binding Libraries

- [diffeqr](https://github.com/SciML/diffeqr)
    - Solving differential equations in R using DifferentialEquations.jl with ModelingToolkit for JIT compilation and GPU-acceleration
- [diffeqpy](https://github.com/SciML/diffeqpy)
    - Solving differential equations in Python using DifferentialEquations.jl

## Note About Third Party Libraries

The SciML documentation references and recommends many third party libraries for improving ones
modeling, simulation, and analysis workflow in Julia. Take these as a positive affirmation of the
quality of these libraries, as these libraries are commonly tested against by SciML developers and
are in contact with the development teams of these groups. It also documents the libraries which
are commonly chosen by SciML as dependencies.
**Do not take omissions as negative affirmations against a given library**, i.e. a library left off
of the list by SciML is not a negative endorsement. Rather, it means that compatibility with SciML
is untested, SciML developers may have a personal preference for another choice, or SciML developers
may be simply unaware of the library's existence. If one would like to add a third party library
to the SciML documentation, open a pull request with the requested text. 

Note that the libraries in this documentation are only those that are meant to be used in the SciML
extended universe of modeling, simulation, and analysis and thus there are many high quality libraries
in other domains (machine learning, data science, etc.) which are purposefully not included. For
an overview of the Julia package ecosystem, see [the JuliaHub Search Engine](https://juliahub.com/ui/Home).

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- There are a few community forums:
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Slack](https://julialang.org/slack/)
    - [JuliaDiffEq](https://gitter.im/JuliaDiffEq/Lobby) on Gitter
    - On the [Julia Discourse forums](https://discourse.julialang.org)
    - See also [SciML Community page](https://sciml.ai/community/)
