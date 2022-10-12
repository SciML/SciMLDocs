# SciML Numerical Utility Libraries

## ExponentialUtilities.jl: Faster Matrix Exponentials

[ExponentialUtilities.jl](https://github.com/SciML/ExponentialUtilities.jl) is a library for efficient computation
of matrix exponentials. While Julia has a built-in `exp(A)` method, ExponentialUtilities.jl offers many features
around this to improve performance in scientific contexts, including:

- Faster methods for (non-allocating) matrix exponentials via `exponential!`
- Methods for computing matrix exponential that are generic to number types and arrays (i.e. GPUs)
- Methods for computing arnoldi iterations on Krylov subspaces
- Direct computation of `exp(t*A)*v`, i.e. exponentiation of a matrix times a vector, without computing the matrix exponential
- Direct computation of `ϕ_m(t*A)*v` operations, where `ϕ_0(z) = exp(z)` and `ϕ_(k+1)(z) = (ϕ_k(z) - 1) / z`

ExponentialUtilities.jl includes complex adaptive time stepping techniques such as KIOPS in order to perform these
calculations in a fast and numerically-stable way.

## QuasiMonteCarlo.jl: Fast Quasi-Random Number Generation

[QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl) is a library for fast generation of
ow discrepency Quasi-Monte Carlo samples, using methods like:

* `GridSample(dx)` where the grid is given by `lb:dx[i]:ub` in the ith direction.
* `UniformSample` for uniformly distributed random numbers.
* `SobolSample` for the Sobol sequence.
* `LatinHypercubeSample` for a Latin Hypercube.
* `LatticeRuleSample` for a randomly-shifted rank-1 lattice rule.
* `LowDiscrepancySample(base)` where `base[i]` is the base in the ith direction.
* `GoldenSample` for a Golden Ratio sequence.
* `KroneckerSample(alpha, s0)` for a Kronecker sequence, where alpha is an length-d vector of irrational numbers (often sqrt(d)) and s0 is a length-d seed vector (often 0).
* `SectionSample(x0, sampler)` where `sampler` is any sampler above and `x0` is a vector of either `NaN` for a free dimension or some scalar for a constrained dimension.

## PoissonRandom.jl: Fast Poisson Random Number Generation

[PoissonRandom.jl](https://github.com/SciML/PoissonRandom.jl) is just fast Poisson random number generation
for Poisson processes, like chemical master equations.

## PreallocationTools.jl: Write Non-Allocating Code Easier

[PreallocationTools.jl](https://github.com/SciML/PreallocationTools.jl) is a library of tools for writing
non-allocating code that interacts well with advanced features like automatic differentiation and symbolics.

## RuntimeGeneratedFunctions.jl: Efficient Staged Programming in Julia

[RuntimeGeneratedFunctions.jl](https://github.com/SciML/RuntimeGeneratedFunctions.jl) allows for staged
programming in Julia, compiling functions at runtime with full optimizations. This is used by many libraries
such as ModelingToolkit.jl to allow for runtime code generation for improved performance.

## EllipsisNotation.jl: Implementation of Ellipsis Array Slicing

[EllipsisNotation.jl](https://github.com/ChrisRackauckas/EllipsisNotation.jl) defines the ellipsis
array slicing notation for Julia. It uses `..` as a catch all for "all dimensions", allow for indexing
like `[..,1]` to mean "[:,:,:,1]` on four dimensional arrays, in a way that is generic to the number
of dimensions in the underlying array.

# Third Party Libraries to Note

## Distributions.jl: Representations of Probability Distributions

[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) is a library for defining distributions
in Julia. It's used all throughout the SciML libraries for specifications of probability distributions.

!!! note

    For full compatibility with automatic differentiation, see
    [DistributionsAD.jl](https://github.com/TuringLang/DistributionsAD.jl)

## FFTW.jl: Fastest Fourier Transformation in the West

[FFTW.jl](https://github.com/JuliaMath/FFTW.jl) is the preferred library for fast Fourier
Transformations on the CPU.

## SpecialFunctions.jl: Implementations of Mathematical Special Functions

[SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl) is a library of
implementations of special functions, like Bessel functions and error functions (`erf`).
This library is compatible with automatic differentiation.

## LoopVectorization.jl: Automated Loop Acceleator

[LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) is a library which
provides the `@turbo` and `@tturbo` macros for accelerating the computation of loops. This
can be used to accelerating the model functions sent to the equation solvers, for example,
accelerating handwritten PDE discretizations.

## Polyester.jl: Cheap Threads

[Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) is a cheaper version of threads for
Julia which use a set pool of threads for lower overhead. Note that Polyester does not
compose with the standard Julia composable theading infrastructure, and thus one must take
care to not compose two levels of Polyester as this will oversubscribe the computation and
lead to performance degredation. Many SciML solvers have options to use Polyseter for
threading to achieve the top performance.

## Tullio.jl: Fast Tensor Calculations and Einstein Notation

[Tullio.jl](https://github.com/mcabbott/Tullio.jl) is a library for fast tensor calculations
with Einstein notation. It allows for defining operations which are compatible with
automatic differentiation, GPUs, and more.

## ParallelStencil.jl: High-Level Code for Parallelized Stencil Computations

[ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) is a library for writing
high level code forparallelized stencil computations. It is
[compatible with SciML equation solvers](https://github.com/omlins/ParallelStencil.jl/issues/29)
and is thus a good way to generate GPU and distributed parallel model code.

## DataInterpolations.jl: One-Dimensional Interpolations

[DataInterpolations.jl](https://github.com/PumasAI/DataInterpolations.jl) is a library of one-dimensional interpolation
schemes which are composable with automatic differentiation and the SciML ecosystem. It includes direct interpolation
methods and regression techniques for handling noisy data. Its methods include:

- `ConstantInterpolation(u,t)` - A piecewise constant interpolation.
- `LinearInterpolation(u,t)` - A linear interpolation.
- `QuadraticInterpolation(u,t)` - A quadratic interpolation.
- `LagrangeInterpolation(u,t,n)` - A Lagrange interpolation of order `n`.
- `QuadraticSpline(u,t)` - A quadratic spline interpolation.
- `CubicSpline(u,t)` - A cubic spline interpolation.
- `BSplineInterpolation(u,t,d,pVec,knotVec)` - An interpolation B-spline.
  This is a B-spline which hits each of the data points. The argument choices are:
  	- `d` - degree of B-spline
  	- `pVec` - Symbol to Parameters Vector, `pVec = :Uniform` for uniform spaced parameters and
      `pVec = :ArcLen` for parameters generated by chord length method.
  	- `knotVec` - Symbol to Knot Vector, `knotVec = :Uniform` for uniform knot vector,
      `knotVec = :Average` for average spaced knot vector.
- `BSplineApprox(u,t,d,h,pVec,knotVec)` - A regression B-spline which smooths the fitting curve.
  The argument choices are the same as the `BSplineInterpolation`, with the additional parameter
  `h<length(t)` which is the number of control points to use, with smaller `h` indicating more smoothing.
- `Curvefit(u,t,m,p,alg)` - An interpolation which is done by fitting a user-given functional form
  `m(t,p)` where `p` is the vector of parameters. The user's input `p` is a an initial value for a
  least-square fitting, `alg` is the algorithm choice to use for optimize the cost function (sum of
  squared deviations) via `Optim.jl` and optimal `p`s are used in the interpolation.

These interpolations match the SciML interfaces and have direct support for packages like ModelingToolkit.jl.

# Julia Utilities

## StaticCompiler.jl

[StaticCompiler.jl](https://github.com/tshort/StaticCompiler.jl) is a package for generating static binaries
from Julia code. It only supports a subset of Julia, so not all equation solver algorithms are compatible
with StaticCompiler.jl.

## PackageCompiler.jl

[PackageCompiler.jl](https://github.com/JuliaLang/PackageCompiler.jl) is a package for generating shared
libraries from Julia code. It the entirety of Julia by bundling a system image with the Julia runtime,
thus it builds complete binaries that can hold all of the functionality of SciML. It can also be used
to generate new system images to decrease startup times and remove JIT-compilation from SciML usage.
