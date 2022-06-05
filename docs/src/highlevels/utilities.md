# SciML Utility Libraries

## Surrogates.jl: Easy Generation of Differentiable Surrogate Models

[Surrogates.jl](https://github.com/SciML/Surrogates.jl) is a library for generating surrogate approximations
to computationally expensive simulations. It has the following high-dimensional function approximators:

- Kriging
- Kriging using Stheno
- Radial Basis
- Wendland
- Linear
- Second Order Polynomial
- Support Vector Machines (Wait for LIBSVM resolution)
- Neural Networks
- Random Forests
- Lobachevsky splines
- Inverse-distance
- Polynomial expansions
- Variable fidelity
- Mixture of experts (Waiting GaussianMixtures package to work on v1.5)
- Earth
- Gradient Enhanced Kriging

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

# Third Party Libraries to Note

## DynamicalSystems.jl: A Suite of Dynamical Systems Analysis

[DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/latest/) is an entire ecosystem
of dynamical systems analysis methods, for computing measures of chaos (dimension estimation, lyopunov coefficients),
generating delay embeddings, and much more. It uses the SciML tools for its internal equation solving
and thus shares much of its API, adding a layer of new tools for extended analyses.

For more information, watch the [tutorial Introduction to DynamicalSystems.jl](https://www.youtube.com/watch?v=A8g9rdEfdNg).

## Distributions.jl: Representations of Probability Distributions

[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) is a library for defining distributions
in Julia. It's used all throughout the SciML libraries for specifications of probability distributions.

## FFTW.jl: Fastest Fourier Transformation in the West

[FFTW.jl](https://github.com/JuliaMath/FFTW.jl) is the preferred library for fast Fourier Transformations
on the CPU.

## SpecialFunctions.jl: Implementations of Mathematical Special Functions

[SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl) is a library of implementations of
special functions, like Bessel functions and error functions (`erf`). This library is compatible with
automatic differentiation.

## LoopVectorization.jl: Automated Loop Acceleator

[LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) is a library which provides the `@turbo`
and `@tturbo` macros for accelerating the computation of loops. This can be used to accelerating the model
functions sent to the equation solvers, for example, accelerating handwritten PDE discretizations.

## Polyester.jl: Cheap Threads

[Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) is a cheaper version of threads for Julia which use
a set pool of threads for lower overhead. Note that Polyester does not compose with the standard Julia composable
theading infrastructure, and thus one must take care to not compose two levels of Polyester as this will oversubscribe
the computation and lead to performance degredation. Many SciML solvers have options to use Polyseter for threading
to achieve the top performance.