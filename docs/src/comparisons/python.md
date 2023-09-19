# [Getting Started with Julia's SciML for the Python User](@id python)

If you're a Python user who has looked into Julia, you're probably wondering what is the
equivalent to SciPy is. And you found it: it's the SciML ecosystem! To a Python developer,
SciML is SciPy, but with the high-performance GPU, capabilities of PyTorch, and
neural network capabilities, all baked right in. With SciML, there is no “separate world”
of machine learning sublanguages: there is just one cohesive package ecosystem.

## Why SciML? High-Level Workflow Reasons

  - **Performance** - The key reason people are moving from SciPy to Julia's SciML in droves
    is performance. Even [simple ODE solvers are much faster!](https://benchmarks.sciml.ai/stable/MultiLanguage/ode_wrapper_packages/),
    demonstrating orders of magnitude performance improvements for differential equations,
    nonlinear solving, optimization, and more. And the performance advantages continue to
    grow as more complex algorithms are required.
  - **Package Management and Versioning** - [Julia's package manager](https://github.com/JuliaLang/Pkg.jl)
    takes care of dependency management, testing, and continuous delivery in order to make
    the installation and maintenance process smoother. For package users, this means it's
    easier to get packages with complex functionality in your hands.
  - **Composable Library Components** - In Python environments, every package feels like
    a silo. Functions made for one file exchange library cannot easily compose with another.
    SciML's generic coding with JIT compilation these connections create new optimized code on
    the fly and allow for a more expansive feature set than can ever be documented. Take
    [new high-precision number types from a package](https://github.com/JuliaArbTypes/ArbFloats.jl)
    and stick them into a nonlinear solver. Take
    [a package for Intel GPU arrays](https://github.com/JuliaGPU/oneAPI.jl) and stick it into
    the differential equation solver to use specialized hardware acceleration.
  - **Easier High-Performance and Parallel Computing** - With Julia's ecosystem,
    [CUDA](https://github.com/JuliaGPU/CUDA.jl) will automatically install of the required
    binaries and `cu(A)*cu(B)` is then all that's required to GPU-accelerate large-scale
    linear algebra. [MPI](https://github.com/JuliaParallel/MPI.jl) is easy to install and
    use. [Distributed computing through password-less SSH](https://docs.julialang.org/en/v1/manual/distributed-computing/). [Multithreading](https://docs.julialang.org/en/v1/manual/multi-threading/)
    is automatic and baked into many libraries, with a specialized algorithm to ensure
    hierarchical usage does not oversubscribe threads. Basically, libraries give you a lot
    of parallelism for free, and doing the rest is a piece of cake.
  - **Mix Scientific Computing with Machine Learning** - Want to [automate the discovery
    of missing physical laws using neural networks embedded in differentiable simulations](https://arxiv.org/abs/2001.04385)? Julia's SciML is the ecosystem with the tooling to integrate machine
    learning into the traditional high-performance scientific computing domains, from
    multiphysics simulations to partial differential equations.

In this plot, `SciPy` in yellow represents Python's most commonly used solvers:

![](https://user-images.githubusercontent.com/1814174/195836404-ea69730e-69a4-4bf0-8d12-f57d5b8fce21.PNG)

## Need Help Translating from Python to Julia?

The following resources can be particularly helpful when adopting Julia for SciML for the
first time:

  - [The Julia Manual's Noteworthy Differences from Python page](https://docs.julialang.org/en/v1/manual/noteworthy-differences/#Noteworthy-differences-from-Python)
  - Double-check your results with [SciPyDiffEq.jl](https://github.com/SciML/SciPyDiffEq.jl)
    (automatically converts and runs ODE definitions with SciPy's solvers)
  - Use [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) to more incrementally move
    code to Julia.

## Python to Julia SciML Functionality Translations

The following chart will help you get quickly acquainted with Julia's SciML Tools:

| Workflow Element             | SciML-Supported Julia packages                                                                                                                                          |
|:---------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Matplotlib                   | [Plots](https://docs.juliaplots.org/stable/), [Makie](https://docs.makie.org/stable/)                                                                                   |
| `scipy.special`              | [SpecialFunctions](https://github.com/JuliaMath/SpecialFunctions.jl)                                                                                                    |
| `scipy.linalg.solve`         | [LinearSolve](http://linearsolve.sciml.ai/dev/)                                                                                                                         |
| `scipy.integrate`            | [Integrals](https://integrals.sciml.ai/)                                                                                                                                |
| `scipy.optimize`             | [Optimization](https://optimization.sciml.ai/)                                                                                                                          |
| `scipy.optimize.fsolve`      | [NonlinearSolve](https://nonlinearsolve.sciml.ai/)                                                                                                                      |
| `scipy.interpolate`          | [DataInterpolations](https://docs.sciml.ai/DataInterpolations/)                                                                                                  |
| `scipy.fft`                  | [FFTW](https://github.com/JuliaMath/FFTW.jl)                                                                                                                            |
| `scipy.linalg`               | [Julia's Built-In Linear Algebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)                                                                               |
| `scipy.sparse`               | [SparseArrays](https://docs.julialang.org/en/v1/stdlib/SparseArrays/#Sparse-Arrays), [ARPACK](https://github.com/JuliaLinearAlgebra/Arpack.jl)                          |
| `odeint`/`solve_ivp`         | [DifferentialEquations](https://diffeq.sciml.ai/latest/)                                                                                                                |
| `scipy.integrate.solve_bvp`  | [Boundary-value problem](https://diffeq.sciml.ai/latest/tutorials/bvp_example/#Boundary-Value-Problems)                                                                 |
| PyTorch                      | [Flux](https://fluxml.ai/), [Lux](http://lux.csail.mit.edu/stable/)                                                                                                     |
| gillespy2                    | [Catalyst](https://catalyst.sciml.ai/dev/), [JumpProcesses](https://github.com/SciML/JumpProcesses.jl)                                                                  |
| scipy.optimize.approx_fprime | [FiniteDiff](https://github.com/JuliaDiff/FiniteDiff.jl)                                                                                                                |
| autograd                     | [ForwardDiff\*](https://github.com/JuliaDiff/ForwardDiff.jl), [Enzyme\*](https://github.com/EnzymeAD/Enzyme.jl), [DiffEqSensitivity](https://sensitivity.sciml.ai/dev/) |
| Stan                         | [Turing](https://turing.ml/stable/)                                                                                                                                     |
| sympy                        | [Symbolics](https://symbolics.juliasymbolics.org/dev/)                                                                                                                  |

## Why is Differentiable Programming Important for Scientific Computing?

Check out [this blog post that goes into detail on how training neural networks in tandem
with simulation improves performance by orders of magnitude](https://www.stochasticlifestyle.com/is-differentiable-programming-actually-necessary-cant-you-just-train-separately/). But can't
you use analytical adjoint definitions? You can, but there are tricks to mix automatic
differentiation into the adjoint definitions for a few orders of magnitude improvement too,
as [explained in this blog post](https://www.stochasticlifestyle.com/direct-automatic-differentiation-of-solvers-vs-analytical-adjoints-which-is-better/).

These facts, along with many others, compose to algorithmic improvements with the
implementation improvements, which leads to orders of magnitude improvements!
