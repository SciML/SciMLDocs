# [Getting Started with  Julia's SciML for the MATLAB User](@id matlab)

If you're a MATLAB user who has looked into Julia for some performance improvements, you
may have noticed that the standard library does not have all of the "batteries" included
with a base MATLAB installation. Where's the ODE solver? Where's `fmincon` and `fsolve`?
Those scientific computing functionalities are the pieces provided by the Julia SciML
ecosystem!

## Why SciML? High-Level Workflow Reasons

* **Performance** - The key reason people are moving from MATLAB to Julia's SciML in droves
  is performance. Even [simple ODE solvers are much faster!](https://benchmarks.sciml.ai/stable/MultiLanguage/ode_wrapper_packages/),
  demonstrating orders of magnitude performance improvements for differential equations,
  nonlinear solving, optimization, and more. And the performance advantages continue to
  grow as more complex algorithms are required.
* **Julia is quick to learn from MATLAB** - Most ODE codes can be translated in a few
  minutes. If you need help, check out the
  [QuantEcon MATLAB-Python-Julia Cheatsheet](https://cheatsheets.quantecon.org/)
* **Package Management and Versioning** - [Julia's package manager](https://github.com/JuliaLang/Pkg.jl)
  takes care of dependency management, testing, and continuous delivery in order to make
  the installation and maintanance process smoother. For package users, this means it's
  easier to get packages with complex functionality in your hands.
* **Free and Open Source** - If you want to know how things are being computed, just look
  [at our Github organization](https://github.com/SciML). Lots of individuals use Julia's
  SciML to research how the algorithms actually work because of how accessible and tweakable
  the ecosystem is!
* **Composable Library Components** - In MATLAB environments, every package feels like
  a silo. Functions made for one file exchange library cannot easily compose with another.
  SciML's generic coding with JIT compilation these connections create new optimized code on
  the fly and allow for a more expansive feature set than can ever be documented. Take
  [new high-precision number types from a package](https://github.com/JuliaArbTypes/ArbFloats.jl)
  and stick them into a nonlinear solver. Take
  [a package for Intel GPU arrays](https://github.com/JuliaGPU/oneAPI.jl) and stick it into
  the differential equation solver to use specialized hardware acceleration.
* **Easier High-Performance and Parallel Computing** - With Julia's ecosystem,
  [CUDA](https://github.com/JuliaGPU/CUDA.jl) will automatically install of the required
  binaries and `cu(A)*cu(B)` is then all that's required to GPU-accelerate large-scale
  linear algebra. [MPI](https://github.com/JuliaParallel/MPI.jl) is easy to install and
  use. [Distributed computing through password-less SSH](https://docs.julialang.org/en/v1/manual/distributed-computing/). [Multithreading](https://docs.julialang.org/en/v1/manual/multi-threading/)
  is automatic and baked into a lot of libraries, with a specialized algorithm to ensure
  hierarchical usage does not oversubscribe threads. Basically, libraries give you a lot
  of parallelism for free, and doing the rest is a piece of cake.
* **Mix Scientific Computing with Machine Learning** - Want to [automate the discovery
  of missing physical laws using neural networks embedded in differentiable simulations](https://arxiv.org/abs/2001.04385)? Julia's SciML is the ecosystem with the tooling to integrate machine
  learning into the traditional high-performance scientific computing domains, from
  multiphysics simulations to partial differential equations.

In this plot, `MATLAB` in orange represents MATLAB's most commonly used solvers:

![](https://user-images.githubusercontent.com/1814174/195836404-ea69730e-69a4-4bf0-8d12-f57d5b8fce21.PNG)

## Need a case study?

Check out [this talk from NASA Scientists getting a 15,000x acceleration by switching from
Simulink to Julia's ModelingToolkit!](https://www.youtube.com/watch?v=tQpqsmwlfY0).

## Need Help Translating from MATLAB to Julia?

The following resources can be particularly helpful when adopting Julia for SciML for the
first time:

* [QuantEcon MATLAB-Python-Julia Cheatsheet](https://cheatsheets.quantecon.org/)
* [The Julia Manual's Noteworthy Differences from MATLAB page](https://docs.julialang.org/en/v1/manual/noteworthy-differences/#Noteworthy-differences-from-MATLAB)
* Double check your results with [MATLABDiffEq.jl](https://github.com/SciML/MATLABDiffEq.jl)
  (automatically converts and runs ODE definitions with MATLAB's solvers)
* Use [MATLAB.jl](https://github.com/JuliaInterop/MATLAB.jl) to more incrementally move
  code to Julia.

## MATLAB to Julia SciML Functionality Translations

The following chart will help you get quickly acquainted with Julia's SciML Tools:

|MATLAB Function|SciML-Supported Julia packages|
| --- | --- |
|`plot`|[Plots](https://docs.juliaplots.org/stable/), [Makie](https://docs.makie.org/stable/)|
|`sparse`|[SparseArrays](https://docs.julialang.org/en/v1/stdlib/SparseArrays/#Sparse-Arrays)|
|`interp1`|[DataInterpolations](https://github.com/PumasAI/DataInterpolations.jl)|
|`\`, `gmres`, `cg`|[LinearSolve](http://linearsolve.sciml.ai/dev/)|
|`fsolve`|[NonlinearSolve](https://nonlinearsolve.sciml.ai/)|
|`quad`|[Integrals](https://integrals.sciml.ai/)|
|`fmincon`|[Optimization](https://optimization.sciml.ai/)|
|`odeXX`|[DifferentialEquations](https://diffeq.sciml.ai/latest/)|
|`ode45`|`Tsit5`|
|`ode113`|`VCABM`|
|`ode23s`|`Rosenbrock23`|
|`ode15s`|`QNDF` or `FBDF`|
|`ode15i`|`IDA`|
|`bvp4c` and `bvp5c`|[DifferentialEquations](https://diffeq.sciml.ai/latest/)|
|Simulink, Simscape|[ModelingToolkit](https://mtk.sciml.ai/dev/)|
|`fft`|[FFTW](https://github.com/JuliaMath/FFTW.jl)|
|chebfun|[ApproxFun](https://juliaapproximation.github.io/ApproxFun.jl/stable/)|
