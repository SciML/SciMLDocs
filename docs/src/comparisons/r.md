# [Getting Started with Julia's SciML for the R User](@id r)

If you're an R user who has looked into Julia, you're probably wondering where all of the
scientific computing packages are. How do I solve ODEs? Solve f(x)=0 for x? Etc. SciML
is the ecosystem for doing this with Julia.

## Why SciML? High-Level Workflow Reasons

* **Performance** - The key reason people are moving from R to Julia's SciML in droves
  is performance. Even [simple ODE solvers are much faster!](https://benchmarks.sciml.ai/stable/MultiLanguage/ode_wrapper_packages/),
  demonstrating orders of magnitude performance improvements for differential equations,
  nonlinear solving, optimization, and more. And the performance advantages continue to
  grow as more complex algorithms are required.
* **Composable Library Components** - In R environments, every package feels like
  a silo. Functions made for one file exchange library cannot easily compose with another.
  SciML's generic coding with JIT compilation these connections create new optimized code on
  the fly and allow for a more expansive feature set than can ever be documented. Take
  [new high-precision number types from a package](https://github.com/JuliaArbTypes/ArbFloats.jl)
  and stick them into a nonlinear solver. Take
  [a package for Intel GPU arrays](https://github.com/JuliaGPU/oneAPI.jl) and stick it into
  the differential equation solver to use specialized hardware acceleration.
* **A Global Harmonious Documentation for Scientific Computing** - R's documenation for
  scientific computing is scattered in a bunch of individual packages where the developers
  do not talk to each other! This not only leads to documentation differences but also
  "style" differences: one package uses `tol` while the other uses `atol`. With Julia's
  SciML, the whole ecosystem is considered together, and inconsitencies are handled at the
  global level. The goal is to be working in one environment with one language.
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

In this plot, `deSolve` in blue represents R's most commonly used solver:

![](https://user-images.githubusercontent.com/1814174/195836404-ea69730e-69a4-4bf0-8d12-f57d5b8fce21.PNG)

## Need Help Translating from R to Julia?

The following resources can be particularly helpful when adopting Julia for SciML for the
first time:

* [The Julia Manual's Noteworthy Differences from R page](https://docs.julialang.org/en/v1/manual/noteworthy-differences/#Noteworthy-differences-from-R)
* [Tutorials on Data Wrangling and Plotting in Julia (Sections 4 and 5)](http://tutorials.pumas.ai/)
  written for folks with a background in R.
* Double check your results with [deSolveDiffEq.jl](https://github.com/SciML/deSolveDiffEq.jl)
  (automatically converts and runs ODE definitions with R's deSolve solvers)
* Use [RCall.jl](https://juliainterop.github.io/RCall.jl/stable/) to more incrementally move
  code to Julia.
* [Comparisons between R and Julia from the DataFrames package](https://dataframes.juliadata.org/stable/man/comparisons/). And an [accessible starting point for Julia's DataFrames](https://bkamins.github.io/julialang/2020/12/24/minilanguage.html).

## R to Julia SciML Functionality Translations

The following chart will help you get quickly acquainted with Julia's SciML Tools:

|R Function/Package|SciML-Supported Julia packages|
| --- | --- |
|`data.frame`|[DataFrames](https://dataframes.juliadata.org/stable/)|
|`plot`|[Plots](https://docs.juliaplots.org/stable/), [Makie](https://docs.makie.org/stable/)|
|`ggplot2`|[AlgebraOfGraphics](https://aog.makie.org/stable/)|
|`deSolve`|[DifferentialEquations](https://diffeq.sciml.ai/latest/)|
|Stan|[Turing](https://turing.ml/stable/)|

## Want to See the Power of Julia?

Check out [this R-Bloggers blog post on diffeqr](https://www.r-bloggers.com/2020/08/gpu-accelerated-ode-solving-in-r-with-julia-the-language-of-libraries/), a package which
uses [ModelingToolkit](https://mtk.sciml.ai/dev/) to translate R code to Julia, and achieves
**350x acceleation over R's popular deSolve** ODE solver package. But when the solve is
done purely in Julia, it achieves **2777x acceleration over deSolve**!
