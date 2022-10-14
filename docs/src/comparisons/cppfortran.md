# [Introduction to Julia's SciML for the C++/Fortran User](@id cppfortran)

You don't need help if you're a Fortran guru. I'm just kidding, you're not a Lisp developer.
If you're coming from C++ or Fortran, you may be familiar with similar high-performance
computing environments to SciML are [PETSc](https://petsc.org/release/),
[Trilinos](https://trilinos.github.io/), or
[Sundials](https://computing.llnl.gov/projects/sundials). The following are some points
to help the transition.

## Why SciML? High-Level Workflow Reasons

If you're coming from "hardcore" C++/Fortran computing environments, some things to check
out with Julia's SciML are:

* **Interactivity** - use the interactive REPL to easily investigate numerical details.
* **Metaprogramming performance tools** - tools like
  [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) can be used
  to generate faster code than even some of the most hand-optimized C++/Fortran code.
  Current benchmarks [show this SIMD-optimized Julia code outperforming OpenBLAS and MKL
  BLAS implementations in many performance regimes](https://www.youtube.com/watch?v=KQ8nvlURX4M).
* **Symbolic modeling languages** - writing models by hand can leave a lot of performance
  on the table. Using high-level modeling tools like
  [ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl) can automate symbolic
  simplifications which
  [improve the stability and performance of numerical solvers](https://www.youtube.com/watch?v=ZFoQihr3xLs).
  On complex models, even the best handwritten C++/Fortran code is orders of mangitude behind
  the code that symbolic tearing algorithms can achieve!
* **Composable Library Components** - In C++/Fortran environments, every package feels like
  a silo. Arrays made for PETSc cannot easily be used in Trilinos, and converting Sundials
  NVector outputs to DataFrames for post-simulation data processing is a process itself.
  The Julia SciML environment embraces interoperability. Don't wait for SciML to do it: by
  using generic coding with JIT compilation these connections create new optimized code on
  the fly and allow for a more expansive feature set than can ever be documented. Take
  [new high-precision number types from a package](https://github.com/JuliaArbTypes/ArbFloats.jl)
  and stick them into a nonlinear solver. Take
  [a package for Intel GPU arrays](https://github.com/JuliaGPU/oneAPI.jl) and stick it into
  the differential equation solver to use specialized hardware acceleration.
* **Wrappers to the Libraries You Know and Trust** - Moving to SciML does not have to be
  a quick transition. SciML has extensive wrappers to many widely-used classical solver
  environments such as [SUNDIALS](https://github.com/SciML/Sundials.jl) and
  [Hairer's classic Fortran ODE solvers (dopri5, dop853, etc.)](https://github.com/SciML/ODEInterfaceDiffEq.jl).
  Using these wrapped solvers is painless and can be swapped in for the Julia versions with
  one line of code. This gives you a way to incrementally adopt new features/methods
  while retaining the older pieces you know and trust.
* **Don't Start from Scratch** - SciML builds on the extensive
  [Base library of Julia](https://docs.julialang.org/en/v1/) and thus grows and improves
  with every update to the language. With hundreds of monthly contributors to SciML and
  hundreds of monthly contributors to Julia, SciML is one of the most actively developed
  open source scientific computing ecosystems out there!
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

## Why SciML? Some Technical Details

Let's face the facts, in the [open benchmarks](https://benchmarks.sciml.ai/dev/) the
pure-Julia solvers tend to outperform the classic "best" C++ and Fortran solvers in almost
every example (with a few notable exceptions). But why?

The answer is two-fold: Julia is as fast as C++/Fortran, and the algorithms are what matter.

### Julia is as Fast as C++/Fortran

While Julia code looks high level like Python or MATLAB, its performance is on par with
C++ and Fortran. At a technical level, when Julia code is type-stable, i.e. that the types
that are returned from a function are deducible at compile-time from the types that go into
a function, then Julia can optimize it as much as C++ or Fortran by automatically
devirtualizing all dynamic behavior and compile-time optimizing the quasi-static code.
This is not an empirical statement, it's a
[provable type-theoretic result](https://arxiv.org/abs/2109.01950). The resulting compiler
used on the resulting quasi-static representation is [LLVM](https://llvm.org/), the same
optimizing compiler used by [clang](https://clang.llvm.org/) and [LFortran](https://lfortran.org/).

For more details on how Julia code is optimized and how to optimize your own Julia code,
check out [this chapter from the SciML Book](https://book.sciml.ai/notes/02/).

### SciML's Julia Algorithms Have Performance Advantages in Many Common Regimes

There are many ways which Julia's algorithms achieve performance advantages. Some facts to
highlight include:

* Julia is at the forefront of numerical methods research in many domains. This is highlighted
  in [the differential equation solver comparisons](https://www.stochasticlifestyle.com/comparison-differential-equation-solver-suites-matlab-r-julia-python-c-fortran/)
  where the Julia solvers were the first to incorporate "newer" optimized Runge-Kutta tableaus,
  around half a decade before other software. Since then, the literature has only continued
  to evolve, and only Julia's SciML keeps up. At this point, many of the publication's first
  implementation is in [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) with
  benchmark results run on the
  [SciML Open Benchmarking platform!](https://benchmarks.sciml.ai/stable/)
* Julia does not take low-level mathematical functions for granted. The
  [common openlibm implementation of mathematical functions](https://openlibm.org/)
  used in many open source projects is maintained by the Julia and SciML developers!
  However, in modern Julia, every function from   `log` to `^` has been reimplemented in the
  Julia standard library to improve numerical correctness and performance. For example,
  [Pumas, the nonlinear mixed effects estimation system](https://www.biorxiv.org/content/10.1101/2020.11.28.402297v2)
  built on SciML and
  [used by Moderna for the vaccine trials](https://www.youtube.com/watch?v=6wGSCD3cI9E)
  notes in its paper that approximations to such math libraries itself gave a 2x performance
  improvement in even the most simple non-stiff ODE solvers over matching Fortran
  implementations. Pure Julia linear algebra tooling, like
  [RecursiveFactorization.jl for LU-factorization](https://github.com/JuliaLinearAlgebra/RecursiveFactorization.jl),
  outperforms common LU-factorization implementations used in open-source projects like
  OpenBLAS by around 5x! This should not be surprising though, given that OpenBLAS was
  a prior [MIT Julia Lab](https://julia.mit.edu/) project!
* Compilers are limited on the transformations that they can perform because they do not
  have high-level context-dependent mathematical knowledge. Julia's SciML makes
  [extensive use of customized symbolic-based compiler transformations](https://twitter.com/ChrisRackauckas/status/1477274812460449793)
  to improve performance with context-based code optimizations. Things like
  [sparsity patterns are automatically deduced from code and optimized on](https://openreview.net/pdf?id=rJlPdcY38B). [Nonlinear equations are symbolically-torn](https://www.youtube.com/watch?v=ZFoQihr3xLs), changing large nonlinear systems into sequential solving of much smaller
  systems and benefiting from an O(n^3) cost reduction. These can be orders of magnitude
  cost reductions which come for free, and unless you know every trick in the book it will
  be hard to match SciML's performance!
* Pervasive automatic differentiation mixed with compiler tricks wins battles. Many
  high-performance libraries in C++ and Fortran cannot assume that all of its code is
  compatible with automatic differentiation, and thus many internal performance tricks are
  not applied. For example
  [ForwardDiff.jl's chunk seeding](https://github.com/JuliaDiff/ForwardDiff.jl/blob/master/docs/src/dev/how_it_works.md) allows for a single call to `f` to generate multiple columns of a Jacobian.
  When mixed with [sparse coloring tools](https://github.com/JuliaDiff/SparseDiffTools.jl),
  entire Jacobians can be constructed with just a few `f` calls. Studies in applications
  have [shown this greatly outperforms finite differencing, especially when Julia's
  implicit multithreading is used](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009598).

### Let's Dig Deep Into One Case: Adjoints of ODEs for Solving Inverse Problems

To really highlight how JIT compilation and automatic differentiation integration can
change algorithms, let's look at the problem of differentiating an ODE solver. As is
[derived an discusssed in detail at a seminar with the American Statistical Association](https://www.youtube.com/watch?v=Xwh42RhB7O4),
there are many ways to implement well-known "adjoint" methods which are required for
performance. Each has different stability and performance trade-offs, and
[Julia's SciML is the only system to systemically offer all of the trade-off options](https://sensitivity.sciml.ai/stable/manual/differential_equation_sensitivities/). In many cases,
using analytical adjoints of a solver is not advised due to performance reasons, [with the
trade-off described in detail here](https://www.stochasticlifestyle.com/direct-automatic-differentiation-of-solvers-vs-analytical-adjoints-which-is-better/).
Likewise, even when analytical adjoints are used, it turns out that for general nonlinear
equations there is a trick which uses automatic differentiation in the construction of
the analytical adjoint to improve its performance. As demonstrated in
[this publication](https://ieeexplore.ieee.org/abstract/document/9622796/), this can lead
to about 2-3 orders of magnitude performance improvements. These AD-enhanced adjoints are
showcased as the seeding methods in this plot:

![](https://i0.wp.com/www.stochasticlifestyle.com/wp-content/uploads/2022/10/Capture7.png?w=2091&ssl=1)

Unless one directly defines special "vjp" functions, this is how the Julia SciML methods
achieve orders of magnitude performance advantages over CVODES's adjoints and PETSC's
TS-adjoint.

Moral of the story, even there are many reasons to use automatic differentiation of a solver,
and even if an analytical adjoint rule is used for some specific performance reason, that
analytical expression can often times be accelerated by orders of magnitude itself by embedding
some form of automatic differentiation into it. This is just one algorithm of many which
are optimized in this fashion.
