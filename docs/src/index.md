# SciML: Differentiable Modeling and Simulation Combined with Machine Learning

The SciML organization is a collection of tools for solving equations and modeling systems
developed in the Julia programming language with bindings to other languages such as R and
Python. The organization provides well-maintained tools which compose together as a
coherent ecosystem. It has a coherent development principle, unified APIs over large
collections of equation solvers, pervasive differentiability and sensitivity analysis, and
features many of the highest performance and parallel implementations one can find.

**Scientific Machine Learning (SciML) = Scientific Computing + Machine Learning**

## Where to Start?

  - Want to get started running some code? Check out the [Getting Started tutorials](@ref getting_started).
  - What is SciML? Check out our [Overview](@ref overview).
  - Want to see some cool end-to-end examples? Check out the [SciML Showcase](@ref showcase).
  - Curious about our performance claims? Check out [the SciML Open Benchmarks](https://benchmarks.sciml.ai/dev/).
  - Want to learn more about how SciML does scientific machine learning? Check out the [SciML Book (from MIT's 18.337 graduate course)](https://book.sciml.ai/).
  - Want to chat with someone? Check out [our chat room](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged) and [forums](https://discourse.julialang.org/).
  - Want to see our code? Check out [the SciML GitHub organization](https://github.com/SciML).

And for diving into the details, use the bar on the top to navigate to the submodule of
interest!

## Reproducibility

```@raw html
<details><summary>The documentation of the <a href="showcase/showcase/#showcase">SciML Showcase</a> was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
