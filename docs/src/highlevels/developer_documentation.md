# Developer Documentation

For uniformity and clarity, the SciML Open-Source Software Organization has many
well-defined rules and practices for its development. However, we stress one
important principle:

**Do not be deterred from contributing if you think you do not know everything. No
one knows everything. These rules and styles are designed for iterative contributions.
Open pull requests and contribute what you can with what you know, and the maintainers
will help you learn and do the rest!**

If you need any help contributing, please feel welcome joining our community channels.

- The diffeq-bridged and sciml-bridged channels in the [Julia Zulip Chat](https://julialang.zulipchat.com/)
- The #diffeq-bridged and #sciml-bridged channels in the [Julia Slack](https://julialang.org/slack/)
- On the [Julia Discourse forums](https://discourse.julialang.org)
- See also [SciML Community page](https://sciml.ai/community/)

We welcome everybody.

## Getting Started With Contributing to SciML

To get started contributing to SciML, check out the following resources:

- [Developing Julia Packages](https://www.youtube.com/watch?v=QVmU29rCjaA)
- [Getting Started with Julia (for Experienced Programmers)](https://www.youtube.com/watch?v=-lJK92bEKow)

## SciMLStyle: The SciML Style Guide for Julia 

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

This is a style guide for how to program in Julia for SciML contributions. It describes
everything one needs to know, from preferred naming schemes of functions to fundamental
dogmas for designing traits. We stress that this style guide is meant to be comprehensive
for the sake of designing automatic formatters and teaching desired rules, but complete
knowledge and adherence to the style guide is not required for contributions!

## COLPRAC: Contributor's Guide on Collaborative Practices for Community Packages

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

What are the rules for when PRs should be merged? What are the rules for whether to tag
a major, minor, or patch release? All of these development rules are defined in COLPRAC.

## DiffEq Developer Documentation

There are many solver libraries which share similar internals, such as OrdinaryDiffEq.jl,
StochasticDiffEq.jl, and DelayDiffEq.jl. This section of the documentation describes the
internal systems of these packages and how they are used to quickly write efficient
solvers.

# Third-Party Libraries to Note

## Documenter.jl

[Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) is the documentation generation
library that the SciML organization uses, and thus its documentation is the documentation
of the documentation.

## JuliaFormatter.jl

[JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) is the formatter used by the
SciML organization to enforce the SciML Style. Setting `style = "sciml"` in a `.JuliaFormatter.toml`
file of a repo and using the standard FormatCheck.yml as part of continuous integration makes
JuliaFormatter check for SciML Style compliance on pull requests.

To run JuliaFormatter in a SciML repository, do:

```julia
using JuliaFomatter, DevedPackage
JuliaFormatter.format(pkgdir(DevedPackage))
```

which will reformat the code according to the SciML Style.

## GitHub Actions Continuous Integrations

The SciML Organization uses continuous integration testing to always ensure tests are passing when merging
pull requests. The organization uses the GitHub Actions supplied by [Julia Actions](https://github.com/julia-actions)
to accomplish this. Common continuous integration scripts are:

- CI.yml, the standard CI script
- Downstream.yml, used to specify packages for downstream testing. This will make packages which depend on the current
  package also be tested to ensure that “non-breaking changes” do not actually break other packages.
- Documentation.yml, used to run the documentation automatic generation with Documenter.jl
- FormatCheck.yml, used to check JuliaFormatter SciML Style compliance

## CompatHelper

[CompatHelper](https://github.com/JuliaRegistries/CompatHelper.jl) is used to automatically create pull requests whenever
a dependent package is upper bounded. The results of CompatHelper PRs should be checked to ensure that the latest version
of the dependencies are grabbed for the test process. After successful CompatHelper PRs, i.e. if the increase of the upper
bound did not cause a break to the tests, a new version tag should follow. It is set up by adding the CompatHelper.yml GitHub action.

## TagBot

[TagBot](https://github.com/JuliaRegistries/TagBot) automatically creates tags in the GitHub repository whenever a package
is registered to the Julia General repository. It is set up by adding the TagBot.yml GitHub action.
