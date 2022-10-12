# SciML-Supported Plotting and Visualization Libraries

The following libraries are the plotting and visualization libraries which are supported
and co-developed by the SciML developers. Other libraries may be used, though these are
the libraries used in the tutorials and which have special hooks to ensure ergonomic usage
with SciML tooling.

## Plots.jl

[Plots.jl](https://github.com/JuliaPlots/Plots.jl) is the current standard plotting system
for the SciML ecosystem. SciML types attempt to include plot recipes for as many types as
possible, allowing for automatic visualization with the Plots.jl system. All current
tutorials and documentation default to using Plots.jl.

## Makie.jl

[Makie.jl](https://makie.juliaplots.org/stable/) is a high-performance interactive plotting
system for the Julia programming language. It's planned to be the default plotting system
used by the SciML organization in the near future.
