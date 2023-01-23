# Model Libraries and Importers

Models are passed on from generation to generation. Many models are not built from scratch
but have a legacy of the known physics, biology, and chemistry embedded into them. Julia's
SciML offers a range of pre-built modeling tools, from reusable acausal components to
direct imports from common file formats.

## ModelingToolkitStandardLibrary.jl: A Standard Library for ModelingToolkit

Given the composable nature of acausal modeling systems, it's helpful to not have to define
every component from scratch and instead build off a common base of standard components.
ModelingToolkitStandardLibrary.jl is that library. It provides components for standard models
to start building everything from circuits and engines to robots.

![](https://user-images.githubusercontent.com/1814174/172000112-3579f5cf-c370-48c2-8047-558fbc46aeb6.png)

## DiffEqCallbacks.jl: Pre-Made Callbacks for DifferentialEquations.jl

DiffEqCallbacks.jl has many event handling and callback definitions which allow for
quickly building up complex differential equation models. It includes:

  - Callbacks for specialized output and saving procedures
  - Callbacks for enforcing domain constraints, positivity, and manifolds
  - Timed callbacks for periodic dosing, presetting of tstops, and more
  - Callbacks for determining and terminating at steady state
  - Callbacks for controlling stepsizes and enforcing CFL conditions
  - Callbacks for quantifying uncertainty with respect to numerical errors

## SBMLToolkit.jl: SBML Import

[SBMLToolkit.jl](https://github.com/SciML/SBMLToolkit.jl) is a library for reading
[SBML files](https://synonym.caltech.edu/#:%7E:text=What%20is%20SBML%3F,field%20of%20the%20life%20sciences.)
into the standard formats for Catalyst.jl and ModelingToolkit.jl. There are well over one thousand biological
models available in the [BioModels Repository](https://www.ebi.ac.uk/biomodels/).

## CellMLToolkit.jl: CellML Import

[CellMLToolkit.jl](https://github.com/SciML/CellMLToolkit.jl) is a library for reading
[CellML files](https://www.cellml.org/) into the standard formats for ModelingToolkit.jl.
There are several hundred biological models available in the
[CellML Model Repository](https://models.cellml.org/cellml).

## ReactionNetworkImporters.jl: BioNetGen Import

[ReactionNetworkImporters.jl](https://github.com/SciML/ReactionNetworkImporters.jl) is a library
for reading [BioNetGen .net files](https://bionetgen.org/) and various stoichiometry matrix representations
into the standard formats for Catalyst.jl and ModelingToolkit.jl.
