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
[SBML files](https://sbml.org/)
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

## ModelingToolkitNeuralNets.jl: Neural Network Blocks in Acausal Models

[ModelingToolkitNeuralNets.jl](https://github.com/SciML/ModelingToolkitNeuralNets.jl) provides
neural-network components in the style of ModelingToolkitStandardLibrary, so a universal
differential equation can be wired into part of an `ODESystem` through `RealInputArray` /
`RealOutputArray` connectors.

## BaseModelica.jl: Base Modelica Import

[BaseModelica.jl](https://github.com/SciML/BaseModelica.jl) parses Base Modelica models into
Julia objects and converts them to ModelingToolkit systems.

## DiffEqPhysics.jl: Hamiltonian and Physics-Based Problem Constructors

[DiffEqPhysics.jl](https://github.com/SciML/DiffEqPhysics.jl) builds differential-equation
problems from physical descriptions such as Hamiltonians, for use with the DiffEq solvers.

## PubChem.jl and PubChemReactions.jl: Chemical Data Import

[PubChem.jl](https://github.com/SciML/PubChem.jl) attaches PubChem metadata to Catalyst species.
[PubChemReactions.jl](https://github.com/SciML/PubChemReactions.jl) generates reaction networks
from PubChem data.

## Pyomo.jl: Pyomo via Symbolics.jl

[Pyomo.jl](https://github.com/SciML/Pyomo.jl) is a Julia interface to Pyomo that builds
expressions through Symbolics.jl, so nonlinear optimization and DAE models defined in Pyomo
can interoperate with the Julia symbolic stack.

## MathML.jl: MathML Parser

[MathML.jl](https://github.com/SciML/MathML.jl) parses MathML into Symbolics.jl expressions.

## CasADi.jl: CasADi Interface

[CasADi.jl](https://github.com/SciML/CasADi.jl) is a Julia interface to CasADi via PythonCall.
