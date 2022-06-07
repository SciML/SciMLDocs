# SciML Symbolic Analysis Libraries

## StructuralIdentifiability.jl: Identifiability Analysis Made Simple

Performing parameter estimation from a data set means attempting to recover parameters
like reaction rates by fitting some model to the data. But how do you know whether you
have enough data to even consider getting the "correct" parameters back? 
[StructuralIdentifiability.jl](https://github.com/SciML/StructuralIdentifiability.jl)
allows for running a structural identifiability analysis on a given model to determine
whether it's theoretically possible to recover the correct parameters. It can state whether
a given type of output data can be used to globally recover the parameters (i.e. only a
unique parameter set for the model produces a given output), whether the parameters are
only locally identifiable (i.e. there are finitely many parameter sets which could generate
the seen data), or whether it's unidentifiable (there are infinitely many parameters which
generate the same output data).

For more information on what StructuralIdentifiability.jl is all about, see the
[SciMLCon 2022 tutorial video](https://www.youtube.com/watch?v=jg1DME3cwjg).

## SymbolicNumericIntegration.jl: Symbolic Integration via Numerical Methods

[SymbolicNumericIntegration.jl](https://github.com/SciML/SymbolicNumericIntegration.jl)
is a package computing the solution to symbolic integration problem using numerical methods
(numerical integration mixed with sparse regression).

# JuliaSymbolics

[JuliaSymbolics](https://juliasymbolics.org/) is a sister organization of SciML. It spawned
out of the symbolic modeling tools being developed within SciML 
([ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)) to become its own
organization dedicated to building a fully-featured Julia-based Computer Algebra System (CAS).
As such, the two organizations are closely aligned in terms of its developer community and
many of the SciML libraries use Symbolics.jl extensively.

## Symbolics.jl: The Computer Algebra System (CAS) of the Julia Programming Language

[Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) is the CAS of the Julia programming
language. If something needs to be done symbolically, most likely Symbolics.jl is the answer.

## MetaTheory.jl: E-Graphs to Automate Symbolic Transformations

[Metatheory.jl](https://github.com/JuliaSymbolics/MetaTheory.jl) is a library for defining e-graph
rewriters for use on the common symbolic interface. This can be used to do all sorts of analysis
and code transformations, such as improving code performance, numerical stability, and more.
See [Automated Code Optimization with E-Graphs](https://arxiv.org/abs/2112.14714) for more details.

## SymbolicUtils.jl: Define Your Own Computer Algebra System

[SymbolicUtils.jl](https://github.com/JuliaSymbolics/SymbolicUtils.jl) is the underlying utility
library and rule-based rewriting language on which Symbolics.jl is developed. Symbolics.jl is
standardized type and rule definitions built using SymbolicUtils.jl. However, if non-standard
types are required, such as [symbolic computing over Fock algebras](https://github.com/qojulia/QuantumCumulants.jl),
then SymbolicUtils.jl is the library from which the new symbolic types can be implemented.

# Third Party Libraries to Note

## SIAN.jl: Structural Identifiability Analyzer

[SIAN.jl](https://github.com/alexeyovchinnikov/SIAN-Julia) is a structural identifiability analysis
package which uses an entirely different algorithm from StructuralIdentifiability.jl. For information
on the differences bewteen the two approaches, see 
[the Structural Identifiability Tools in Julia tutoral](https://www.youtube.com/watch?v=jg1DME3cwjg).