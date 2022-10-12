# Symbolic Tooling and JuliaSymbolics

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
