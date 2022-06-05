# The SciML Interface Libraries

## SciMLBase.jl: The SciML Common Interface

[SciMLBase.jl](https://github.com/SciML/SciMLBase.jl) defines the core interfaces
of the SciML libraries, such as the definitions of abstract types like `SciMLProblem`,
along with their instantiations like `ODEProblem`. While SciMLBase.jl is insufficient
to solve any equations, it holds all of the equation definitions, and thus downstream
libraries which wish to allow for using SciML solvers without depending on any solvers
can directly depend on SciMLBase.jl.

## SciMLOperators.jl: The AbstractSciMLOperator Interface

[SciMLOperators.jl](https://github.com/SciML/SciMLOperators.jl) defines the interface
for how matrix-free linear and affine operators are defined and used throughout the
SciML ecosystem.

## DiffEqNoiseProcess.jl: The SciML Common Noise Interface

[DiffEqNoiseProcess.jl](https://github.com/SciML/DiffEqNoiseProcess.jl) defines the
common interface for stochastic noise processes used by the equation solvers of the
SciML ecosystem.

## CommonSolve.jl: The Common Definition of Solve

[CommonSolve.jl](https://github.com/SciML/CommonSolve.jl) is the library that defines
the `solve`, `solve!`, and `init` interfaces which are used throughout all of the SciML
equation solvers. It's defined as an extremely lightweight library so that other
ecosystems can build off of the same `solve` definition without clashing with SciML
when both export.

## Static.jl: A Shared Interface for Static Compile-Time Computation

[Static.jl](https://github.com/SciML/Static.jl) is a set of statically parameterized types
for performing operations in a statically-defined (compiler-optimized) way with respect
to values. 

## DiffEqBase.jl: A Library of Shared Components for Differential Equation Solvers

[DiffEqBase.jl](https://github.com/SciML/DiffEqBase.jl) is the core shared component of the
DifferentialEquations.jl ecosystem. It's not intended for non-developer users to interface
directly with, instead it's used for the common functionality for uniformity of implementation
between the solver libraries.

# Third Party Libraries to Note

## ArrayInterface.jl: Extensions to the Julia AbstractArray Interface

[ArrayInterface.jl](https://github.com/JuliaArrays/ArrayInterface.jl) are traits and functions
which extend the Julia Base `AbstractArray` interface, giving a much larger set of queries
to allow for writing high-performance generic code over all array types. For example, functions
include `can_change_size` to know if an `AbstractArray` type is compatible with `resize!`,
`fast_scalar_indexing` to know whether direct scalar indexing `A[i]` is optimized, and functions
like `findstructralnz` to get the structural non-zeros of arbtirary sparse and structured matrices.

## Adapt.jl: Conversion to Allow Chip-Generic Programs

[Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) makes it possible to write code that is generic
to the compute devices, i.e. code that works on both CPUs and GPUs. It defines the `adapt` function
which acts like `convert(T, x)`, but without the restriction of returning a `T`. This allows you to 
"convert" wrapper types like `Adjoint` to be GPU compatible (for example) without throwing away the wrapper.

Example usage:

```julia
adapt(CuArray, ::Adjoint{Array})::Adjoint{CuArray}
```

## AbstractFFTs.jl: High Level Shared Interface for Fast Fourier Transformation Libraries

[AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl) defines the common interface for 
Fast Fourier Transformations (FFTs) in Julia. Similar to SciMLBase.jl, AbstractFFTs.jl is not a
solver library but instead a shared API which is extended by solver libraries such as
[FFTW.jl](https://github.com/JuliaMath/FFTW.jl). Code written using AbstractFFTs.jl can be made
compatible with FFT libraries without having an explicit dependency on a solver.

## GPUArrays.jl: Common Interface for GPU-Based Array Types

[GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl) defines the shared higher-level operations
for GPU-based array types like [CUDA.jl's CuArray](https://github.com/JuliaGPU/CUDA.jl) and
[AMDGPU.jl's ROCmArray](https://github.com/JuliaGPU/AMDGPU.jl). Packages in SciML use the designation
`x isa AbstractGPUArray` in order to find out if a user's operation is on the GPU and specialize
computations.

## Tables.jl: Common Interface for Tablular Data Types

[Tables.jl](https://github.com/JuliaData/Tables.jl) is a common interface for defining tabular data
structures, such as [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl). SciML's libraries
extend the Tables.jl interface to allow for automated conversions into data frame libraries without
explicit dependence on any singular implementation.

## EllipsisNotation.jl: Implementation of Ellipsis Array Slicing

[EllipsisNotation.jl](https://github.com/ChrisRackauckas/EllipsisNotation.jl) defines the ellipsis
array slicing notation for Julia. It uses `..` as a catch all for "all dimensions", allow for indexing
like `[..,1]` to mean "[:,:,:,1]` on four dimensional arrays, in a way that is generic to the number
of dimensions in the underlying array.