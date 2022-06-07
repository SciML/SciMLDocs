# SciML Machine Learning Libraries Overview

While SciML is not an ecosystem for machine learning, SciML has many libraries for doing
machine learning with its equation solver libraries and machine learning libraries which
are integrated into the equation solvers.

## DiffEqFlux.jl: High Level Pre-Built Architectures for Implicit Deep Learning

[DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) is a library of pre-built architectures
for implicit deep learning, including layer definitions for methods like:

- [Neural Ordinary Differential Equations (Neural ODEs)](https://arxiv.org/abs/1806.07366)
- [Collocation-Based Neural ODEs (Neural ODEs without a solver, by far the fastest way!)](https://www.degruyter.com/document/doi/10.1515/sagmb-2020-0025/html)
- [Multiple Shooting Neural Ordinary Differential Equations](https://arxiv.org/abs/2109.06786)
- [Neural Stochastic Differential Equations (Neural SDEs)](https://arxiv.org/abs/1907.07587)
- [Neural Differential-Algebriac Equations (Neural DAEs)](https://arxiv.org/abs/2001.04385)
- [Neural Delay Differential Equations (Neural DDEs)](https://arxiv.org/abs/2001.04385)
- [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681)
- [Hamiltonian Neural Networks (with specialized second order and symplectic integrators)](https://arxiv.org/abs/1906.01563)
- [Continuous Normalizing Flows (CNF)](https://arxiv.org/abs/1806.07366) and [FFJORD](https://arxiv.org/abs/1810.01367)

## ReservoirComputing.jl: Fast and Flexible Reservoir Computing Methods

[ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl) is a library for
doing machine learning using reservoir computing techniques, such as with methods like Echo
State Networks (ESNs). Its reservoir computing methods make it stabilized for usage with
difficult equations like stiff dynamics, chaotic equations, and more.

## DeepEquilibriumNetworks.jl: Deep Equilibrium Models Made Fast

[FastDEQ.jl](https://github.com/SciML/FastDEQ.jl) is a library of optimized layer implementations
for Deep Equilibrium Models (DEQs). It uses special training techniques such as implicit-explicit
regularization in order to accelerate the convergence over traditional implementations, all while
using the optimized and flexible SciML libraries under the hood.

# Third Party Libraries to Note

## Flux.jl: the ML library that doesn't make you tensor

[Flux.jl](https://github.com/FluxML/Flux.jl) is the most popular machine learning library in the
Julia programming language. SciML's libraries are heavily tested with it and its automatic
differentiation engine [Zygote.jl](https://github.com/FluxML/Zygote.jl) for composability and
compatibility.

## Lux.jl: Explicitly Parameterized Neural Networks in Julia

[Lux.jl](https://github.com/avik-pal/Lux.jl) is a library for fully explicitly parameterized
neural networks. Thus while alternative interfaces are required to use Flux with many equation
solvers (i.e. `Flux.destructure`), Lux.jl's explicit design merries very easily with the
SciML equation solver libraries. For this reason, SciML's library are also heavily tested with
Lux to ensure compatibility with neural network definitions from here.

## SimpleChains.jl: Fast Small-Scale Machine Learning

[SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) is a library specialized for
small-scale machine learning. It uses non-allocating mutating forms to be highly efficient
for the cases where matrix multiplication kernels are not able to overcome the common overheads
of machine learning libraries. Thus for SciML cases with small neural networks (<100 node layers) 
and non-batched usage (many/most use cases), SimpleChains.jl can be the fastest choice for the
neural network definitions.

## NNLib.jl: Neural Network Primitives with Multiple Backends

[NNLib.jl](https://github.com/FluxML/NNlib.jl) is the core library which defines the handling
of common functions, like `conv` and how they map to device accelerators such as the NVIDA
cudnn. This library can thus be used to directly grab many of the core functions used in
machine learning, such as common activation functions and gather/scatter operations, without
depending on the given style of any machine learning library.

## GeometricFlux.jl: Geometric Deep Learning and Graph Neural Networks

[GeometricFlux.jl](https://github.com/FluxML/GeometricFlux.jl) is a library for graph neural
networks and geometric deep learning. It is the one that is used and tested by the SciML
developers for mixing with equation solver applications.

## AbstractGPs.jl: Fast and Flexible Gaussian Processes

[AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) is the fast and
flexible Gaussian Process library that is used by the SciML packages and recommended
for downstream usage.

## MLDatasets.jl: Common Machine Learning Datasets

[MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl)  is a common interface for
accessing common machine learning datasets. For example, if you want to run a test on
MNIST data, MLDatasets is the quicket way to obtain it.

## MLUtils.jl: Utility Functions for Machine Learning Pipelines

[MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) is a library of utility functions for
making writing common machine learning pipelines easier. This includes functionality for:

- An extensible dataset interface  (`numobs` and `getobs`).
- Data iteration and dataloaders (`eachobs` and `DataLoader`).
- Lazy data views (`obsview`). 
- Resampling procedures (`undersample` and `oversample`).
- Train/test splits (`splitobs`) 
- Data partitioning and aggregation tools (`batch`, `unbatch`, `chunk`, `group_counts`, `group_indices`).
- Folds for cross-validation (`kfolds`, `leavepout`).
- Datasets lazy tranformations (`mapobs`, `filterobs`, `groupobs`, `joinobs`, `shuffleobs`).
- Toy datasets for demonstration purpose. 
- Other data handling utilities (`flatten`, `normalise`, `unsqueeze`, `stack`, `unstack`).