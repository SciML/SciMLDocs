# Function Approximation

While SciML is not an ecosystem for machine learning, SciML has many libraries for doing
machine learning with its equation solver libraries and machine learning libraries which
are integrated into the equation solvers.

## Surrogates.jl: Easy Generation of Differentiable Surrogate Models

[Surrogates.jl](https://github.com/SciML/Surrogates.jl) is a library for generating surrogate
approximations to computationally expensive simulations. It has the following
high-dimensional function approximators:

- Kriging
- Kriging using Stheno
- Radial Basis
- Wendland
- Linear
- Second Order Polynomial
- Support Vector Machines (Wait for LIBSVM resolution)
- Neural Networks
- Random Forests
- Lobachevsky splines
- Inverse-distance
- Polynomial expansions
- Variable fidelity
- Mixture of experts (Waiting GaussianMixtures package to work on v1.5)
- Earth
- Gradient Enhanced Kriging

## ReservoirComputing.jl: Fast and Flexible Reservoir Computing Methods

[ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl) is a library for
doing machine learning using reservoir computing techniques, such as with methods like Echo
State Networks (ESNs). Its reservoir computing methods make it stabilized for usage with
difficult equations like stiff dynamics, chaotic equations, and more.

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
