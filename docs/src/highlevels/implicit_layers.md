# Implicit Layer Deep Learning

Implicit layer deep learning is a field which uses implicit rules, such as differential
equations and nonlinear solvers, to define the layers of neural networks. This field has
brought the potential to automatically optimize network depth and improve training
performance. SciML's differentiable solver ecosystem is specifically designed to accomodate
implicit layer methodologies, and provides libraries with pre-built layers for common
methods.

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

## DeepEquilibriumNetworks.jl: Deep Equilibrium Models Made Fast

[DeepEquilibriumNetworks.jl](https://github.com/SciML/DeepEquilibriumNetworks.jl)
is a library of optimized layer implementations for Deep Equilibrium Models (DEQs). It uses
special training techniques such as implicit-explicit regularization in order to accelerate
the convergence over traditional implementations, all while using the optimized and flexible
SciML libraries under the hood.
