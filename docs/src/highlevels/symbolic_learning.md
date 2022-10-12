# Symbolic Learning and Artificial Intelligence

Symbolic learning, the classical artificial intelligence, is a set of methods for learning
symbolic equations from data and numerical functions. SciML offers an array of symbolic
learning utilities which connect with the other machine learning and equation solver
functionalities to make it easy to embed prior knowledge and discover missing physics.
For more information, see
[Universal Differential Equations for Scientific Machine Learning](https://arxiv.org/abs/2001.04385)

## DataDrivenDiffEq.jl: Data-Driven Modeling and Automated Discovery of Dynamical Systems

DataDrivenDiffEq.jl is a general interface for data-driven modeling, containing a large
array of techniques such as:

* Koopman operator methods (Dynamic-Mode Decomposition (DMD) and variations)
* Sparse Identification of Dynamical Systems (SINDy and variations like iSINDy)
* Sparse regression methods (STSLQ, SR3, etc.)
* PDEFind
* Wrappers for SymbolicRegression.jl
* AI Feynman
* OccamNet

## SymbolicNumericIntegration.jl: Symbolic Integration via Numerical Methods

[SymbolicNumericIntegration.jl](https://github.com/SciML/SymbolicNumericIntegration.jl)
is a package computing the solution to symbolic integration problem using numerical methods
(numerical integration mixed with sparse regression).

# Third-Party Libraries to Note

## SymbolicRegression.jl

[SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) is a
symbolic regression library which uses genetic algorithms with parallelization to achieve
fast and robust symbolic learning.
