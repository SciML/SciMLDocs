# [SciML Package Index](@id package_index)

This page is the complete inventory of non-archived repositories in the
[SciML GitHub organization](https://github.com/SciML), plus a few archived
docs sites that are still published (DiffEqDevDocs, SciMLTutorialsOutput,
DiffEqOperators). Every package has a place here. Packages with their own
Documenter site appear in the **top navigation bar** of [docs.sciml.ai](https://docs.sciml.ai).
Some solver packages (OrdinaryDiffEq, Sundials, …) ship their API pages inside
[DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/); that is the intended home
for those docs rather than a second copy in the navigation bar.

| Placement | Meaning |
| --- | --- |
| dropdown | Top navigation (independent docs) |
| embedded | Embedded in another docs site |
| overview | Overview pages (no independent docs site yet) |
| extra | Learning resources / language bindings |
| infra | Not a user-facing package |

## Top navigation (independent docs)

### Advanced Solver APIs

| Package | Where | Summary |
| --- | --- | --- |
| [BoundaryValueDiffEq.jl](https://github.com/SciML/BoundaryValueDiffEq.jl) | Navigation → Advanced Solver APIs | Boundary-value problem solvers. |
| [DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl) | Navigation → Advanced Solver APIs | GPU ensemble acceleration for DiffEq. |
| [IRKGaussLegendre.jl](https://github.com/SciML/IRKGaussLegendre.jl) | Navigation → Advanced Solver APIs | 16th-order implicit RK Gauss–Legendre. |
| [MATLABDiffEq.jl](https://github.com/SciML/MATLABDiffEq.jl) | Navigation → Advanced Solver APIs | MATLAB ODE solver wrappers on the SciML interface. |
| [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) | Navigation → Advanced Solver APIs | High-performance ODE/DAE solvers; API also embedded in DiffEqDocs. |
| [OrdinaryDiffEqOperatorSplitting.jl](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl) | Navigation → Advanced Solver APIs | Operator-splitting solvers for split ODE/DAE formulations. |
| [QuantumNLDiffEq.jl](https://github.com/SciML/QuantumNLDiffEq.jl) | Navigation → Advanced Solver APIs | Differential quantum circuits for nonlinear DEs. |
| [SteadyStateDiffEq.jl](https://github.com/SciML/SteadyStateDiffEq.jl) | Navigation → Advanced Solver APIs | Steady-state solvers for DiffEq; API also in DiffEqDocs. |

### Array Libraries

| Package | Where | Summary |
| --- | --- | --- |
| [ComponentArrays.jl](https://github.com/SciML/ComponentArrays.jl) | Navigation → Array Libraries | Arrays with arbitrarily nested named components. |
| [LabelledArrays.jl](https://github.com/SciML/LabelledArrays.jl) | Navigation → Array Libraries | Named elements on arrays without overhead. |
| [MultiScaleArrays.jl](https://github.com/SciML/MultiScaleArrays.jl) | Navigation → Array Libraries | Multiscale array types that compose with equation solvers. |
| [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl) | Navigation → Array Libraries | Nested arrays compatible with SciML solvers. |

### Developer Documentation

| Package | Where | Summary |
| --- | --- | --- |
| [ColPrac](https://github.com/SciML/ColPrac) | Navigation → Developer Documentation | Collaborative practices for community packages. |
| [DiffEqDevDocs.jl](https://github.com/SciML/DiffEqDevDocs.jl) | Navigation → Developer Documentation | DiffEq developer documentation (archived repo, still published). |
| [OrgMaintenanceScripts.jl](https://github.com/SciML/OrgMaintenanceScripts.jl) | Navigation → Developer Documentation | Scripts for maintaining a large GitHub organization. |
| [SciMLStyle](https://github.com/SciML/SciMLStyle) | Navigation → Developer Documentation | SciML Julia style guide. |

### Equation Solvers

| Package | Where | Summary |
| --- | --- | --- |
| [Corleone.jl](https://github.com/SciML/Corleone.jl) | Navigation → Equation Solvers | Optimal control with SciML. |
| [DiffEqDocs.jl](https://github.com/SciML/DiffEqDocs.jl) | Navigation → Equation Solvers | DifferentialEquations.jl documentation (umbrella for the DiffEq solvers). |
| [DifferenceEquations.jl](https://github.com/SciML/DifferenceEquations.jl) | Navigation → Equation Solvers | Deterministic and stochastic difference equations and state-space likelihoods. |
| [Evolutionary.jl](https://github.com/SciML/Evolutionary.jl) | Navigation → Equation Solvers | Evolutionary and genetic algorithms. |
| [Integrals.jl](https://github.com/SciML/Integrals.jl) | Navigation → Equation Solvers | Unified interface for quadrature. |
| [JumpProcesses.jl](https://github.com/SciML/JumpProcesses.jl) | Navigation → Equation Solvers | Jump processes, Gillespie SSAs, and jump-diffusions. |
| [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) | Navigation → Equation Solvers | Unified interface for linear solvers. |
| [LineSearch.jl](https://github.com/SciML/LineSearch.jl) | Navigation → Equation Solvers | Unified line-search interface. |
| [NeuralLinearSolve.jl](https://github.com/SciML/NeuralLinearSolve.jl) | Navigation → Equation Solvers | CNN that picks UMFPACK/KLU/Pardiso for a sparse matrix. |
| [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) | Navigation → Equation Solvers | Unified interface for nonlinear solvers and bracketing. |
| [Optimization.jl](https://github.com/SciML/Optimization.jl) | Navigation → Equation Solvers | Unified interface for mathematical optimization. |

### Extra Resources

| Package | Where | Summary |
| --- | --- | --- |
| [ModelingToolkitCourse](https://github.com/SciML/ModelingToolkitCourse) | Navigation → Extra Resources | ModelingToolkit course notes. |
| [SciMLBenchmarksOutput](https://github.com/SciML/SciMLBenchmarksOutput) | Navigation → Extra Resources | Rendered SciML benchmarks site. |
| [SciMLTutorialsOutput](https://github.com/SciML/SciMLTutorialsOutput) | Navigation → Extra Resources | Rendered extended tutorials (archived repo, still published). |
| [SciMLWorkshop.jl](https://github.com/SciML/SciMLWorkshop.jl) | Navigation → Extra Resources | Workshop materials for SciML training. |

### Function Approximation

| Package | Where | Summary |
| --- | --- | --- |
| [ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl) | Navigation → Function Approximation | Reservoir computing. |
| [Surrogates.jl](https://github.com/SciML/Surrogates.jl) | Navigation → Function Approximation | Surrogate modeling and optimization. |

### High-Level Interfaces

| Package | Where | Summary |
| --- | --- | --- |
| [ADTypes.jl](https://github.com/SciML/ADTypes.jl) | Navigation → High-Level Interfaces | Automatic-differentiation backend types. |
| [CommonSolve.jl](https://github.com/SciML/CommonSolve.jl) | Navigation → High-Level Interfaces | Shared `solve` / `init` / `solve!` definition. |
| [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl) | Navigation → High-Level Interfaces | Core SciML problem/algorithm/solution interfaces. |
| [SciMLIterators.jl](https://github.com/SciML/SciMLIterators.jl) | Navigation → High-Level Interfaces | Iterators over SciML solutions and integrators. |
| [SciMLLogging.jl](https://github.com/SciML/SciMLLogging.jl) | Navigation → High-Level Interfaces | Verbosity and logging for SciML solvers. |
| [SciMLOperators.jl](https://github.com/SciML/SciMLOperators.jl) | Navigation → High-Level Interfaces | Matrix-free linear and affine operators. |
| [SciMLStructures.jl](https://github.com/SciML/SciMLStructures.jl) | Navigation → High-Level Interfaces | Queryable structure interface for user data and parameters. |
| [Static.jl](https://github.com/SciML/Static.jl) | Navigation → High-Level Interfaces | Statically parameterized types for compile-time computation. |
| [SurrogatesBase.jl](https://github.com/SciML/SurrogatesBase.jl) | Navigation → High-Level Interfaces | API for deterministic and stochastic surrogates. |
| [SymbolicIndexingInterface.jl](https://github.com/SciML/SymbolicIndexingInterface.jl) | Navigation → High-Level Interfaces | Symbolic indexing of SciML objects. |

### Implicit Layer Deep Learning

| Package | Where | Summary |
| --- | --- | --- |
| [DeepEquilibriumNetworks.jl](https://github.com/SciML/DeepEquilibriumNetworks.jl) | Navigation → Implicit Layer Deep Learning | Deep equilibrium networks. |
| [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) | Navigation → Implicit Layer Deep Learning | Prebuilt implicit deep-learning architectures (Neural ODEs, …). |
| [NeuralLyapunov.jl](https://github.com/SciML/NeuralLyapunov.jl) | Navigation → Implicit Layer Deep Learning | Search for neural Lyapunov functions. |

### Inverse Problems / Estimation

| Package | Where | Summary |
| --- | --- | --- |
| [CurveFit.jl](https://github.com/SciML/CurveFit.jl) | Navigation → Inverse Problems / Estimation | Least-squares and curve fitting on the CommonSolve interface. |
| [DiffEqBayes.jl](https://github.com/SciML/DiffEqBayes.jl) | Navigation → Inverse Problems / Estimation | Simplified Bayesian estimation of DiffEq parameters. |
| [DiffEqParamEstim.jl](https://github.com/SciML/DiffEqParamEstim.jl) | Navigation → Inverse Problems / Estimation | Simplified parameter-estimation loss functions. |
| [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) | Navigation → Inverse Problems / Estimation | Local sensitivity, adjoints, and AD of solvers. |

### Model Libraries and Importers

| Package | Where | Summary |
| --- | --- | --- |
| [BaseModelica.jl](https://github.com/SciML/BaseModelica.jl) | Navigation → Model Libraries and Importers | Base Modelica importer for ModelingToolkit. |
| [CellMLToolkit.jl](https://github.com/SciML/CellMLToolkit.jl) | Navigation → Model Libraries and Importers | CellML importer for ModelingToolkit. |
| [DiffEqCallbacks.jl](https://github.com/SciML/DiffEqCallbacks.jl) | Navigation → Model Libraries and Importers | Premade callbacks for hybrid differential-equation models. |
| [DiffEqFinancial.jl](https://github.com/SciML/DiffEqFinancial.jl) | Navigation → Model Libraries and Importers | Financial models (Heston, Black–Scholes, …) on DifferentialEquations. |
| [DiffEqPhysics.jl](https://github.com/SciML/DiffEqPhysics.jl) | Navigation → Model Libraries and Importers | Hamiltonian and physics-based DiffEq problem constructors. |
| [FiniteStateProjection.jl](https://github.com/SciML/FiniteStateProjection.jl) | Navigation → Model Libraries and Importers | Finite-state projection of chemical master equations. |
| [MathML.jl](https://github.com/SciML/MathML.jl) | Navigation → Model Libraries and Importers | MathML parser into Symbolics expressions. |
| [ModelingToolkitNeuralNets.jl](https://github.com/SciML/ModelingToolkitNeuralNets.jl) | Navigation → Model Libraries and Importers | Neural-network blocks for universal differential equations in MTK. |
| [ModelingToolkitStandardLibrary.jl](https://github.com/SciML/ModelingToolkitStandardLibrary.jl) | Navigation → Model Libraries and Importers | Standard component library for ModelingToolkit. |
| [PubChem.jl](https://github.com/SciML/PubChem.jl) | Navigation → Model Libraries and Importers | PubChem metadata attached to Catalyst species. |
| [Pyomo.jl](https://github.com/SciML/Pyomo.jl) | Navigation → Model Libraries and Importers | Pyomo interface via Symbolics.jl. |
| [ReactionNetworkImporters.jl](https://github.com/SciML/ReactionNetworkImporters.jl) | Navigation → Model Libraries and Importers | BioNetGen and stoichiometry-matrix importers. |
| [SBMLToolkit.jl](https://github.com/SciML/SBMLToolkit.jl) | Navigation → Model Libraries and Importers | SBML importer for Catalyst and ModelingToolkit. |

### Modeling Languages

| Package | Where | Summary |
| --- | --- | --- |
| [Catalyst.jl](https://github.com/SciML/Catalyst.jl) | Navigation → Modeling Languages | Chemical reaction networks and systems biology. |
| [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) | Navigation → Modeling Languages | Acausal symbolic modeling. |
| [MomentClosure.jl](https://github.com/SciML/MomentClosure.jl) | Navigation → Modeling Languages | Moment-closure equations for chemical master equations and SDEs. |
| [NBodySimulator.jl](https://github.com/SciML/NBodySimulator.jl) | Navigation → Modeling Languages | N-body, astrophysical, and molecular dynamics. |
| [ParameterizedFunctions.jl](https://github.com/SciML/ParameterizedFunctions.jl) | Navigation → Modeling Languages | Simple DSL for defining differential equations. |
| [ProcessSimulator.jl](https://github.com/SciML/ProcessSimulator.jl) | Navigation → Modeling Languages | Process simulation on ModelingToolkit. |

### Numerical Utilities

| Package | Where | Summary |
| --- | --- | --- |
| [BipartiteGraphs.jl](https://github.com/SciML/BipartiteGraphs.jl) | Navigation → Numerical Utilities | Bipartite graph types and utilities. |
| [ConcreteStructs.jl](https://github.com/SciML/ConcreteStructs.jl) | Navigation → Numerical Utilities | Macros for concrete struct field types. |
| [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) | Navigation → Numerical Utilities | 1D interpolation and smoothing. |
| [DataInterpolationsND.jl](https://github.com/SciML/DataInterpolationsND.jl) | Navigation → Numerical Utilities | N-dimensional interpolation on hyperrectangles. |
| [DiffEqNoiseProcess.jl](https://github.com/SciML/DiffEqNoiseProcess.jl) | Navigation → Numerical Utilities | Noise processes for SDEs and related solvers. |
| [EllipsisNotation.jl](https://github.com/SciML/EllipsisNotation.jl) | Navigation → Numerical Utilities | `..` ellipsis indexing. |
| [ExponentialUtilities.jl](https://github.com/SciML/ExponentialUtilities.jl) | Navigation → Numerical Utilities | Matrix exponentials, KIOPS, expmv, φ-functions. |
| [FastAlmostBandedMatrices.jl](https://github.com/SciML/FastAlmostBandedMatrices.jl) | Navigation → Numerical Utilities | Almost-banded matrices (used by BoundaryValueDiffEq). |
| [FastBroadcast.jl](https://github.com/SciML/FastBroadcast.jl) | Navigation → Numerical Utilities | `@..` broadcast that compiles to tight loops. |
| [FindFirstFunctions.jl](https://github.com/SciML/FindFirstFunctions.jl) | Navigation → Numerical Utilities | Faster specialized `findfirst`. |
| [FunctionWrappersWrappers.jl](https://github.com/SciML/FunctionWrappersWrappers.jl) | Navigation → Numerical Utilities | Double FunctionWrappers layer used in solver internals. |
| [LHLFactorization.jl](https://github.com/SciML/LHLFactorization.jl) | Navigation → Numerical Utilities | Hessenberg reduction for families of shifted linear systems. |
| [LightweightStats.jl](https://github.com/SciML/LightweightStats.jl) | Navigation → Numerical Utilities | Basic statistics with minimal dependencies. |
| [MuladdMacro.jl](https://github.com/SciML/MuladdMacro.jl) | Navigation → Numerical Utilities | `@muladd` fused multiply-add rewriting. |
| [PoissonRandom.jl](https://github.com/SciML/PoissonRandom.jl) | Navigation → Numerical Utilities | Fast Poisson random numbers. |
| [PreallocationTools.jl](https://github.com/SciML/PreallocationTools.jl) | Navigation → Numerical Utilities | Caches compatible with automatic differentiation. |
| [PureGebal.jl](https://github.com/SciML/PureGebal.jl) | Navigation → Numerical Utilities | Pure-Julia LAPACK xGEBAL-style matrix balancing. |
| [PureKLU.jl](https://github.com/SciML/PureKLU.jl) | Navigation → Numerical Utilities | Pure-Julia KLU sparse LU, no SuiteSparse_jll. |
| [PureUMFPACK.jl](https://github.com/SciML/PureUMFPACK.jl) | Navigation → Numerical Utilities | Pure-Julia UMFPACK-style sparse LU. |
| [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl) | Navigation → Numerical Utilities | Quasi-Monte Carlo sequences. |
| [RespecializeParams.jl](https://github.com/SciML/RespecializeParams.jl) | Navigation → Numerical Utilities | Type-stable opaque parameter containers for solvers. |
| [RootedTrees.jl](https://github.com/SciML/RootedTrees.jl) | Navigation → Numerical Utilities | Rooted trees and Runge–Kutta order conditions. |
| [RuntimeGeneratedFunctions.jl](https://github.com/SciML/RuntimeGeneratedFunctions.jl) | Navigation → Numerical Utilities | Runtime-generated functions without world-age issues. |
| [SparseBandedMatrices.jl](https://github.com/SciML/SparseBandedMatrices.jl) | Navigation → Numerical Utilities | Sparse banded matrices. |

### PDE Solvers

| Package | Where | Summary |
| --- | --- | --- |
| [DiffEqOperators.jl](https://github.com/SciML/DiffEqOperators.jl) | Navigation → PDE Solvers | Archived FDM operator library; prefer MethodOfLines. |
| [FEniCS.jl](https://github.com/SciML/FEniCS.jl) | Navigation → PDE Solvers | FEniCS finite-element wrappers. |
| [FiniteVolumeMethod.jl](https://github.com/SciML/FiniteVolumeMethod.jl) | Navigation → PDE Solvers | 2D finite-volume method for conservation laws. |
| [FiniteVolumeMethod1D.jl](https://github.com/SciML/FiniteVolumeMethod1D.jl) | Navigation → PDE Solvers | 1D finite-volume method. |
| [HighDimPDE.jl](https://github.com/SciML/HighDimPDE.jl) | Navigation → PDE Solvers | High-dimensional PDE solvers (Deep BSDE, Feynman–Kac). |
| [MethodOfLines.jl](https://github.com/SciML/MethodOfLines.jl) | Navigation → PDE Solvers | Automated finite-difference discretization of PDESystem. |
| [NeuralOperators.jl](https://github.com/SciML/NeuralOperators.jl) | Navigation → PDE Solvers | Fourier neural operators, DeepONets, and related operator learning. |
| [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) | Navigation → PDE Solvers | Physics-informed neural network PDE solvers. |

### Parameter Analysis

| Package | Where | Summary |
| --- | --- | --- |
| [CatalystNetworkAnalysis.jl](https://github.com/SciML/CatalystNetworkAnalysis.jl) | Navigation → Parameter Analysis | Network-analysis algorithms on Catalyst reaction networks. |
| [EasyModelAnalysis.jl](https://github.com/SciML/EasyModelAnalysis.jl) | Navigation → Parameter Analysis | High-level queries on simulation output. |
| [GlobalSensitivity.jl](https://github.com/SciML/GlobalSensitivity.jl) | Navigation → Parameter Analysis | Global sensitivity analysis (Sobol, Morris, eFAST, …). |
| [MinimallyDisruptiveCurves.jl](https://github.com/SciML/MinimallyDisruptiveCurves.jl) | Navigation → Parameter Analysis | Curves in parameter space that leave the solution nearly unchanged. |
| [StructuralIdentifiability.jl](https://github.com/SciML/StructuralIdentifiability.jl) | Navigation → Parameter Analysis | Structural identifiability of ODE models. |

### Symbolic Learning

| Package | Where | Summary |
| --- | --- | --- |
| [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl) | Navigation → Symbolic Learning | Data-driven dynamical systems and equation discovery. |
| [SymbolicNumericIntegration.jl](https://github.com/SciML/SymbolicNumericIntegration.jl) | Navigation → Symbolic Learning | Symbolic-numeric integration. |

### Symbolic Tools

| Package | Where | Summary |
| --- | --- | --- |
| [FunctionProperties.jl](https://github.com/SciML/FunctionProperties.jl) | Navigation → Symbolic Tools | Detects function properties (branches, etc.) for compiler/AD optimizations. |
| [ModelOrderReduction.jl](https://github.com/SciML/ModelOrderReduction.jl) | Navigation → Symbolic Tools | Automated model-order reduction on ModelingToolkit systems. |
| [SymbolicAnalysis.jl](https://github.com/SciML/SymbolicAnalysis.jl) | Navigation → Symbolic Tools | Disciplined-programming property propagation for optimization. |
| [SymbolicLimits.jl](https://github.com/SciML/SymbolicLimits.jl) | Navigation → Symbolic Tools | Symbolic limits and zero-equivalence of log-exp functions. |

### Uncertainty Quantification

| Package | Where | Summary |
| --- | --- | --- |
| [OptimalUncertaintyQuantification.jl](https://github.com/SciML/OptimalUncertaintyQuantification.jl) | Navigation → Uncertainty Quantification | Bounds on expectations without a fully specified input distribution. |
| [PolyChaos.jl](https://github.com/SciML/PolyChaos.jl) | Navigation → Uncertainty Quantification | Polynomial chaos expansions. |
| [SciMLExpectations.jl](https://github.com/SciML/SciMLExpectations.jl) | Navigation → Uncertainty Quantification | Fast expectations of equation solutions. |

## Embedded in another docs site

### Equation Solvers

| Package | Where | Summary |
| --- | --- | --- |
| [DASKR.jl](https://github.com/SciML/DASKR.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | DASKR DAE solver wrapper. |
| [DASSL.jl](https://github.com/SciML/DASSL.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | DASSL BDF DAE solver. |
| [deSolveDiffEq.jl](https://github.com/SciML/deSolveDiffEq.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | R deSolve wrappers on the SciML interface. |
| [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | Umbrella DiffEq package; documented as DiffEqDocs. |
| [GeometricIntegratorsDiffEq.jl](https://github.com/SciML/GeometricIntegratorsDiffEq.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | GeometricIntegrators.jl wrappers on the SciML interface. |
| [ODEInterfaceDiffEq.jl](https://github.com/SciML/ODEInterfaceDiffEq.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | Hairer Fortran ODEInterface wrappers. |
| [SciPyDiffEq.jl](https://github.com/SciML/SciPyDiffEq.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | SciPy ODE wrappers on the SciML interface. |
| [SimpleBoundaryValueDiffEq.jl](https://github.com/SciML/SimpleBoundaryValueDiffEq.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | Minimal BVP solvers. |
| [SimpleDiffEq.jl](https://github.com/SciML/SimpleDiffEq.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | Minimal no-cruft ODE solvers. |
| [Sundials.jl](https://github.com/SciML/Sundials.jl) | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | CVODE, ARKODE, IDA, and KINSOL wrappers. |

## Overview pages (no independent docs site yet)

### Developer Documentation

| Package | Where | Summary |
| --- | --- | --- |
| [SciMLTesting.jl](https://github.com/SciML/SciMLTesting.jl) | Overview pages | Shared GROUP-based test harness for SciML packages. |

### Equation Solvers

| Package | Where | Summary |
| --- | --- | --- |
| [BlackBoxOptim.jl](https://github.com/SciML/BlackBoxOptim.jl) | Overview pages | Derivative-free global/black-box optimization (OptimizationBBO). |
| [ComplementaritySolve.jl](https://github.com/SciML/ComplementaritySolve.jl) | Overview pages | Complementarity problems with ChainRules-compatible gradients. |
| [ParallelParticleSwarms.jl](https://github.com/SciML/ParallelParticleSwarms.jl) | Overview pages | GPU-accelerated particle-swarm optimization. |
| [SpatialBranchAndBound.jl](https://github.com/SciML/SpatialBranchAndBound.jl) | Overview pages | Spatial branch-and-bound. |

### Function Approximation

| Package | Where | Summary |
| --- | --- | --- |
| [YOLOWeights.jl](https://github.com/SciML/YOLOWeights.jl) | Overview pages | Pinned, checksummed Ultralytics YOLO ONNX weights. |

### High-Level Interfaces

| Package | Where | Summary |
| --- | --- | --- |
| [SciMLPublic.jl](https://github.com/SciML/SciMLPublic.jl) | Overview pages | `@public` backport of Julia 1.11's `public` keyword. |

### Model Libraries and Importers

| Package | Where | Summary |
| --- | --- | --- |
| [CasADi.jl](https://github.com/SciML/CasADi.jl) | Overview pages | CasADi interface via PythonCall (fork of ichatzinikolaidis/CasADi.jl). |
| [DiffEqProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl) | Overview pages | Premade DiffEq problems for examples and testing. |
| [PubChemReactions.jl](https://github.com/SciML/PubChemReactions.jl) | Overview pages | Reaction-network generation from PubChem data. |
| [SBMLToolkitTestSuite.jl](https://github.com/SciML/SBMLToolkitTestSuite.jl) | Overview pages | Runner for the SBML Test Suite against SBMLToolkit. |

### Numerical Utilities

| Package | Where | Summary |
| --- | --- | --- |
| [BinaryHeaps.jl](https://github.com/SciML/BinaryHeaps.jl) | Overview pages | Invalidation-free binary heaps extracted from DataStructures.jl. |
| [CommonWorldInvalidations.jl](https://github.com/SciML/CommonWorldInvalidations.jl) | Overview pages | Pre-invalidates common MethodError paths to cut latency. |
| [FastPower.jl](https://github.com/SciML/FastPower.jl) | Overview pages | Faster, slightly less accurate floating-point power. |
| [MaybeInplace.jl](https://github.com/SciML/MaybeInplace.jl) | Overview pages | Bang-bang macros that pick in-place vs out-of-place by array mutability. |
| [ResettableStacks.jl](https://github.com/SciML/ResettableStacks.jl) | Overview pages | Stacks with `reset!` that avoid GC in solver internals. |
| [SIMDRK.jl](https://github.com/SciML/SIMDRK.jl) | Overview pages | Generation of SIMD-compatible Runge–Kutta tableaus. |
| [SimpleNorm.jl](https://github.com/SciML/SimpleNorm.jl) | Overview pages | `norm` without a LinearAlgebra/BLAS dependency. |
| [SparseColumnPivotedQR.jl](https://github.com/SciML/SparseColumnPivotedQR.jl) | Overview pages | Rank-revealing column-pivoted Householder QR for SparseMatrixCSR. |
| [SparseMatrixIdentification.jl](https://github.com/SciML/SparseMatrixIdentification.jl) | Overview pages | Sparse matrix structure identification. |
| [SparseWithDenseRowColMatrices.jl](https://github.com/SciML/SparseWithDenseRowColMatrices.jl) | Overview pages | Sparse plus dense row/column matrices with Sherman–Morrison–Woodbury. |
| [SpecializingFactorizations.jl](https://github.com/SciML/SpecializingFactorizations.jl) | Overview pages | Type-stable detection of dense-matrix structure with specialized factorizations. |
| [TupleLU.jl](https://github.com/SciML/TupleLU.jl) | Overview pages | LU factorization on tuple-based small matrices. |

### PDE Solvers

| Package | Where | Summary |
| --- | --- | --- |
| [PDEBase.jl](https://github.com/SciML/PDEBase.jl) | Overview pages | Shared types and interface for ModelingToolkit PDE discretizers. |
| [PDESystemLibrary.jl](https://github.com/SciML/PDESystemLibrary.jl) | Overview pages | Library of ModelingToolkit PDESystem examples. |

### Plots and Visualization

| Package | Where | Summary |
| --- | --- | --- |
| [DimensionalPlotRecipes.jl](https://github.com/SciML/DimensionalPlotRecipes.jl) | Overview pages | Plot recipes for high-dimensional numbers and reductions. |

### Symbolic Learning

| Package | Where | Summary |
| --- | --- | --- |
| [DataCollocations.jl](https://github.com/SciML/DataCollocations.jl) | Overview pages | Non-parametric collocation for smoothing timeseries and estimating derivatives. |

## Learning resources / language bindings

### Extra Resources

| Package | Where | Summary |
| --- | --- | --- |
| [2025-JuliaCon-DifferentialEquations-Workshop](https://github.com/SciML/2025-JuliaCon-DifferentialEquations-Workshop) | Learning resources | JuliaCon 2025 DifferentialEquations workshop. |
| [Catalyst_PLOS_COMPBIO_2023](https://github.com/SciML/Catalyst_PLOS_COMPBIO_2023) | Learning resources | Companion notebooks for the 2023 Catalyst PLOS Comp Bio paper. |
| [GeiloInverseProblemWorkshop](https://github.com/SciML/GeiloInverseProblemWorkshop) | Learning resources | Geilo inverse-problems workshop notes. |
| [Julia_Modeling_Workshop](https://github.com/SciML/Julia_Modeling_Workshop) | Learning resources | High-performance scientific modeling workshop. |
| [ModelingToolkitWorkshop_JuliaCon2024](https://github.com/SciML/ModelingToolkitWorkshop_JuliaCon2024) | Learning resources | JuliaCon 2024 ModelingToolkit workshop materials. |
| [Scientific_Modeling_Cheatsheet](https://github.com/SciML/Scientific_Modeling_Cheatsheet) | Learning resources | MATLAB / Python / Julia scientific-modeling cheatsheet. |
| [SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl) | Learning resources | Benchmark sources; rendered output is SciMLBenchmarksOutput. |
| [SciMLBook](https://github.com/SciML/SciMLBook) | Learning resources | Parallel Computing and Scientific Machine Learning lecture notes (MIT 18.337). |

### Language Bindings

| Package | Where | Summary |
| --- | --- | --- |
| [diffeqpy](https://github.com/SciML/diffeqpy) | Learning resources | Python bindings for DifferentialEquations.jl. |
| [diffeqr](https://github.com/SciML/diffeqr) | Learning resources | R bindings for DifferentialEquations.jl. |
| [juliatorch](https://github.com/SciML/juliatorch) | Learning resources | Wrap Julia functions as PyTorch autograd functions. |

## Not a user-facing package

### Infrastructure

| Package | Where | Summary |
| --- | --- | --- |
| [.github](https://github.com/SciML/.github) | Not documented as a package | Organization-wide GitHub Actions and metadata. |
| [demo-repository](https://github.com/SciML/demo-repository) | Not documented as a package | GitHub demo repository, not a SciML package. |
| [LinearSolveAutotuneResults.jl](https://github.com/SciML/LinearSolveAutotuneResults.jl) | Not documented as a package | Stored LinearSolve autotune results, not a solver package. |
| [ModelDiscovery.jl](https://github.com/SciML/ModelDiscovery.jl) | Not documented as a package | Empty placeholder repository. |
| [OptimalUncertaintyQuantification-DEV.jl](https://github.com/SciML/OptimalUncertaintyQuantification-DEV.jl) | Not documented as a package | Development duplicate of OptimalUncertaintyQuantification.jl. |
| [PropertyModels.jl](https://github.com/SciML/PropertyModels.jl) | Not documented as a package | Empty placeholder repository. |
| [sciml.ai](https://github.com/SciML/sciml.ai) | Not documented as a package | Organization website. |
| [SciMLAssets](https://github.com/SciML/SciMLAssets) | Not documented as a package | Shared website/assets. |
| [SciMLDocs](https://github.com/SciML/SciMLDocs) | Not documented as a package | This documentation aggregator (Home in the nav). |
