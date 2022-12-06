# [Massively Data-Parallel ODE Solving on GPUs](@id datagpu)

## Before we start: the two ways to accelerate ODE solvers with GPUs

Before we dive deeper, let us remark that there are two very different ways that one can
accelerate an ODE solution with GPUs. There is one case where `u` is very big and `f`
is very expensive but very structured, and you use GPUs to accelerate the computation
of said `f`. The other use case is where `u` is very small but you want to solve the ODE
`f` over many different initial conditions (`u0`) or parameters `p`. In that case, you can
use GPUs to parallelize over different parameters and initial conditions. In other words:

| Type of Problem                           | SciML Solution                                                                                           |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Accelerate a big ODE                      | Use [CUDA.jl's](https://cuda.juliagpu.org/stable/) CuArray as `u0`                                       |
| Solve the same ODE with many `u0` and `p` | Use [DiffEqGPU.jl's](https://docs.sciml.ai/DiffEqGPU/stable/) `EnsembleGPUArray` and `EnsembleGPUKernel` |

This showcase will focus on the latter case. For the former, see the
[massively parallel GPU ODE solving showcase](@ref gpu_spde).
