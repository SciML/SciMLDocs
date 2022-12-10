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
[massively parallel GPU ODE solving showcase](@ref gpuspde).

## Problem Setup

## Defining the Ensemble Problem for CPU

```@example diffeqgpu
using DiffEqGPU, OrdinaryDiffEq, StaticArrays
function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
@time sol = solve(monteprob,Tsit5(),EnsembleThreads(),trajectories=10_000,saveat=1.0f0)
```

```@example diffeqgpu
@time sol = solve(monteprob,Tsit5(),EnsembleThreads(),trajectories=10_000,saveat=1.0f0)
```

## Taking the Ensemble to the GPU

```@example diffeqgpu
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0)
```

```@example diffeqgpu
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0)
```

```@example diffeqgpu
@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000, adaptive = false, dt = 0.1f0)
```

```@example diffeqgpu
@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000, adaptive = true, dt = 0.1f0, save_everystep = false)
```
