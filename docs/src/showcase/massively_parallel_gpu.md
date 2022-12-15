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

Let's say we wanted to quantify the uncertainty in the solution of a differential equation.
One simple way to do this would be to a Monte Carlo simulation of the same ODE, randomly
jiggling around some parameters according to an uncertainty distribution. We could do
that on a CPU, but that's not hip. What's hip are GPUs! GPUs have thousands of cores, so
could we make each core of our GPU solve the same ODE but with different parameters?
The [ensembling tools of DiffEqGPU.jl](https://docs.sciml.ai/DiffEqGPU/stable/) solve
exactly this issue, and today you will learn how to master the GPUniverse.

Let's dive right in.

## Defining the Ensemble Problem for CPU

DifferentialEquations.jl
[has an ensemble interface for solving many ODEs](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/).
DiffEqGPU conveniently uses exactly the same interface, so just a change of a few characters
is all that's required to change a CPU-parallelized code into a GPU-paralleized code.
Given that, let's start with the CPU-parallelized code.

Let's implement the Lorenz equation out-of-place. If you don't know what that means,
see the [getting started with DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/)


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
```

Notice we use `SVector`s, i.e. StaticArrays, in order to define our arrays. This is
important for later since the GPUs will want a fully non-allocating code to build a
kernel on.

Now from this problem, we build an `EnsembleProblem` as per the DifferentialEquations.jl
specification. A `prob_func` jiggles the parameters and we solve 10_000 trajectories:

```@example diffeqgpu
prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
sol = solve(monteprob,Tsit5(),EnsembleThreads(),trajectories=10_000,saveat=1.0f0)
```

## Taking the Ensemble to the GPU

Now uhh, we just change `EnsembleThreads()` to `EnsembleGPUArray()`

```@example diffeqgpu
sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0)
```

Or for a more efficient version, `EnsembleGPUKernel()`. But that requires special solvers
so we also change to `GPUTsit5()`.

```@example diffeqgpu
sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000)
```

Okay so that was anticlimactic but that's the point: if it was harder then that then it
wouldn't be automatic! Now go check out [DiffEqGPU.jl's documentation for more details,](https://docs.sciml.ai/DiffEqGPU/stable/)
that's the end of our show.
