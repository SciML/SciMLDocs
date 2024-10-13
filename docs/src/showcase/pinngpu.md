# [GPU-Accelerated Physics-Informed Neural Network (PINN) PDE Solvers](@id pinngpu)

Machine learning is all the rage. Everybody thinks physics is cool.

Therefore, using machine learning to solve physics equations? 🧠💥

So let's be cool and use a physics-informed neural network (PINN) to solve the Heat
Equation. Let's be even cooler by using GPUs (ironically, creating even more heat, but
it's the heat equation so that's cool).

## Step 1: Import Libraries

To solve PDEs using neural networks, we will use the
[NeuralPDE.jl package](https://neuralpde.sciml.ai/stable/). This package uses
ModelingToolkit's symbolic `PDESystem` as an input, and it generates an
[Optimization.jl](https://docs.sciml.ai/Optimization/stable/) `OptimizationProblem` which,
when solved, gives the weights of the neural network that solve the PDE. In the end, our
neural network `NN` satisfies the PDE equations and is thus the solution to the PDE! Thus
our packages look like:

```@example pinn
# High Level Interface
using NeuralPDE
import ModelingToolkit: Interval

# Optimization Libraries
using Optimization, OptimizationOptimisers

# Machine Learning Libraries and Helpers
using Lux, LuxCUDA, ComponentArrays
const gpud = gpu_device() # allocate a GPU device

# Standard Libraries
using Printf, Random

# Plotting
using Plots
```

## Problem Setup

Let's solve the 2+1-dimensional Heat Equation. This is the PDE:

```math
∂_t u(x, y, t) = ∂^2_x u(x, y, t) + ∂^2_y u(x, y, t) \, ,
```

with the initial and boundary conditions:

```math
\begin{align*}
u(x, y, 0) &= e^{x+y} \cos(x + y)      \, ,\\
u(0, y, t) &= e^{y}   \cos(y + 4t)     \, ,\\
u(2, y, t) &= e^{2+y} \cos(2 + y + 4t) \, ,\\
u(x, 0, t) &= e^{x}   \cos(x + 4t)     \, ,\\
u(x, 2, t) &= e^{x+2} \cos(x + 2 + 4t) \, ,
\end{align*}
```

on the space and time domain:

```math
x \in [0, 2] \, ,\ y \in [0, 2] \, , \ t \in [0, 2] \, ,
```

with physics-informed neural networks.

## Step 2: Define the PDESystem

First, let's use ModelingToolkit's `PDESystem` to represent the PDE. To do this, basically
just copy-paste the PDE definition into Julia code. This looks like:

```@example pinn
@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
t_min = 0.0
t_max = 2.0
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 2.0

# 2D PDE
eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)
# Initial and boundary conditions
bcs = [u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
    u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
    u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
    u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
    u(t, x, y_max) ~ analytic_sol_func(t, x, y_max)]

# Space and time domains
domains = [t ∈ Interval(t_min, t_max),
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)]

@named pde_system = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
```

!!! note
    
    We used the wildcard form of the variable definition `@variables u(..)` which then
    requires that we always specify what the dependent variables of `u` are. This is because in the boundary conditions we change from using `u(t,x,y)` to
    more specific points and lines, like `u(t,x_max,y)`.

## Step 3: Define the Lux Neural Network

Now let's define the neural network that will act as our solution.
We will use a simple multi-layer perceptron, like:

```@example pinn
using Lux
inner = 25
chain = Chain(Dense(3, inner, Lux.σ),
    Dense(inner, inner, Lux.σ),
    Dense(inner, inner, Lux.σ),
    Dense(inner, inner, Lux.σ),
    Dense(inner, 1))
ps = Lux.setup(Random.default_rng(), chain)[1]
```

## Step 4: Place it on the GPU.

Just plop it on that sucker. We must ensure that our initial parameters for the neural
network are on the GPU. If that is done, then the internal computations will all take place
on the GPU. This is done by using the `gpud` function (i.e. the GPU
device we created at the start) on the initial parameters, like:

```@example pinn
ps = ps |> ComponentArray |> gpud .|> Float64
```

## Step 5: Discretize the PDE via a PINN Training Strategy

```@example pinn
strategy = GridTraining(0.05)
discretization = PhysicsInformedNN(chain,
    strategy,
    init_params = ps)
prob = discretize(pde_system, discretization)
```

## Step 6: Solve the Optimization Problem

```@example pinn
callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, Adam(0.01); callback = callback, maxiters = 2500);
```

We then use the `remake` function to rebuild the PDE problem to start a new
optimization at the optimized parameters, and continue with a lower learning rate:

```@example pinn
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, Adam(0.001); callback = callback, maxiters = 2500);
```

## Step 7: Inspect the PINN's Solution

Finally, we inspect the solution:

```julia
phi = discretization.phi
ts, xs, ys = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
u_real = [analytic_sol_func(t, x, y) for t in ts for x in xs for y in ys]
u_predict = [first(Array(phi([t, x, y], res.u))) for t in ts for x in xs for y in ys]

function plot_(res)
    # Animate
    anim = @animate for (i, t) in enumerate(0:0.05:t_max)
        @info "Animating frame $i..."
        u_real = reshape([analytic_sol_func(t, x, y) for x in xs for y in ys],
            (length(xs), length(ys)))
        u_predict = reshape([Array(phi([t, x, y], res.u))[1] for x in xs for y in ys],
            length(xs), length(ys))
        u_error = abs.(u_predict .- u_real)
        title = @sprintf("predict, t = %.3f", t)
        p1 = plot(xs, ys, u_predict, st = :surface, label = "", title = title)
        title = @sprintf("real")
        p2 = plot(xs, ys, u_real, st = :surface, label = "", title = title)
        title = @sprintf("error")
        p3 = plot(xs, ys, u_error, st = :contourf, label = "", title = title)
        plot(p1, p2, p3)
    end
    gif(anim, "3pde.gif", fps = 10)
end

plot_(res)
```

![3pde](https://user-images.githubusercontent.com/12683885/129949743-9471d230-c14f-4105-945f-6bc52677d40e.gif)
