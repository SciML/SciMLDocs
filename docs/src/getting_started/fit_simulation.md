# [Fit a simulation to a dataset](@id fit_simulation)

Running simulations is only half of the battle. Many times, in order to make the simulation
realistic, you need to fit the simulation to data. The SciML ecosystem has **integration with
automatic differentiation and adjoint methods** to automatically make the fitting process
stable and efficient. Let's see this in action.

## Required Dependencies

The following parts of the SciML Ecosystem will be used in this tutorial:

| Module                                                                                                           | Description                                               |
|:---------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------- |
| [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)                                             | The differential equation solvers                         |
| [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)                                                    | The numerical optimization package                        |
| [OptimizationPolyalgorithms.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/polyalgorithms/) | The optimizers we will use                                |
| [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/dev/)                                               | The connection of the SciML ecosystems to differentiation |

Along with the following general ecosystem packages:

| Module                                                         | Description                            |
|:-------------------------------------------------------------- |:-------------------------------------- |
| [Plots.jl](https://docs.juliaplots.org/stable/)                | The plotting and visualization package |
| [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/) | The automatic differentiation package  |

## Problem Setup: Fitting Lotka-Volterra Data

Assume that we know that the dynamics of our system are given by the
[Lotka-Volterra dynamical system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations):
Let $x(t)$ be the number of rabbits in the environment and $y(t)$ be the number of wolves.
This is the same dynamical system as [the first tutorial!](@ref first_sim)
The equation that defines the evolution of the species is given as follows:

```math
\begin{align}
\frac{dx}{dt} &= \alpha x - \beta x y\\
\frac{dy}{dt} &= -\gamma y + \delta x y
\end{align}
```

where ``\alpha, \beta, \gamma, \delta`` are parameters. Starting from equal numbers of
rabbits and wolves, ``x(0) = 1`` and ``y(0) = 1``.

Now, in [the first tutorial](@ref first_sim), we assumed:

> Luckily, a local guide provided our with some parameters that seem to match the system!

Sadly, magical nymphs do not always show up and give us parameters. Thus in this case,
let's assume that we are just given data that is representative of the solution with
``\alpha = 1.5``, ``\beta = 1.0``, ``\gamma = 3.0``, and ``\delta = 1.0``. This data
is given over a time span of ``t_0 = 0`` to ``t_f = 10`` with data taken on both rabbits
and wolves at every ``\Delta t = 1.`` Can we figure out what the parameter values should
be directly from the data?

## Solution as Copy-Pastable Code

```@example
using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using ForwardDiff, Plots

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval
tspan = (0.0, 10.0)

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
datasol = solve(prob, saveat = 1)
data = Array(datasol)

## Now do the optimization process
function loss(newp)
    newprob = remake(prob, p = newp)
    sol = solve(newprob, saveat = 1)
    loss = sum(abs2, sol .- data)
    return loss, sol
end

callback = function (p, l, sol)
    display(l)
    plt = plot(sol, ylim = (0, 6), label = "Current Prediction")
    scatter!(plt, datasol, label = "Data")
    display(plt)
    # Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

adtype = Optimization.AutoForwardDiff()
pguess = [1.0, 1.2, 2.5, 1.2]
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pguess)

result_ode = Optimization.solve(optprob, PolyOpt(),
                                callback = callback,
                                maxiters = 200)
```

## Step-by-Step Solution

### Step 1: Install and Import the Required Packages

To do this tutorial, we will need a few components. This is done using the Julia Pkg REPL:

```julia
using Pkg
Pkg.add([
            "DifferentialEquations",
            "Optimization",
            "OptimizationPolyalgorithms",
            "SciMLSensitivity",
            "ForwardDiff",
            "Plots",
        ])
```

Now we're ready. Let's load in these packages:

```@example odefit
using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using ForwardDiff, Plots
```

### Step 2: Generate the Training Data

In our example, we assumed that we had data representative of the solution with
``\alpha = 1.5``, ``\beta = 1.0``, ``\gamma = 3.0``, and ``\delta = 1.0``. Let's make that
training data. The way we can do that is by defining the ODE with those parameters and
simulating it. Unlike [the first tutorial](@ref first_sim), which used ModelingToolkit,
let's demonstrate using [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
to directly define the ODE for the numerical solvers.

To do this, we define a vector-based mutating function that calculates the derivatives for
our system. We will define our system as a vector `u = [x,y]`, and thus `u[1] = x` and
`u[2] = y`. This means that we need to calculate the derivative as `du = [dx,dy]`. Our parameters
will simply be the vector `p = [α, β, δ, γ]`. Writing down the Lotka-Volterra equations in the
DifferentialEquations.jl direct form thus looks like the following:

```@example odefit
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
end
```

Now we need to define the initial condition, time span, and parameter vector to simulate with.
Following the problem setup, this looks like:

```@example odefit
# Initial condition
u0 = [1.0, 1.0]

# Simulation interval
tspan = (0.0, 10.0)

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]
```

Now we bring these pieces all together to define the `ODEProblem` and solve it. Note that we solve
this equation with the keyword argument `saveat = 1` so that it saves a point at every ``\Delta t = 1``.

```@example odefit
# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
datasol = solve(prob, saveat = 1)
```

```@example odefit
data = Array(datasol)
```

!!! note
    

For more details on using DifferentialEquations.jl, check out the
[getting started with DifferentialEquations.jl tutorial](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/).

### Step 3: Set Up the Cost Function for Optimization

Now let's start the estimation process. First, let's define a loss function. For our loss function, we want to
take a set of parameters, create a new ODE which has everything the same except for the changed parameters,
solve this ODE with new parameters, and compare its predictions against the data. For this parameter changing,
there is a useful functionality in the
[SciML problems interface](https://docs.sciml.ai/DiffEqDocs/stable/basics/problem/#Modification-of-problem-types)
called `remake` which creates a new version of an existing `SciMLProblem` with the aspect you want changed.
For example, if we wanted to change the initial condition `u0` of our ODE, we could do `remake(prob, u0 = newu0)`
For our case, we want to change around just the parameters, so we can do `remake(prob, p = newp)`

!!! note
    
    `remake` can change multiple items at once by passing more keyword arguments! I.e., `remake(prob, u0 = newu0, p = newp)`
    This can be used to extend the example to simultaneously learn the initial conditions and parameters!

Now use `remake` to build the cost function. After we solve the new problem, we will calculate the sum of squared errors
as our metric. The sum of squares can be quickly written in Julia via `sum(abs2,x)`. Using this information, our loss
looks like:

```@example odefit
function loss(newp)
    newprob = remake(prob, p = newp)
    sol = solve(newprob, saveat = 1)
    loss = sum(abs2, sol .- data)
    return loss, sol
end
```

Notice that our loss function returns the loss value as the first return, but returns extra information (the solution at the
new parameters) as extra return arguments. We will explain why this extra return information is helpful in the next section.

### Step 4: Solve the Optimization Problem

This step will look very similar to [the first optimization tutorial](@ref first_opt), except now we have a new
cost function. Just like in that tutorial, we want to define a callback to monitor the solution process. However,
this time, our function returns two things. The callback syntax is always `(value being optimized, arguments of loss return)`
and thus this time the callback is given `(p, l, sol)`. See, returning the solution along with the loss as part of the
loss function is useful because we have access to it in the callback to do things like plot the current solution
against the data! Let's do that in the following way:

```@example odefit
callback = function (p, l, sol)
    display(l)
    plt = plot(sol, ylim = (0, 6), label = "Current Prediction")
    scatter!(plt, datasol, label = "Data")
    display(plt)
    # Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
    return false
end
```

Thus every step of the optimization will show us the loss and a plot of how the solution
looks vs the data at our current parameters.

Now, just like [the first optimization tutorial](@ref first_opt), we setup our optimization
problem. To do this, we need to come up with a `pguess`, an initial condition for the parameters
which is our best guess of the true parameters. For this, we will use `pguess = [1.0, 1.2, 2.5, 1.2]`.
Together, this looks like:

```@example odefit
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
pguess = [1.0, 1.2, 2.5, 1.2]
optprob = Optimization.OptimizationProblem(optf, pguess)
```

Now we solve the optimization problem:

```@example odefit
result_ode = Optimization.solve(optprob, PolyOpt(),
                                callback = callback,
                                maxiters = 200)
```

and the answer from the optimization is our desired parameters.
