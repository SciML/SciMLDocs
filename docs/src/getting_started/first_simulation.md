# [Build and run your first simulation with Julia's SciML](@id first_sim)

In this tutorial we will build and run our first simulation with SciML!

!!! note

    This tutorial assumes that you have already installed Julia on your system. If you have
    not done so already, please [follow the installation tutorial first](@ref installation).

To build our simulation, we will use the [ModelingToolkit](https://mtk.sciml.ai/dev/)
system for modeling and simulation. ModelingToolkit is a bit higher level than directly
defining code for a differential equation system: it's a symbolic system that will
automatically simplify our models, optimize our code, and generate compelling visualizations.
Sounds neat? Let's dig in.

## Our Problem: Simulate the Lotka-Volterra Predator-Prey Dynamics

The dynamics of our system are given by the
[Lotka-Volterra dynamical system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations):
Let $x(t)$ be the number of rabbits in the environment and $y(t)$ be the number of wolves.
The equation that defines the evolution of the species is given as follows:

```math
\begin{align}
\frac{dx}{dt} &= \alpha x - \beta x y
\frac{dy}{dt} &= -\gamma y + \delta x y
\end{align}
```

where ``\alpha, \beta, \gamma, \delta`` are parameters. Starting from equal numbers of
rabbits and wolves, ``x(0) = 1`` and ``y(0) = 1``, we want to simulate this system from time
``t_0 = 0`` to ``t_f = 10``. Luckily, a local guide provided our with some parameters that
seem to match the system! These are ``\alpha = 1.5``, ``\beta = 1.0``, ``\gamma = 3.0``,
``\delta = 1.0``. How many rabbits and wolves will there be 10 months from now? And if
`z = x + y`, i.e. the total number of animals at a given time, can we visualize this total
number of animals at each time?

## Solution as Copy-Pastable Code

```@examples
using ModelingToolkit, DifferentialEquations, Plots

# Define our state variables: state(t) = initial condition
@variables t x(t)=1 y(t)=1 z(t)=2

# Define our parameters
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0

# Define our differential: takes the derivative with respect to `t`
D = Differential(t)

# Define the differential equations
eqs = [
    D(x) ~ α*x - β*x*y
    D(y) ~ -γ*y + δ*x*y
    z ~ x + y
]

# Bring these pieces together into an ODESystem with independent variable t
@named sys = ODESystem(eqs,t)

# Symbolically Simplify the System
simpsys = structural_simplify(sys)

# Convert from a symbolic to a numerical problem to simulate
tspan = (0.0,10.0)
prob = ODEProblem(simpsys, [], tspan)

# Solve the ODE
sol = solve(prob)

# Plot the solution
p1 = plot(sol,title = "Rabbits vs Wolves")
p2 = plot(sol,idxs=z,title = "Total Animals")

plot(p1,p2,layout=(2,1))
```

## Step by Step Solution

### Step 1: Install and Import the Required Packages

To do this tutorial we will need a few components:

* [ModelingToolkit.jl, our modeling environment](https://mtk.sciml.ai/dev/)
* [DifferentialEquations.jl, the differential equation solvers](https://diffeq.sciml.ai/stable/)
* [Plots.jl, our visualization tool](https://docs.juliaplots.org/stable/)

To start, let's add these packages [as demonstrated in the installation tutorial](@ref installation):

```julia
]add ModelingToolkit DifferentialEquations Plots
```

Now we're ready. Let's load in these packages:

```julia
using ModelingToolkit, DifferentialEquations, Plots
```

### Step 2: Define our ODE System

```julia
# Define our state variables: state(t) = initial condition
@variables t x(t)=1 y(t)=1 z(t)=2

# Define our parameters
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0

# Define our differential: takes the derivative with respect to `t`
D = Differential(t)

# Define the differential equations
eqs = [
    D(x) ~ α*x - β*x*y
    D(y) ~ -γ*y + δ*x*y
    z ~ x + y
]

# Bring these pieces together into an ODESystem with independent variable t
@named sys = ODESystem(eqs,t)

# Symbolically Simplify the System
simpsys = structural_simplify(sys)

# Convert from a symbolic to a numerical problem to simulate
tspan = (0.0,10.0)
prob = ODEProblem(simpsys, [], tspan)
```

### Step 3: Solve the ODE System

```julia
# Solve the ODE
sol = solve(prob)
```

### Step 4: Visualize the Solution

```julia
# Plot the solution
p1 = plot(sol,title = "Rabbits vs Wolves")
p2 = plot(sol,idxs=z,title = "Total Animals")

plot(p1,p2,layout=(2,1))
```
