# [Build and run your first simulation with Julia's SciML](@id first_sim)

In this tutorial, we will build and run our first simulation with SciML!

!!! note
    
    This tutorial assumes that you have already installed Julia on your system. If you have
    not done so already, please [follow the installation tutorial first](@ref installation).

To build our simulation, we will use the
[ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/)
system for modeling and simulation. ModelingToolkit is a bit higher level than directly
defining code for a differential equation system: it's a symbolic system that will
automatically simplify our models, optimize our code, and generate compelling visualizations.
Sounds neat? Let's dig in.

## Required Dependencies

The following parts of the SciML Ecosystem will be used in this tutorial:

| Module                                                               | Description                            |
|:-------------------------------------------------------------------- |:-------------------------------------- |
| [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)  | The symbolic modeling environment      |
| [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) | The differential equation solvers      |
| [Plots.jl](https://docs.juliaplots.org/stable/)                      | The plotting and visualization package |

## Our Problem: Simulate the Lotka-Volterra Predator-Prey Dynamics

The dynamics of our system are given by the
[Lotka-Volterra dynamical system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations):
Let $x(t)$ be the number of rabbits in the environment and $y(t)$ be the number of wolves.
The equation that defines the evolution of the species is given as follows:

```math
\begin{align}
\frac{dx}{dt} &= \alpha x - \beta x y\\
\frac{dy}{dt} &= -\gamma y + \delta x y
\end{align}
```

where ``\alpha, \beta, \gamma, \delta`` are parameters. Starting from equal numbers of
rabbits and wolves, ``x(0) = 1`` and ``y(0) = 1``, we want to simulate this system from time
``t_0 = 0`` to ``t_f = 10``. Luckily, a local guide provided us with some parameters that
seem to match the system! These are ``\alpha = 1.5``, ``\beta = 1.0``, ``\gamma = 3.0``,
``\delta = 1.0``. How many rabbits and wolves will there be 10 months from now? And if
`z = x + y`, i.e. the total number of animals at a given time, can we visualize this total
number of animals at each time?

## Solution as Copy-Pastable Code

```@example
import DifferentialEquations as DE
import ModelingToolkit as MTK
import Plots
import ModelingToolkit: t_nounits as t, D_nounits as D,
                        @variables, @parameters, @named, @mtkbuild

# Define our state variables: state(t) = initial condition
@variables x(t)=1 y(t)=1 z(t)

# Define our parameters
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0

# Define the differential equations
eqs = [D(x) ~ α * x - β * x * y
       D(y) ~ -γ * y + δ * x * y
       z ~ x + y]

# Bring these pieces together into an ODESystem with independent variable t
@mtkbuild sys = MTK.ODESystem(eqs, t)

# Convert from a symbolic to a numerical problem to simulate
tspan = (0.0, 10.0)
prob = DE.ODEProblem(sys, [], tspan)

# Solve the ODE
sol = DE.solve(prob)

# Plot the solution
p1 = Plots.plot(sol, title = "Rabbits vs Wolves")
p2 = Plots.plot(sol, idxs = z, title = "Total Animals")

Plots.plot(p1, p2, layout = (2, 1))
```

## Step-by-Step Solution

### Step 1: Install and Import the Required Packages

To do this tutorial, we will need a few components:

  - [ModelingToolkit.jl, our modeling environment](https://docs.sciml.ai/ModelingToolkit/stable/)
  - [DifferentialEquations.jl, the differential equation solvers](https://docs.sciml.ai/DiffEqDocs/stable/)
  - [Plots.jl, our visualization tool](https://docs.juliaplots.org/stable/)

To start, let's add these packages [as demonstrated in the installation tutorial](@ref installation):

```julia
using Pkg
Pkg.add(["ModelingToolkit", "DifferentialEquations", "Plots"])
```

Now we're ready. Let's load in these packages:

```@example first_sim
import DifferentialEquations as DE
import ModelingToolkit as MTK
import Plots
import ModelingToolkit: t_nounits as t, D_nounits as D, @variables, @parameters, @named, @mtkbuild
```

### Step 2: Define our ODE Equations

Now let's define our ODEs. We use the `ModelingToolkit.@variabes` statement to declare
our variables. We have the independent variable time `t`, and then define our 3 state
variables:

```@example first_sim
# Define our state variables: state(t) = initial condition
@variables x(t)=1 y(t)=1 z(t)
```

Notice here that we use the form `state = default`, where on the right-hand side the default
value of a state is interpreted to be its initial condition. Note that since `z` will be given
by an algebraic equation, we do not need to specify its initial condition.

This is then done similarly for parameters, where the default value is now the parameter value:

```@example first_sim
# Define our parameters
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0
```

!!! note
    
    Julia's text editors like VS Code are compatible with Unicode defined in a LaTeX form.
    Thus if you write `\alpha` into your REPL and then press `Tab`, it will auto-complete
    that into the α symbol. That can make your code look a lot more like the mathematical
    expressions!

Next, we define our set of differential equations.

!!! note
    
    Note that in ModelingToolkit and Symbolics, `~` is used for equation equality. This is
    separate from `=` which is the “assignment operator” in the Julia programming language.
    For example, `x = x + 1` is a valid assignment in a programming language, and it is
    invalid for that to represent “equality”, which is why a separate operator
    is used!

```@example first_sim
# Define the differential equations
eqs = [D(x) ~ α * x - β * x * y
       D(y) ~ -γ * y + δ * x * y
       z ~ x + y]
```

Notice that in the display, it will automatically generate LaTeX. If one is interested in
generating this LaTeX locally, one can simply do:

```julia
using Latexify # add the package first
latexify(eqs)
```

## Step 3: Define the ODEProblem

Now we bring these pieces together. In ModelingToolkit, we can bring these pieces together
to represent an `ODESystem` with the following:

```@example first_sim
# Bring these pieces together into an ODESystem with independent variable t
@mtkbuild sys = MTK.ODESystem(eqs, t)
```

Notice that in our equations we have an algebraic equation `z ~ x + y`. This is not a
differential equation but an algebraic equation, and thus we call this set of equations a
Differential-Algebraic Equation (DAE). The symbolic system of ModelingToolkit can eliminate
such equations to return simpler forms to numerically approximate.

Notice that what is returned is an `ODESystem`, but now with the simplified set of
equations. `z` has been turned into an “observable”, i.e. a state that is not computed
but can be constructed on-demand. This is one of the ways that SciML reaches its speed:
you can have 100,000 equations, but solve only 1,000 to then automatically reconstruct
the full set. Here, it's just 3 equations to 2, but as models get more complex, the
symbolic system will find ever more clever interactions!

Now that we have simplified our system, let's turn it into a numerical problem to
approximate. This is done with the `ODEProblem` constructor, that transforms it from
a symbolic `ModelingToolkit` representation to a numerical `DifferentialEquations`
representation. We need to tell it the numerical details now:

 1. Whether to override any of the default values for the initial conditions and parameters.
 2. What is the initial time point.
 3. How long to integrate it for.

In this case, we will use the default values for all our variables, so we will pass a
blank override `[]`. If for example we did want to change the initial condition of `x`
to `2.0` and `α` to `4.0`, we would do `[x => 2.0, α => 4.0]`. Then secondly, we pass a
tuple for the time span, `(0.0,10.0)` meaning start at `0.0` and end at `10.0`. This looks
like:

```@example first_sim
# Convert from a symbolic to a numerical problem to simulate
tspan = (0.0, 10.0)
prob = DE.ODEProblem(sys, [], tspan)
```

### Step 4: Solve the ODE System

Now we solve the ODE system. Julia's SciML solvers have a defaulting system that can
automatically determine an appropriate solver for a given system, so we can just tell it
to solve:

```@example first_sim
# Solve the ODE
sol = DE.solve(prob)
```

### Step 5: Visualize the Solution

Now let's visualize the solution! Notice that our solution only has two states. If we
recall, the simplified system only has two states: `z` was symbolically eliminated. We
can access any of the values, even the eliminated values, using the symbolic variable
as the index. For example:

```@example first_sim
sol[z]
```

returns the time series of the observable `z` at time points corresponding to `sol.t`.
We can use this with the automated plotting functionality. First let's create a plot of
`x` and `y` over time using `plot(sol)` which will plot all of the states. Then next,
we will explicitly tell it to make a plot with the index being `z`, i.e. `idxs=z`.

!!! note
    
    Note that one can pass an array of indices as well, so `idxs=[x,y,z]` would make a plot
    with all three lines together!

```@example first_sim
# Plot the solution
p1 = Plots.plot(sol, title = "Rabbits vs Wolves")
```

```@example first_sim
p2 = Plots.plot(sol, idxs = z, title = "Total Animals")
```

Finally, let's make a plot where we merge these two plot elements. To do so, we can take our
two plot objects, `p1` and `p2`, and make a plot with both of them. Then we tell Plots to
do a layout of `(2,1)`, or 2 rows and 1 columns. Let's see what happens when we bring these
together:

```@example first_sim
Plots.plot(p1, p2, layout = (2, 1))
```

And tada, we have a full analysis of our ecosystem!

## Bonus Step: Emoji Variables

If you made it this far, then congrats, you get to learn a fun fact! Since Julia code can
use Unicode, emojis work for variable names. Here's the simulation using emojis of rabbits
and wolves to define the system:

```@example first_sim
import DifferentialEquations as DE
import ModelingToolkit as MTK
import ModelingToolkit: t_nounits as t, D_nounits as D, @variables, @parameters, @named
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0
@variables t 🐰(t)=1.0 🐺(t)=1.0
D = MTK.Differential(t)
eqs = [D(🐰) ~ α * 🐰 - β * 🐰 * 🐺,
    D(🐺) ~ -γ * 🐺 + δ * 🐰 * 🐺]

@mtkbuild sys = MTK.ODESystem(eqs, t)
prob = DE.ODEProblem(sys, [], (0.0, 10.0))
sol = DE.solve(prob)
```

Now go make your professor mad that they have to grade a fully emojified code. I'll vouch
for you: the documentation told you to do this.
