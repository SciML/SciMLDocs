# [Find the root of an equation (i.e. solve f(u)=0)](@id find_root)

A nonlinear system $$f(u) = 0$$ is specified by defining a function `f(u,p)`,
where `p` are the parameters of the system. Many problems can be written in
such a way that solving a nonlinear rootfinding problem gives the solution.
For example:

  - Do you want to know ``u`` such that ``4^u + 6^u = 7^u``? Then solve
    ``f(u) = 4^u + 6^u - 7^u = 0`` for `u`!
  - If you have an ODE ``u' = f(u)``, what is the point where the solution
    will be completely still, i.e. `u' = 0`?

All of these problems are solved by using a numerical rootfinder. Let's solve
our first rootfind problem!

## Required Dependencies

The following parts of the SciML Ecosystem will be used in this tutorial:

| Module                                                              | Description                                   |
|:------------------------------------------------------------------- |:--------------------------------------------- |
| [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) | The symbolic modeling environment             |
| [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/)   | The numerical solvers for nonlinear equations |

## Problem Setup

For example, the following solves the vector equation:

```math
\begin{aligned}
0 &= σ*(y-x)\\
0 &= x*(ρ-z)-y\\
0 &= x*y - β*z\\
\end{aligned}
```

With the parameter values ``\sigma = 10.0``, ``\rho = 26.0``, ``\beta = 8/3``.

```@example
# Import the packages
import ModelingToolkit as MTK
import NonlinearSolve as NLS
import ModelingToolkit: @variables, @parameters, @mtkbuild

# Define the nonlinear system
@variables x=1.0 y=0.0 z=0.0
@parameters σ=10.0 ρ=26.0 β=8 / 3

eqs = [0 ~ σ * (y - x),
    0 ~ x * (ρ - z) - y,
    0 ~ x * y - β * z]
@mtkbuild ns = MTK.NonlinearSystem(eqs, [x, y, z], [σ, ρ, β])

# Convert the symbolic system into a numerical system
prob = NLS.NonlinearProblem(ns, [])

# Solve the numerical problem
sol = NLS.solve(prob, NLS.NewtonRaphson())

# Analyze the solution
@show sol[[x, y, z]], sol.resid
```

## Step-by-Step Solution

### Step 1: Import the Packages

To do this tutorial, we will need a few components:

  - [ModelingToolkit.jl, our modeling environment](https://docs.sciml.ai/ModelingToolkit/stable/)
  - [NonlinearSolve.jl, the nonlinear system solvers](https://docs.sciml.ai/NonlinearSolve/stable/)

To start, let's add these packages [as demonstrated in the installation tutorial](@ref installation):

```julia
import Pkg
Pkg.add(["ModelingToolkit", "NonlinearSolve"])
```

Now we're ready. Let's load in these packages:

```@example first_rootfind
# Import the packages
import ModelingToolkit as MTK
import NonlinearSolve as NLS
import ModelingToolkit: @variables, @parameters, @mtkbuild
```

### Step 2: Define the Nonlinear System

Now let's define our nonlinear system. We use the `ModelingToolkit.@variabes` statement to
declare our 3 state variables:

```@example first_rootfind
# Define the nonlinear system
@variables x=1.0 y=0.0 z=0.0
```

Notice that we are using the form `state = initial condition`. This is a nice shorthand
for coupling an initial condition to our states. We now must similarly define our parameters,
which we can associate default values via the form `parameter = default value`. This looks
like:

```@example first_rootfind
@parameters σ=10.0 ρ=26.0 β=8 / 3
```

Now we create an array of equations to define our nonlinear system that must be satisfied.
This looks as follows:

!!! note
    
    Note that in ModelingToolkit and Symbolics, `~` is used for equation equality. This is
    separate from `=` which is the “assignment operator” in the Julia programming language.
    For example, `x = x + 1` is a valid assignment in a programming language, and it is
    invalid for that to represent “equality”, which is why a separate operator
    is used!

```@example first_rootfind
eqs = [0 ~ σ * (y - x),
    0 ~ x * (ρ - z) - y,
    0 ~ x * y - β * z]
```

Finally, we bring these pieces together, the equation along with its states and parameters,
define our `NonlinearSystem`:

```@example first_rootfind
@mtkbuild ns = MTK.NonlinearSystem(eqs, [x, y, z], [σ, ρ, β])
```

### Step 3: Convert the Symbolic Problem to a Numerical Problem

Now that we have created our system, let's turn it into a numerical problem to
approximate. This is done with the `NonlinearProblem` constructor, that transforms it from
a symbolic `ModelingToolkit` representation to a numerical `NonlinearSolve`
representation. We need to tell it the numerical details for whether to override any of the
default values for the initial conditions and parameters.

In this case, we will use the default values for all our variables, so we will pass a
blank override `[]`. This looks like:

```@example first_rootfind
# Convert the symbolic system into a numerical system
prob = NLS.NonlinearProblem(ns, [])
```

If we did want to change the initial condition of `x`
to `2.0` and the parameter `σ` to `4.0`, we would do `[x => 2.0, σ => 4.0]`. This looks
like:

```@example first_rootfind
prob2 = NLS.NonlinearProblem(ns, [x => 2.0, σ => 4.0])
```

### Step 4: Solve the Numerical Problem

Now we solve the nonlinear system. For this, we choose a solver from the
[NonlinearSolve.jl's solver options.](https://docs.sciml.ai/NonlinearSolve/stable/solvers/nonlinear_system_solvers/)
We will choose `NewtonRaphson` as follows:

```@example first_rootfind
# Solve the numerical problem
sol = NLS.solve(prob, NLS.NewtonRaphson())
```

### Step 5: Analyze the Solution

Now let's check out the solution. First of all, what kind of thing is the `sol`? We can
see that by asking for its type:

```@example first_rootfind
typeof(sol)
```

From this, we can see that it is an `NonlinearSolution`. We can see the documentation for
how to use the `NonlinearSolution` by checking the
[NonlinearSolve.jl solution type page.](https://docs.sciml.ai/NonlinearSolve/stable/basics/nonlinear_solution/)
For example, the solution is stored as `.u`.
What is the solution to our nonlinear system, and what is the final residual value?
We can check it as follows:

```@example first_rootfind
# Analyze the solution
@show sol[[x, y, z]], sol.resid
```
