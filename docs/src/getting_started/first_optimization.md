# [Solve your first optimization problem](@id first_opt)

**Numerical optimization** is the process of finding some numerical values that
minimize some equation.

  - How much fuel should you put into an airplane to have the minimum weight that
    can go to its destination?
  - What parameters should I choose for my simulation so that it minimizes the
    distance of its predictions from my experimental data?

All of these are examples of problems solved by numerical optimization.
Let's solve our first optimization problem!

## Required Dependencies

The following parts of the SciML Ecosystem will be used in this tutorial:

| Module                                                                                         | Description                        |
|:---------------------------------------------------------------------------------------------- |:---------------------------------- |
| [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)                                  | The numerical optimization package |
| [OptimizationNLopt.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/nlopt/) | The NLopt optimizers we will use   |
| [ForwardDiff.jl](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#Optimization.AutoForwardDiff) | The automatic differentiation library for gradients  |

## Problem Setup

First, what are we solving? Let's take a look at the Rosenbrock equation:

```math
L(u,p) = (p_1 - u_1)^2 + p_2 * (u_2 - u_1^2)^2
```

What we want to do is find the  values of ``u_1`` and ``u_2`` such that ``L``
achieves its minimum value possible. We will do this under a few constraints:
we want to find this optimum within some bounded domain, i.e. ``u_i \in [-1,1]``.
This should be done with the parameter values ``p_1 = 1.0`` and ``p_2 = 100.0``.
What should ``u = [u_1,u_2]`` be to achieve this goal? Let's dive in!

!!! note
    
    The upper and lower bounds are optional for the solver! If your problem does not
    need to have such bounds, just leave off the parts with `lb` and `ub`!

## Copy-Pastable Code

```@example
# Import the package
using Optimization, OptimizationNLopt, ForwardDiff

# Define the problem to optimize
L(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
u0 = zeros(2)
p = [1.0, 100.0]
optfun = OptimizationFunction(L, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optfun, u0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])

# Solve the optimization problem
sol = solve(prob, NLopt.LD_LBFGS())

# Analyze the solution
@show sol.u, L(sol.u, p)
```

## Step-by-Step Solution

### Step 1: Import the packages

To do this tutorial, we will need a few components:

  - [Optimization.jl](https://docs.sciml.ai/Optimization/stable/), the optimization interface.
  - [OptimizationNLopt.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/nlopt/), the optimizers we will use.
  - [ForwardDiff.jl](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#Optimization.AutoForwardDiff), 
    the automatic differentiation library for gradients
    
Note that Optimization.jl is an interface for optimizers, and thus we always have to choose
which optimizer we want to use. Here we choose to demonstrate `OptimizationNLopt` because
of its efficiency and versatility. But there are many other possible choices. Check out
the
[solver compatibility chart](https://docs.sciml.ai/Optimization/stable/#Overview-of-the-Optimizers)
for a quick overview of what optimizer packages offer.

To start, let's add these packages [as demonstrated in the installation tutorial](@ref installation):

```julia
using Pkg
Pkg.add(["Optimization", "OptimizationNLopt", "ForwardDiff"])
```

Now we're ready. Let's load in these packages:

```@example first_opt
using Optimization, OptimizationNLopt, ForwardDiff
```

### Step 2: Define the Optimization Problem

Now let's define our problem to optimize. We start by defining our loss function. In
Optimization.jl's `OptimizationProblem` interface, the states are given by an array
`u`. Thus we can designate `u[1]` to be `u_1` and `u[2]` to be `u_2`, similarly with our
parameters, and write out the loss function on a vector-defined state as follows:

```@example first_opt
# Define the problem to optimize
L(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
```
Next we need to create an `OptimizationFunction` where we tell Optimization.jl to use the ForwardDiff.jl
package for creating the gradient and other derivatives required by the optimizer.

```@example first_opt
#Create the OptimizationFunction
optfun = OptimizationFunction(L, Optimization.AutoForwardDiff())
```

Now we need to define our `OptimizationProblem`. If you need help remembering how to define
the `OptimizationProblem`, you can always refer to the
[Optimization.jl problem definition page](https://docs.sciml.ai/Optimization/stable/API/optimization_problem/).

Thus what we need to define is an initial condition `u0` and our parameter vector `p`.
We will make our initial condition have both values as zero, which is done by the Julia
shorthand `zeros(2)` that creates a vector `[0.0,0.0]`. We manually define the parameter
vector `p` to input our values. Then we set the lower bound and upper bound for the
optimization as follows:

```@example first_opt
u0 = zeros(2)
p = [1.0, 100.0]
prob = OptimizationProblem(optfun, u0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
```

#### Note about defining uniform bounds

Note that we can simplify the code a bit for the lower and upper bound definition by
using the Julia Base command `ones`, which returns a vector where each value is a one.
Thus for example, `ones(2)` is equivalent to `[1.0,1.0]`. Therefore `-1 * ones(2)` is
equivalent to `[-1.0,-1.0]`, meaning we could have written our problem as follows:

```@example first_opt
prob = OptimizationProblem(optfun, u0, p, lb = -1 * ones(2), ub = ones(2))
```

### Step 3: Solve the Optimization Problem

Now we solve the `OptimizationProblem` that we have defined. This is done by passing
our `OptimizationProblem` along with a chosen solver to the `solve` command. At
the beginning, we explained that we will use the `OptimizationNLopt` set of solvers, which
are
[documented in the OptimizationNLopt page](https://docs.sciml.ai/Optimization/stable/optimization_packages/nlopt/).
From here, we are choosing the `NLopt.LD_LBFGS()` for its mixture of robustness and
performance. To perform this solve, we do the following:

```@example first_opt
# Solve the optimization problem
sol = solve(prob, NLopt.LD_LBFGS())
```

### Step 4: Analyze the Solution

Now let's check out the solution. First of all, what kind of thing is the `sol`? We can
see that by asking for its type:

```@example first_opt
typeof(sol)
```

From this, we can see that it is an `OptimizationSolution`. We can see the documentation for
how to use the `OptimizationSolution` by checking the
[Optimization.jl solution type page](https://docs.sciml.ai/Optimization/stable/API/optimization_solution/).
For example, the solution is stored as `.u`. What is the solution to our
optimization, and what is the final loss value? We can check it as follows:

```@example first_opt
# Analyze the solution
@show sol.u, L(sol.u, p)
```
