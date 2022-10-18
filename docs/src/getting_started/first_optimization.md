# [Solve your first optimization problem](@id first_opt)

**Numerical optimization** is the process of finding some numerical values that
minimize some equation. 

* How much fuel should you put into an airplane to have the minimum weight that 
  can go to its destination?
* What parameters should I choose for my simulation so that it minimizes the
  distance of its predictions from my experimental data?
* 

All of these are examples of problems solved by numerical optimization. 
Let's solve our first optimization problem!

## Problem Setup

First, what are we solving? Let's take a look at the Rosenbrock equation:

```math
L(u,p) = (p_1 - u_1)^2 + p_2 * (u_2 - u_1)^2
```

What we want to do is find the  values of ``u_1`` and ``u_2`` such that ``L`` 
achieves its minimum value possible. We will do this under a few constraints: 
we want to find this optima within some bounded domain, i.e. ``u_i \in [-1,1]``. 
This should be done with the parameter values ``p_1 = 1.0`` and `p_2 = 100.0``. 
What should ``u = [u_1,u_2]`` be to achieve this goal? Let's dive in!

!!! note

    The upper and lower bounds are optional for the solver! If your problem does not
    need to have such bounds, just leave off the parts with `lb` and `ub`!

## Copy-Pastable Code

```@example
# Import the package 
using Optimization, OptimizationNLopt

# Define the problem to optimize
L(u,p) =  (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
u0 = zeros(2)
p  = [1.0,100.0]
prob = OptimizationProblem(L, u0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])

# Solve the optimization problem
sol = solve(prob,LD_LBFGS())

# Analyze the solution
@show sol.u, L(sol.u,p)
```

## Step 1: Import the packages

```@example first_opt 
using Optimization, OptimizationNLopt
```

## Step 2: Define the Optimization Problem

```@example first_opt
# Define the problem to optimize
L(u,p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
```

```@example first_opt
u0 = zeros(2)
p  = [1.0,100.0]
prob = OptimizationProblem(rosenbrock, u0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
```

## Step 3: Solve the Optimization Problem

```@example first_opt
# Solve the optimization problem
sol = solve(prob,LN_BFGS())
```

## Step 4: Analyze the Solution

```@example first_opt
# Analyze the solution
@show sol.u, L(sol.u,p)
```