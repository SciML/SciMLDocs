# [Find the root of an equation (i.e. solve f(x)=0)](@id find_root)

A nonlinear system $$f(u) = 0$$ is specified by defining a function `f(u,p)`,
where `p` are the parameters of the system. Many problems can be written in
such as way that solving a nonlinear rootfinding problem gives the solution.
For example: 

* Do you want to know ``u`` such that ``4^u + 6^u = 7^u``? Then solve 
  ``f(u) = 4^u + 6^u - 7^u = 0`` for `u`!
* If you have an ODE ``u' = f(u)``, what is the point where the solution
  will be completely still, i.e. `u' = 0`?
* 

All of these problems are solved by using a numerical rootfinder. Let's solve 
our first rootfind problem!

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
using ModelingToolkit, NonlinearSolve

# Define the nonlinear system
@variables x=1.0 y=0.0 z=0.0
@parameters σ=10.0 ρ=26.0 β=8/3

eqs = [0 ~ σ*(y-x),
       0 ~ x*(ρ-z)-y,
       0 ~ x*y - β*z]
@named ns = NonlinearSystem(eqs, [x,y,z], [σ,ρ,β])

# Convert the symbolic system into a numerical system
prob = NonlinearProblem(ns,[])

# Solve the numerical problem
sol = solve(prob,NewtonRaphson())

# Analyze the solution
@show sol.u, prob.f(sol.u,prob.p)
```

## Step 1: Import the Packages

```@example first_rootfind
# Import the packages
using ModelingToolkit, NonlinearSolve
```

## Step 2: Define the Nonlinear System

```@example first_rootfind
# Define the nonlinear system
@variables x=1.0 y=0.0 z=0.0
@parameters σ=10.0 ρ=26.0 β=8/3

eqs = [0 ~ σ*(y-x),
       0 ~ x*(ρ-z)-y,
       0 ~ x*y - β*z]
@named ns = NonlinearSystem(eqs, [x,y,z], [σ,ρ,β])
```

## Step 3: Convert the Symbolic Problem to a Numerical Problem

```@example first_rootfind
# Convert the symbolic system into a numerical system
prob = NonlinearProblem(ns,[])
```

## Step 4: Solve the Numerical Problem

```@example first_rootfind
# Solve the numerical problem
sol = solve(prob,NewtonRaphson())
```

## Step 5: Analyze the Solution

```@example first_rootfind
# Analyze the solution
@show sol.u, prob.f(sol.u,prob.p)
```