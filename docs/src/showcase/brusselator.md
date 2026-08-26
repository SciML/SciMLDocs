# [Automated Efficient Solution of Nonlinear Partial Differential Equations](@id brusselator)

Solving nonlinear partial differential equations (PDEs) is hard. Solving nonlinear PDEs fast
and accurately is even harder. Doing it all in an automated method from just a symbolic
description is just plain fun. That's what we'd demonstrate here: how to solve a nonlinear
PDE from a purely symbolic definition using the combination of ModelingToolkit,
MethodOfLines, and DifferentialEquations.jl.

!!! note
    
    This example is a combination of the
    [Brusselator tutorial from MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/tutorials/brusselator/)
    and the [Solving Large Stiff Equations tutorial from DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/).

## Required Dependencies

The following parts of the SciML Ecosystem will be used in this tutorial:

| Module                                                               | Description                                 |
|:-------------------------------------------------------------------- |:------------------------------------------- |
| [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)  | The symbolic modeling environment           |
| [MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/)      | The symbolic PDE discretization tooling     |
| [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) | The numerical differential equation solvers |

## Problem Setup

The Brusselator PDE is defined as follows:

```math
\begin{align}
\frac{\partial u}{\partial t} &= 1 + u^2v - 4.4u + \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) + f(x, y, t)\\
\frac{\partial v}{\partial t} &= 3.4u - u^2v + \alpha \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)
\end{align}
```

where

```math
f(x, y, t) = \begin{cases}
5 & \quad \text{if } (x-0.3)^2+(y-0.6)^2 ≤ 0.1^2 \text{ and } t ≥ 1.1 \\
0 & \quad \text{else}
\end{cases}
```

and the initial conditions are

```math
\begin{align}
u(x, y, 0) &= 22\cdot (y(1-y))^{3/2} \\
v(x, y, 0) &= 27\cdot (x(1-x))^{3/2}
\end{align}
```

with the periodic boundary condition

```math
\begin{align}
u(x+1,y,t) &= u(x,y,t) \\
u(x,y+1,t) &= u(x,y,t)
\end{align}
```

We wish to obtain the solution to this PDE on a timespan of ``t \in [0,11.5]``.

## Defining the symbolic PDEsystem with ModelingToolkit.jl

With `ModelingToolkit.jl`, we first symbolically define the system, see also the docs for [`PDESystem`](https://docs.sciml.ai/ModelingToolkit/stable/API/PDESystem/):

```@example bruss
import MethodOfLines
import OrdinaryDiffEq as ODE
import SciMLBase
import DomainSets: Interval
using ModelingToolkit: @named, @parameters, @variables, Differential, PDESystem

@parameters x y t
@variables u(..) v(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

∇²(u) = Dxx(u) + Dyy(u)

brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

α = 10.0

u0(x, y, t) = 22(y * (1 - y))^(3 / 2)
v0(x, y, t) = 27(x * (1 - x))^(3 / 2)

eq = [
    Dt(u(x, y, t)) ~ 1.0 + v(x, y, t) * u(x, y, t)^2 - 4.4 * u(x, y, t) +
        α * ∇²(u(x, y, t)) + brusselator_f(x, y, t),
    Dt(v(x, y, t)) ~ 3.4 * u(x, y, t) - v(x, y, t) * u(x, y, t)^2 + α * ∇²(v(x, y, t)),
]

domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    t ∈ Interval(t_min, t_max),
]

# Periodic BCs
bcs = [
    u(x, y, 0) ~ u0(x, y, 0),
    u(0, y, t) ~ u(1, y, t),
    u(x, 0, t) ~ u(x, 1, t), v(x, y, 0) ~ v0(x, y, 0),
    v(0, y, t) ~ v(1, y, t),
    v(x, 0, t) ~ v(x, 1, t),
]

@named pdesys = PDESystem(eq, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])
```

Looks just like the LaTeX description, right? Now let's solve it.

## Automated symbolic discretization with MethodOfLines.jl

Next we create the discretization. Here we will use the finite difference method via
method of lines. Method of lines is a method of recognizing that a discretization of a
partial differential equation transforms it into a new numerical problem. For example:

| Discretization Form                                                                                            | Numerical Problem Type                                  |
|:-------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------- |
| Finite Difference, Finite Volume, Finite Element, discretizing all variables                                   | `NonlinearProblem`                                      |
| MethodOfLines finite differences, discretizing all variables except time                                       | Array-form `DAEProblem`                                 |
| Physics-Informed Neural Network                                                                                | `OptimizationProblem`                                   |
| Feynman-Kac Formula                                                                                            | `SDEProblem`                                            |
| Universal Stochastic Differential Equation ([High dimensional PDEs](https://docs.sciml.ai/HighDimPDE/stable/)) | `OptimizationProblem` inverse problem over `SDEProblem` |

Thus the process of solving a PDE is fundamentally about transforming its symbolic form
to a standard numerical problem and solving the standard numerical problem using one of the
solvers in the SciML ecosystem! Here we will demonstrate one of the most classic methods:
the finite difference method. Since the Brusselator is a time-dependent PDE with heavy
stiffness in the time domain, we will leave time undiscretized. MethodOfLines discretizes
the `x` and `y` domains while representing operations over the complete arrays of grid
values. The result is a differential-algebraic equation (DAE) for how those values evolve
over time.

To do this, we use the `MOLFiniteDifference` construct of
[MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/) as follows:

```@example bruss
N = 32

dx = (x_max - x_min) / N
dy = (y_max - y_min) / N

order = 2

discretization = MethodOfLines.MOLFiniteDifference(
    [x => dx, y => dy], t, approx_order = order,
    grid_align = MethodOfLines.center_align
)
```

Next, we `discretize` the system. MethodOfLines v1 constructs an array-form `DAEProblem`
for supported time-dependent systems. The `fallback = false` keyword enforces that array
form so this tutorial fails if the PDE is not supported by it:

```@example bruss
prob = MethodOfLines.discretize(pdesys, discretization; fallback = false)
@assert prob isa SciMLBase.DAEProblem
```

The symbolic equations operate on whole array slices rather than generating one scalar
equation per grid point. For a fixed PDE system, this keeps symbolic compilation independent
of the spatial grid size. Discretization, storage, and numerical solution still scale with
the number of grid points.

## Solving the PDE

Calling `solve` without an explicit algorithm lets OrdinaryDiffEq select its default DAE
solver and preserves the array-form compilation path:

```@example bruss
sol = ODE.solve(prob; saveat = 0.1);
```

## Examining Results via the Symbolic Solution Interface

Now that we have solved the DAE representation of the PDE, we have a `PDETimeSeriesSolution`
that wraps the numerical DAE solution, which we can get with `sol.original_sol`. The raw
solver state stores the grid values in its numerical ordering, but that ordering is not the
most useful way to interpret a PDE solution.

To make the handling of such cases a lot simpler, MethodOfLines.jl implements a
symbolic interface for the solution object that allows for interpreting the computation
through its original representation. For example, if we want to know how to interpret
the values of the grid corresponding to the independent variables, we can just index using
symbolic variables:

```@example bruss
discrete_x = sol[x];
discrete_y = sol[y];
discrete_t = sol[t];
```

What this tells us is that, for a solution at a given time point, say `original_sol[1]` for the
solution at the initial time (the initial condition), the value `original_sol[1][1]` is the solution
at the grid point `(discrete_x[1], discrete_y[1])`. For values that are not the initial
time point, `original_sol[i]` corresponds to the solution at `discrete_t[i]`.

But we also have two dependent variables, `u` and `v`. How do we interpret which of the
results correspond to the different dependent variables? This is done by indexing the
solution by the dependent variables! For example:

```@example bruss
solu = sol[u(x, y, t)];
solv = sol[v(x, y, t)];
```

This then gives an array of results for the `u` and `v` separately, each dimension
corresponding to the discrete form of the independent variables.

Using this high-level indexing, we can create an animation of the solution of the
Brusselator as follows. For `u` we receive:

```julia
import Plots
anim = Plots.@animate for k in 1:length(discrete_t)
    Plots.heatmap(solu[2:end, 2:end, k], title = "$(discrete_t[k])") # 2:end since end = 1, periodic condition
end
Plots.gif(anim, "plots/Brusselator2Dsol_u.gif", fps = 8)
```

![Brusselator2Dsol_u](https://user-images.githubusercontent.com/9698054/159934498-e5c21b13-c63b-4cd2-9149-49e521765141.gif)

and for `v`:

```julia
anim = Plots.@animate for k in 1:length(discrete_t)
    Plots.heatmap(solv[2:end, 2:end, k], title = "$(discrete_t[k])")
end
Plots.gif(anim, "plots/Brusselator2Dsol_v.gif", fps = 8)
```

![Brusselator2Dsol_v](https://i.imgur.com/3kQNMI3.gif)

## Why Keep the Array-Form DAE?

MethodOfLines keeps the spatial operations as array-slice equations when it constructs the
`DAEProblem`. Code generation therefore follows the array representation instead of
generating a separate scalar expression for every grid point. Keep this array-form problem
and solve it with a [DAE solver](https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/)
to retain grid-independent symbolic compilation.

If you're interested in figuring out what's the fastest current solver for this kind of
PDE, check out the
[Brusselator benchmark in SciMLBenchmarks.jl](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/StiffODE/Bruss/)
