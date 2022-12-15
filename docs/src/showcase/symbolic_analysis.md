# [Symbolic-Numeric Analysis of Parameter Identifiability and Model Stability](@id symbolic_analysis)

The mixture of symbolic computing with numeric computing, which we call symbolic-numeric
programming, is one of the central features of the SciML ecosystem. With core aspects
like the [Symbolics.jl Computer Algebra System](https://symbolics.juliasymbolics.org/stable/)
and its integration via [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/),
the SciML ecosystem gracefully mixes analytical symbolic computations with the numerical
solver processes to accelerate solvers, give additional information
(sparsity, identifiability), automatically fix numerical stability issues, and more.

In this showcase we will highlight two aspects of symbolic-numeric programming.

1. Automated index reduction of DAEs. While arbitrary [differential-algebraic equation
   systems can be written in DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dae_example/),
   not all mathematical formulations of a system are equivalent. Some are numerically
   difficult to solve, or even require special solvers. Some are easy. Can we recognize
   which formulations are hard and automatically transform them into the easy ones? Yes.
2. Structural parameter identifiability. When fitting parameters to data, there's always
   assumptions about whether there is a unique parameter set that achieves such a data
   fit. But is this actually the case? The structural identifiability tooling allows one
   to analytically determine whether, in the limit of infinite data on a subset of
   observables, one could in theory uniquely identify the parameters (global identifiability),
   identify the parameters up to a discrete set (local identifiability), or whether
   there's an infinite manifold of solutions to the inverse problem (nonidentifiable).

Let's dig into these two cases!

# Automated Index Reduction of DAEs

In many cases one may accidentally write down a DAE that is not easily solvable
by numerical methods. In this tutorial we will walk through an example of a
pendulum which accidentally generates an index-3 DAE, and show how to use the
`modelingtoolkitize` to correct the model definition before solving.

## Copy-Pastable Example

```@example indexred
using ModelingToolkit
using LinearAlgebra
using OrdinaryDiffEq
using Plots

function pendulum!(du, u, p, t)
    x, dx, y, dy, T = u
    g, L = p
    du[1] = dx
    du[2] = T*x
    du[3] = dy
    du[4] = T*y - g
    du[5] = x^2 + y^2 - L^2
    return nothing
end
pendulum_fun! = ODEFunction(pendulum!, mass_matrix=Diagonal([1,1,1,1,0]))
u0 = [1.0, 0, 0, 0, 0]
p = [9.8, 1]
tspan = (0, 10.0)
pendulum_prob = ODEProblem(pendulum_fun!, u0, tspan, p)
traced_sys = modelingtoolkitize(pendulum_prob)
pendulum_sys = structural_simplify(dae_index_lowering(traced_sys))
prob = ODAEProblem(pendulum_sys, [], tspan)
sol = solve(prob, Tsit5(),abstol=1e-8,reltol=1e-8)
plot(sol, vars=states(traced_sys))
```

## Explanation

### Attempting to Solve the Equation

In this tutorial we will look at the pendulum system:

```math
\begin{aligned}
    x^\prime &= v_x\\
    v_x^\prime &= Tx\\
    y^\prime &= v_y\\
    v_y^\prime &= Ty - g\\
    0 &= x^2 + y^2 - L^2
\end{aligned}
```

As a good DifferentialEquations.jl user, one would follow
[the mass matrix DAE tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dae_example/#Mass-Matrix-Differential-Algebraic-Equations-(DAEs))
to arrive at code for simulating the model:

```@example indexred
using OrdinaryDiffEq, LinearAlgebra
function pendulum!(du, u, p, t)
    x, dx, y, dy, T = u
    g, L = p
    du[1] = dx; du[2] = T*x
    du[3] = dy; du[4] = T*y - g
    du[5] = x^2 + y^2 - L^2
end
pendulum_fun! = ODEFunction(pendulum!, mass_matrix=Diagonal([1,1,1,1,0]))
u0 = [1.0, 0, 0, 0, 0]; p = [9.8, 1]; tspan = (0, 10.0)
pendulum_prob = ODEProblem(pendulum_fun!, u0, tspan, p)
solve(pendulum_prob,Rodas4())
```

However, one will quickly be greeted with the unfortunate message:

```
┌ Warning: First function call produced NaNs. Exiting.
└ @ OrdinaryDiffEq C:\Users\accou\.julia\packages\OrdinaryDiffEq\yCczp\src\initdt.jl:76
┌ Warning: Automatic dt set the starting dt as NaN, causing instability.
└ @ OrdinaryDiffEq C:\Users\accou\.julia\packages\OrdinaryDiffEq\yCczp\src\solve.jl:485
┌ Warning: NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.
└ @ SciMLBase C:\Users\accou\.julia\packages\SciMLBase\DrPil\src\integrator_interface.jl:325
```

Did you implement the DAE incorrectly? No. Is the solver broken? No.

### Understanding DAE Index

It turns out that this is a property of the DAE that we are attempting to solve.
This kind of DAE is known as an index-3 DAE. For a complete discussion of DAE
index, see [this article](http://www.scholarpedia.org/article/Differential-algebraic_equations).
Essentially the issue here is that we have 4 differential variables (``x``, ``v_x``, ``y``, ``v_y``)
and one algebraic variable ``T`` (which we can know because there is no `D(T)`
term in the equations). An index-1 DAE always satisfies that the Jacobian of
the algebraic equations is non-singular. Here, the first 4 equations are
differential equations, with the last term the algebraic relationship. However,
the partial derivative of `x^2 + y^2 - L^2` w.r.t. `T` is zero, and thus the
Jacobian of the algebraic equations is the zero matrix and thus it's singular.
This is a very quick way to see whether the DAE is index 1!

The problem with higher order DAEs is that the matrices used in Newton solves
are singular or close to singular when applied to such problems. Because of this
fact, the nonlinear solvers (or Rosenbrock methods) break down, making them
difficult to solve. The classic paper [DAEs are not ODEs](https://epubs.siam.org/doi/10.1137/0903023)
goes into detail on this and shows that many methods are no longer convergent
when index is higher than one. So it's not necessarily the fault of the solver
or the implementation: this is known.

But that's not a satisfying answer, so what do you do about it?

### Transforming Higher Order DAEs to Index-1 DAEs

It turns out that higher order DAEs can be transformed into lower order DAEs.
[If you differentiate the last equation two times and perform a substitution,
you can arrive at the following set of equations](https://courses.seas.harvard.edu/courses/am205/g_act/dae_notes.pdf):

```math
\begin{aligned}
x^\prime =& v_x \\
v_x^\prime =& x T \\
y^\prime =& v_y \\
v_y^\prime =& y T - g \\
0 =& 2 \left(v_x^{2} + v_y^{2} + y ( y T - g ) + T x^2 \right)
\end{aligned}
```

Note that this is mathematically-equivalent to the equation that we had before,
but the Jacobian w.r.t. `T` of the algebraic equation is no longer zero because
of the substitution. This means that if you wrote down this version of the model
it will be index-1 and solve correctly! In fact, this is how DAE index is
commonly defined: the number of differentiations it takes to transform the DAE
into an ODE, where an ODE is an index-0 DAE by substituting out all of the
algebraic relationships.

### Automating the Index Reduction

However, requiring the user to sit there and work through this process on
potentially millions of equations is an unfathomable mental overhead. But,
we can avoid this by using methods like
[the Pantelides algorithm](https://ptolemy.berkeley.edu/projects/embedded/eecsx44/lectures/Spring2013/modelica-dae-part-2.pdf)
for automatically performing this reduction to index 1. While this requires the
ModelingToolkit symbolic form, we use `modelingtoolkitize` to transform
the numerical code into symbolic code, run `dae_index_lowering` lowering,
then transform back to numerical code with `ODEProblem`, and solve with a
numerical solver. Let's try that out:

```@example indexred
traced_sys = modelingtoolkitize(pendulum_prob)
pendulum_sys = structural_simplify(dae_index_lowering(traced_sys))
prob = ODEProblem(pendulum_sys, Pair[], tspan)
sol = solve(prob, Rodas4())

using Plots
plot(sol, vars=states(traced_sys))
```

Note that plotting using `states(traced_sys)` is done so that any
variables which are symbolically eliminated, or any variable reorderings
done for enhanced parallelism/performance, still show up in the resulting
plot and the plot is shown in the same order as the original numerical
code.

Note that we can even go a little bit further. If we use the `ODAEProblem`
constructor, we can remove the algebraic equations from the states of the
system and fully transform the index-3 DAE into an index-0 ODE which can
be solved via an explicit Runge-Kutta method:

```@example indexred
traced_sys = modelingtoolkitize(pendulum_prob)
pendulum_sys = structural_simplify(dae_index_lowering(traced_sys))
prob = ODAEProblem(pendulum_sys, Pair[], tspan)
sol = solve(prob, Tsit5(),abstol=1e-8,reltol=1e-8)
plot(sol, vars=states(traced_sys))
```

And there you go: this has transformed the model from being too hard to
solve with implicit DAE solvers, to something that is easily solved with
explicit Runge-Kutta methods for non-stiff equations.

# Parameter Identifiability in ODE Models

Ordinary differential equations are commonly used for modeling real-world processes. The problem of parameter identifiability is one of the key design challenges for mathematical models. A parameter is said to be _identifiable_ if one can recover its value from experimental data. _Structural_ identifiability is a theoretical property of a model that answers this question. In this tutorial, we will show how to use `StructuralIdentifiability.jl` with `ModelingToolkit.jl` to assess identifiability of parameters in ODE models. The theory behind `StructuralIdentifiability.jl` is presented in paper [^4].

We will start by illustrating **local identifiability** in which a parameter is known up to _finitely many values_, and then proceed to determining **global identifiability**, that is, which parameters can be identified _uniquely_.

To install `StructuralIdentifiability.jl`, simply run
```julia
using Pkg
Pkg.add("StructuralIdentifiability")
```

The package has a standalone data structure for ordinary differential equations but is also compatible with `ODESystem` type from `ModelingToolkit.jl`.

## Local Identifiability
### Input System

We will consider the following model:

$$\begin{cases}
\frac{d\,x_4}{d\,t} = - \frac{k_5 x_4}{k_6 + x_4},\\
\frac{d\,x_5}{d\,t} = \frac{k_5 x_4}{k_6 + x_4} - \frac{k_7 x_5}{(k_8 + x_5 + x_6)},\\
\frac{d\,x_6}{d\,t} = \frac{k_7 x_5}{(k_8 + x_5 + x_6)} - \frac{k_9  x_6  (k_{10} - x_6) }{k_{10}},\\
\frac{d\,x_7}{d\,t} = \frac{k_9  x_6  (k_{10} - x_6)}{ k_{10}},\\
y_1 = x_4,\\
y_2 = x_5\end{cases}$$

This model describes the biohydrogenation[^1] process[^2] with unknown initial conditions.

### Using the `ODESystem` object
To define the ode system in Julia, we use `ModelingToolkit.jl`.

We first define the parameters, variables, differential equations and the output equations.
```julia
using StructuralIdentifiability, ModelingToolkit

# define parameters and variables
@variables t x4(t) x5(t) x6(t) x7(t) y1(t) y2(t)
@parameters k5 k6 k7 k8 k9 k10
D = Differential(t)

# define equations
eqs = [
    D(x4) ~ - k5 * x4 / (k6 + x4),
    D(x5) ~ k5 * x4 / (k6 + x4) - k7 * x5/(k8 + x5 + x6),
    D(x6) ~ k7 * x5 / (k8 + x5 + x6) - k9 * x6 * (k10 - x6) / k10,
    D(x7) ~ k9 * x6 * (k10 - x6) / k10
]

# define the output functions (quantities that can be measured)
measured_quantities = [y1 ~ x4, y2 ~ x5]

# define the system
de = ODESystem(eqs, t, name=:Biohydrogenation)

```

After that we are ready to check the system for local identifiability:
```julia
# query local identifiability
# we pass the ode-system
local_id_all = assess_local_identifiability(de, measured_quantities=measured_quantities, p=0.99)
                # [ Info: Preproccessing `ModelingToolkit.ODESystem` object
                # 6-element Vector{Bool}:
                #  1
                #  1
                #  1
                #  1
                #  1
                #  1
```
We can see that all states (except $x_7$) and all parameters are locally identifiable with probability 0.99.

Let's try to check specific parameters and their combinations
```julia
to_check = [k5, k7, k10/k9, k5+k6]
local_id_some = assess_local_identifiability(de, measured_quantities=measured_quantities, funcs_to_check=to_check, p=0.99)
                # 4-element Vector{Bool}:
                #  1
                #  1
                #  1
                #  1
```

Notice that in this case, everything (except the state variable $x_7$) is locally identifiable, including combinations such as $k_{10}/k_9, k_5+k_6$

## Global Identifiability

In this part tutorial, let us cover an example problem of querying the ODE for globally identifiable parameters.

### Input System

Let us consider the following four-dimensional model with two outputs:

$$\begin{cases}
    x_1'(t) = -b  x_1(t) + \frac{1 }{ c + x_4(t)},\\
    x_2'(t) = \alpha  x_1(t) - \beta  x_2(t),\\
    x_3'(t) = \gamma  x_2(t) - \delta  x_3(t),\\
    x_4'(t) = \sigma  x_4(t)  \frac{(\gamma x_2(t) - \delta x_3(t))}{ x_3(t)},\\
    y(t) = x_1(t)
\end{cases}$$

We will run a global identifiability check on this enzyme dynamics[^3] model. We will use the default settings: the probability of correctness will be `p=0.99` and we are interested in identifiability of all possible parameters

Global identifiability needs information about local identifiability first, but the function we chose here will take care of that extra step for us.

__Note__: as of writing this tutorial, UTF-symbols such as Greek characters are not supported by one of the project's dependencies, see [this issue](https://github.com/SciML/StructuralIdentifiability.jl/issues/43).

```julia
using StructuralIdentifiability, ModelingToolkit
@parameters b c a beta g delta sigma
@variables t x1(t) x2(t) x3(t) x4(t) y(t) y2(t)
D = Differential(t)

eqs = [
    D(x1) ~ -b * x1 + 1/(c + x4),
    D(x2) ~ a * x1 - beta * x2,
    D(x3) ~ g * x2 - delta * x3,
    D(x4) ~ sigma * x4 * (g * x2 - delta * x3)/x3
]

measured_quantities = [y~x1+x2, y2~x2]


ode = ODESystem(eqs, t, name=:GoodwinOsc)

@time global_id = assess_identifiability(ode, measured_quantities=measured_quantities)
                    # 30.672594 seconds (100.97 M allocations: 6.219 GiB, 3.15% gc time, 0.01% compilation time)
                    # Dict{Num, Symbol} with 7 entries:
                    #   a     => :globally
                    #   b     => :globally
                    #   beta  => :globally
                    #   c     => :globally
                    #   sigma => :globally
                    #   g     => :nonidentifiable
                    #   delta => :globally
```
We can see that only parameters `a, g` are unidentifiable and everything else can be uniquely recovered.

Let us consider the same system but with two inputs and we will try to find out identifiability with probability `0.9` for parameters `c` and `b`:

```julia
using StructuralIdentifiability, ModelingToolkit
@parameters b c a beta g delta sigma
@variables t x1(t) x2(t) x3(t) x4(t) y(t) u1(t) [input=true] u2(t) [input=true]
D = Differential(t)

eqs = [
    D(x1) ~ -b * x1 + 1/(c + x4),
    D(x2) ~ a * x1 - beta * x2 - u1,
    D(x3) ~ g * x2 - delta * x3 + u2,
    D(x4) ~ sigma * x4 * (g * x2 - delta * x3)/x3
]
measured_quantities = [y~x1+x2, y2~x2]

# check only 2 parameters
to_check = [b, c]

ode = ODESystem(eqs, t, name=:GoodwinOsc)

global_id = assess_identifiability(ode, measured_quantities=measured_quantities, funcs_to_check=to_check, p=0.9)
            # Dict{Num, Symbol} with 2 entries:
            #   b => :globally
            #   c => :globally
```

Both parameters `b, c` are globally identifiable with probability `0.9` in this case.

[^1]:
    > R. Munoz-Tamayo, L. Puillet, J.B. Daniel, D. Sauvant, O. Martin, M. Taghipoor, P. Blavy [*Review: To be or not to be an identifiable model. Is this a relevant question in animal science modelling?*](https://doi.org/10.1017/S1751731117002774), Animal, Vol 12 (4), 701-712, 2018. The model is the ODE system (3) in Supplementary Material 2, initial conditions are assumed to be unknown.

[^2]:
    > Moate P.J., Boston R.C., Jenkins T.C. and Lean I.J., [*Kinetics of Ruminal Lipolysis of Triacylglycerol and Biohydrogenationof Long-Chain Fatty Acids: New Insights from Old Data*](doi:10.3168/jds.2007-0398), Journal of Dairy Science 91, 731–742, 2008

[^3]:
    > Goodwin, B.C. [*Oscillatory behavior in enzymatic control processes*](https://doi.org/10.1016/0065-2571(65)90067-1), Advances in Enzyme Regulation, Vol 3 (C), 425-437, 1965

[^4]:
    > Dong, R., Goodbrake, C., Harrington, H. A., & Pogudin, G. [*Computing input-output projections of dynamical models with applications to structural identifiability*](https://arxiv.org/pdf/2111.00991). arXiv preprint arXiv:2111.00991.
