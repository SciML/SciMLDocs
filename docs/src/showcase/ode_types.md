# [Automatic Uncertainty Quantification, Arbitrary Precision, and Unit Checking in ODE Solutions using Julia's Type System](@id ode_types)

One of the nice things about DifferentialEquations.jl is that it is designed with Julia's
type system in mind. What this means is, if you have properly defined a Number type, you
can use this number type in DifferentialEquations.jl's algorithms! There's more than a
few useful/interesting types that can be used:

| Julia Type Name          | Julia Package                                                                         | Use case                                                        |
|--------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| BigFloat                 | Base Julia                                                                            | Higher precision solutions                                      |
| ArbFloat                 | [ArbFloats.jl](https://github.com/JuliaArbTypes/ArbFloats.jl)                         | More efficient higher precision solutions                       |
| Measurement              | [Measurements.jl](https://github.com/JuliaPhysics/Measurements.jl)                    | Uncertainty propagation                                         |
| MonteCarloMeasurement    | [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) | Uncertainty propagation                                         |
| Unitful                  | [Unitful.jl](https://painterqubits.github.io/Unitful.jl/stable/)                      | Unit-checked arithmetic                                         |
| Quaternion               | [Quaternions.jl](https://juliageometry.github.io/Quaternions.jl/stable/)              | Quaternions, duh.                                               |
| Fun                      | [ApproxFun.jl](https://juliaapproximation.github.io/ApproxFun.jl/latest/)             | Representing PDEs as ODEs in function spaces                    |
| AbstractOrthoPoly        | [PolyChaos.jl](https://docs.sciml.ai/PolyChaos/stable/)                               | Polynomial Chaos Expansion (PCE) for uncertainty quantification |
| Num                      | [Symbolics.jl](https://symbolics.juliasymbolics.org/stable/)                          | Build symbolic expressions of ODE solution approximations       |
| Taylor                   | [TaylorSeries.jl](https://github.com/JuliaDiff/TaylorSeries.jl)                       | Build a Taylor series around a solution point                   |
| Dual                     | [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/)                        | Perform forward-mode automatic differentiation                  |
| TrackedArray\TrackedReal | [ReverseDiff.jl](https://juliadiff.org/ReverseDiff.jl/stable/)                        | Perform reverse-mode automatic differentiation                  |

and on and on. That's only a subset of types people have effectively used on the SciML tools.

We will look into the `BigFloat`, `Measurement`, and `Unitful` cases to demonstrate the
utility of alternative numerical types.

## How Type Support Works in DifferentialEquations.jl / SciML

DifferentialEquations.jl determines the numbers to use in its solvers via the types that
are designated by `tspan` and the initial condition `u0` of the problem. It will keep the
time values in the same type as tspan, and the solution values in the same type as the
initial condition.

!!! note
    Support for this feature is restricted to the native algorithms of OrdinaryDiffEq.jl.
    The other solvers such as Sundials.jl, and ODEInterface.jl are incompatible with some
    number systems.

!!! warn
    Adaptive timestepping requires that the time type is compatible with `sqrt` and `^`
    functions. Thus for example, `tspan` cannot be `Int` if adaptive timestepping is chosen.

Let's use this feature in some cool ways!

## Arbitrary Precision: Rationals and BigFloats

Let's solve the linear ODE. First, define an easy way to get `ODEProblem`s for the linear ODE:

```@example odetypes
using DifferentialEquations
f(u,p,t) = p*u
prob_ode_linear = ODEProblem(f,1/2,(0.0,1.0),1.01);
```
Next, let's solve it using Float64s. To do so, we just need to set u0 to a Float64 (which is done by the default) and dt should be a float as well.

```@example odetypes
sol = solve(prob_ode_linear,Tsit5())
```

Notice that both the times and the solutions were saved as Float64. Let's change the state
to use `BigFloat` values. We do this by changing the `u0` to use `BigFloat`s like:


```@example odetypes
prob_ode_linear_bigu = ODEProblem(f,big(1/2),(0.0,1.0),1.01);
sol = solve(prob_ode_linear_bigu,Tsit5())
```

Now we see that `u` is in arbitrary precision `BigFloat`s, while `t` is in `Float64`. We
can then change `t` to be arbitrary precision `BigFloat`s by changing the types of the
`tspan` like:

```@example odetypes
prob_ode_linear_big = ODEProblem(f,big(1/2),(big(0.0),big(1.0)),1.01);
sol = solve(prob_ode_linear_big,Tsit5())
```

Now let's send it into the bizarre territory. Let's use rational values for everything.
Let's start by making the time type `Rational`. Rationals are incompatible with adaptive
time stepping since they do not have an L2 norm (this can be worked around by defining
`internalnorm`, but we will skip that in this tutorial). To account for this, let's turn
off adaptivity as well. Thus the following is a valid use of rational time (and parameter):

```@example odetypes
prob = ODEProblem(f,1/2,(0//1,1//1),101//100);
sol = solve(prob,RK4(),dt=1//2^(6),adaptive=false)
```

Now let's change the state to use `Rational{BigInt}`. You will see that we will need to
use the arbitrary-sized integers because... well... there's a reason people use
floating-point numbers with ODE solvers:

```@example odetypes
prob = ODEProblem(f,BigInt(1)//BigInt(2),(0//1,1//1),101//100);
sol =solve(prob,RK4(),dt=1//2^(6),adaptive=false)
```

Yeah...

```@example odetypes
sol[end]
```

That's one huge fraction! 0 floating-point error ODE solve achieved.

## Unit Checked Arithmetic via Unitful.jl

Units and dimensional analysis are standard tools across the sciences for checking the
correctness of your equation. However, most ODE solvers only allow for the equation to be
in dimensionless form, leaving it up to the user to both convert the equation to a
dimensionless form, punch in the equations, and hopefully not make an error along the way.

DifferentialEquations.jl allows for one to use Unitful.jl to have unit-checked arithmetic
natively in the solvers. Given the dispatch implementation of the Unitful, this has little
overhead because the unit checks occur at compile-time and not at runtime, and thus it does
not have a runtime effect unless conversions are required (i.e. converting `cm` to `m`),
which automatically adds a floating-point operation for the multiplication.

Let's see this in action.

### Using Unitful

To use Unitful, you need to have the package installed. Then you can add units to your
variables. For example:

```@example odetypes
using Unitful
t = 1.0u"s"
```

Notice that `t` is a variable with units in seconds. If we make another value with seconds,
they can add:

```@example odetypes
t2 = 1.02u"s"
t+t2
```

and they can multiply:

```@example odetypes
t*t2
```

You can even do rational roots:

```@example odetypes
sqrt(t)
```

Many operations work. These operations will check to make sure units are correct, and will
throw an error for incorrect operations:

```@example odetypes
#t + sqrt(t)
```

### Using Unitful with DifferentialEquations.jl

Just like with other number systems, you can choose the units for your numbers by simply
specifying the units of the initial condition and the timespan. For example, to solve the
linear ODE where the variable has units of Newton's and `t` is in seconds, we would use:

```@example odetypes
using DifferentialEquations
f(u,p,t) = 0.5*u
u0 = 1.5u"N"
prob = ODEProblem(f,u0,(0.0u"s",1.0u"s"))
#sol = solve(prob,Tsit5())
```

Notice that we received a unit mismatch error. This is correctly so! Remember that for an
ODE:

```math
\frac{dy}{dt} = f(t,y)
```

we must have that `f` is a rate, i.e. `f` is a change in `y` per unit time. So, we need to
fix the units of `f` in our example to be `N/s`. Notice that we then do not receive an
error if we do the following:

```@example odetypes
f(y,p,t) = 0.5*y/3.0u"s"
prob = ODEProblem(f,u0,(0.0u"s",1.0u"s"))
sol = solve(prob,Tsit5())
```

This gives a normal solution object. Notice that the values are all with the correct units:

```@example odetypes
print(sol[:])
```

And when we plot the solution, it automatically adds the units:

```@example odetypes
using Plots
gr()
plot(sol,lw=3)
```

# Measurements.jl: Numbers with Linear Uncertainty Propagation

The result of a measurement should be given as a number with an attached uncertainty,
besides the physical unit, and all operations performed involving the result of the
measurement should propagate the uncertainty, taking care of correlation between quantities.

There is a Julia package for dealing with numbers with uncertainties:
[`Measurements.jl`](https://github.com/JuliaPhysics/Measurements.jl). Thanks to Julia's
features, `DifferentialEquations.jl` easily works together with `Measurements.jl`
out-of-the-box.

Let's try to automate uncertainty propagation through number types on some classical
physics examples!

### Warning about `Measurement` type

Before going on with the tutorial, we must point up a subtlety of `Measurements.jl` that
you should be aware of:

```@example odetypes
using Measurements
5.23 ± 0.14 === 5.23 ± 0.14
```

```@example odetypes
(5.23± 0.14) - (5.23 ± 0.14)
```

```@example odetypes
(5.23 ± 0.14) / (5.23 ± 0.14)
```

The two numbers above, even though have the same nominal value and the same uncertainties,
are actually two different measurements that only by chance share the same figures and
their difference and their ratio have a non-zero uncertainty.  It is common in physics to
get very similar, or even equal, results for a repeated measurement, but the two
measurements are not the same thing.

Instead, if you have *one measurement* and want to perform some operations involving it,
you have to assign it to a variable:

```@example odetypes
x = 5.23 ± 0.14
x === x
```

```@example odetypes
x - x
```

```@example odetypes
x / x
```

With that in mind, let's start using Measurements.jl for realsies.

### Automated UQ on an ODE: Radioactive Decay of Carbon-14

The rate of decay of carbon-14 is governed by a first order linear ordinary differential
equation:

```math
\frac{\mathrm{d}u(t)}{\mathrm{d}t} = -\frac{u(t)}{\tau}
```

where $\tau$ is the mean lifetime of carbon-14, which is related to the half-life
$t_{1/2} = (5730 \pm 40)$ years by the relation $\tau = t_{1/2}/\ln(2)$. Writing this in
DifferentialEquations.jl syntax, this looks like:

```@example odetypes
# Half-life and mean lifetime of radiocarbon, in years
t_12 = 5730 ± 40
τ = t_12 / log(2)

#Setup
u₀ = 1 ± 0
tspan = (0.0, 10000.0)

#Define the problem
radioactivedecay(u,p,t) = - u / τ

#Pass to solver
prob = ODEProblem(radioactivedecay, u₀, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-8)
```

And bingo: numbers with uncertainty went in, so numbers with uncertainty came out. But can
we trust those values for the uncertainty?

We can check the uncertainty quantification by evaluating an analytical solution to the
ODE. Since it's a linear ODE, the analytical solution is simply given by the exponential:

```@example odetypes
u = exp.(- sol.t / τ)
```

How do the two solutions compare?

```@example odetypes
plot(sol.t, sol.u, label = "Numerical", xlabel = "Years", ylabel = "Fraction of Carbon-14")
plot!(sol.t, u, label = "Analytic")
```

The two curves are perfectly superimposed, indicating that the numerical solution matches
the analytic one.  We can check that also the uncertainties are correctly propagated in the
numerical solution:

```@example odetypes
println("Quantity of carbon-14 after ",  sol.t[11], " years:")
println("Numerical: ", sol[11])
println("Analytic:  ", u[11])
```

Bullseye. Both the value of the numerical solution and its uncertainty match the analytic
solution within the requested tolerance.  We can also note that close to 5730 years after
the beginning of the decay (half-life of the radioisotope), the fraction of carbon-14 that
survived is about 0.5.

### Simple pendulum: Small angles approximation

The next problem we are going to study is the simple pendulum in the approximation of
small angles.  We address this simplified case because there exists an easy analytic
solution to compare.

The differential equation we want to solve is:

```math
\ddot{\theta} + \frac{g}{L} \theta = 0
```

where ``g = (9.79 \pm 0.02)~\mathrm{m}/\mathrm{s}^2`` is the gravitational acceleration
measured where the experiment is carried out, and ``L = (1.00 \pm 0.01)~\mathrm{m}`` is the
length of the pendulum.

When you set up the problem for `DifferentialEquations.jl` remember to define the
measurements as variables, as seen above.

```@example odetypes
using DifferentialEquations, Measurements, Plots

g = 9.79 ± 0.02; # Gravitational constants
L = 1.00 ± 0.01; # Length of the pendulum

#Initial Conditions
u₀ = [0 ± 0, π / 60 ± 0.01] # Initial speed and initial angle
tspan = (0.0, 6.3)

#Define the problem
function simplependulum(du,u,p,t)
    θ  = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(g/L)*θ
end

#Pass to solvers
prob = ODEProblem(simplependulum, u₀, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-6)
```

And that's it! What about comparing it this time to the analytical solution?

```@example odetypes
u = u₀[2] .* cos.(sqrt(g / L) .* sol.t)

plot(sol.t, getindex.(sol.u, 2), label = "Numerical")
plot!(sol.t, u, label = "Analytic")
```

Bingo. Also in this case there is a perfect superimposition between the two curves,
including their uncertainties.

We can also have a look at the difference between the two solutions:

```@example odetypes
plot(sol.t, getindex.(sol.u, 2) .- u, label = "")
```

Tiny difference on the order of the chosen `1e-6` tolerance.

### Simple pendulum: Arbitrary amplitude

Now that we know how to solve differential equations involving numbers with uncertainties,
we can solve the simple pendulum problem without any approximation. This time, the
differential equation to solve is the following:

```math
\ddot{\theta} + \frac{g}{L} \sin(\theta) = 0
```

That would be done via:

```@example odetypes
g = 9.79 ± 0.02; # Gravitational constants
L = 1.00 ± 0.01; # Length of the pendulum

#Initial Conditions
u₀ = [0 ± 0, π / 3 ± 0.02] # Initial speed and initial angle
tspan = (0.0, 6.3)

#Define the problem
function simplependulum(du,u,p,t)
    θ  = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(g/L) * sin(θ)
end

#Pass to solvers
prob = ODEProblem(simplependulum, u₀, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-6)

plot(sol.t, getindex.(sol.u, 2), label = "Numerical")
```

### Warning about Linear Uncertainty Propagation

Measurements.jl uses linear uncertainty propagation, which has an error associated with it.
[MonteCarloMeasurements.jl has a page which showcases where this method can lead to incorrect uncertainty measurements](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable/comparison/).
Thus for more nonlinear use cases, it's suggested that one uses one of the more powerful
UQ methods, such as:

* [MonteCarloMeasurements.jl](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable/)
* [PolyChaos.jl](https://docs.sciml.ai/PolyChaos/stable/)
* [SciMLExpectations.jl](https://docs.sciml.ai/SciMLExpectations/stable/)
* [The ProbInts Uncertainty Quantification callbacks](https://docs.sciml.ai/DiffEqCallbacks/stable/uncertainty_quantification/)

Basically, types can make the algorithm you want to run exceedingly simple to do, but make
sure it's the correct algorithm!
