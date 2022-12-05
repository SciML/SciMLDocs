# [Automatic Uncertainty Quantification, Arbitrary Precision, and Unit Checking in ODE Solutions using Julia's Type System](@id ode_types)

One of the nice things about DifferentialEquations.jl is that it is designed with Julia's type system in mind. What this means is, if you have properly defined a Number type, you can use this number type in DifferentialEquations.jl's algorithms! [Note that this is restricted to the native algorithms of OrdinaryDiffEq.jl. The other solvers such as ODE.jl, Sundials.jl, and ODEInterface.jl are not compatible with some number systems.]

DifferentialEquations.jl determines the numbers to use in its solvers via the types that are designated by `tspan` and the initial condition of the problem. It will keep the time values in the same type as tspan, and the solution values in the same type as the initial condition. [Note that adaptive timestepping requires that the time type is compaible with `sqrt` and `^` functions. Thus dt cannot be Integer or numbers like that if adaptive timestepping is chosen].

Let's solve the linear ODE first define an easy way to get ODEProblems for the linear ODE:

```julia
using DifferentialEquations
f = (u,p,t) -> (p*u)
prob_ode_linear = ODEProblem(f,1/2,(0.0,1.0),1.01);
```

First let's solve it using Float64s. To do so, we just need to set u0 to a Float64 (which is done by the default) and dt should be a float as well.

```julia
prob = prob_ode_linear
sol =solve(prob,Tsit5())
println(sol)
```

Notice that both the times and the solutions were saved as Float64. Let's change the time to use rational values. Rationals are not compatible with adaptive time stepping since they do not have an L2 norm (this can be worked around by defining `internalnorm`, but rationals already explode in size!). To account for this, let's turn off adaptivity as well:

```julia
prob = ODEProblem(f,1/2,(0//1,1//1),101//100);
sol = solve(prob,RK4(),dt=1//2^(6),adaptive=false)
println(sol)
```

Now let's do something fun. Let's change the solution to use `Rational{BigInt}` and print out the value at the end of the simulation. To do so, simply change the definition of the initial condition.

```julia
prob = ODEProblem(f,BigInt(1)//BigInt(2),(0//1,1//1),101//100);
sol =solve(prob,RK4(),dt=1//2^(6),adaptive=false)
println(sol[end])
```

That's one huge fraction!

# Unit Checked Arithmetic via Unitful.jl

Units and dimensional analysis are standard tools across the sciences for checking the correctness of your equation. However, most ODE solvers only allow for the equation to be in dimensionless form, leaving it up to the user to both convert the equation to a dimensionless form, punch in the equations, and hopefully not make an error along the way.

DifferentialEquations.jl allows for one to use Unitful.jl to have unit-checked arithmetic natively in the solvers. Given the dispatch implementation of the Unitful, this has little overhead.

## Using Unitful

To use Unitful, you need to have the package installed. Then you can add units to your variables. For example:

```julia; wrap=false
using Unitful
t = 1.0u"s"
```

Notice that `t` is a variable with units in seconds. If we make another value with seconds, they can add

```julia; wrap=false
t2 = 1.02u"s"
t+t2
```

and they can multiply:

```julia; wrap=false
t*t2
```

You can even do rational roots:

```julia; wrap=false
sqrt(t)
```

Many operations work. These operations will check to make sure units are correct, and will throw an error for incorrect operations:

```julia; wrap=false
t + sqrt(t)
```

# Using Unitful with DifferentialEquations.jl

Just like with other number systems, you can choose the units for your numbers by simply specifying the units of the initial condition and the timestep. For example, to solve the linear ODE where the variable has units of Newton's and `t` is in Seconds, we would use:

```julia; wrap=false
using DifferentialEquations
f = (y,p,t) -> 0.5*y
u0 = 1.5u"N"
prob = ODEProblem(f,u0,(0.0u"s",1.0u"s"))
sol = solve(prob,Tsit5())
```

Notice that we recieved a unit mismatch error. This is correctly so! Remember that for an ODE:

$$\frac{dy}{dt} = f(t,y)$$

we must have that `f` is a rate, i.e. `f` is a change in `y` per unit time. So we need to fix the units of `f` in our example to be `N/s`. Notice that we then do not receive an error if we do the following:

```julia; wrap=false
f = (y,p,t) -> 0.5*y/3.0u"s"
prob = ODEProblem(f,u0,(0.0u"s",1.0u"s"))
sol = solve(prob,Tsit5())
```

This gives a a normal solution object. Notice that the values are all with the correct units:

```julia; wrap=false
print(sol[:])
```

We can plot the solution by removing the units:

```julia; wrap=false
using Plots
gr()
plot(ustrip(sol.t),ustrip(sol[:]),lw=3)
```

# Numbers with Uncertainties

The result of a measurement should be given as a number with an attached uncertainties, besides the physical unit, and all operations performed involving the result of the measurement should propagate the uncertainty, taking care of correlation between quantities.

There is a Julia package for dealing with numbers with uncertainties: [`Measurements.jl`](https://github.com/JuliaPhysics/Measurements.jl).  Thanks to Julia's features, `DifferentialEquations.jl` easily works together with `Measurements.jl` out-of-the-box.

This notebook will cover some of the examples from the tutorial about classical Physics.

## Caveat about `Measurement` type

Before going on with the tutorial, we must point up a subtlety of `Measurements.jl` that you should be aware of:

```julia
using Measurements

5.23 ± 0.14 === 5.23 ± 0.14
```

```julia
(5.23± 0.14) - (5.23 ± 0.14)
```

```julia
(5.23 ± 0.14) / (5.23 ± 0.14)
```

The two numbers above, even though have the same nominal value and the same uncertainties, are actually two different measurements that only by chance share the same figures and their difference and their ratio have a non-zero uncertainty.  It is common in physics to get very similar, or even equal, results for a repeated measurement, but the two measurements are not the same thing.

Instead, if you have *one measurement* and want to perform some operations involving it, you have to assign it to a variable:

```julia
x = 5.23 ± 0.14
x === x
```

```julia
x - x
```

```julia
x / x
```

## Radioactive Decay of Carbon-14

The rate of decay of carbon-14 is governed by a first order linear ordinary differential equation

$$\frac{\mathrm{d}u(t)}{\mathrm{d}t} = -\frac{u(t)}{\tau}$$

where $\tau$ is the mean lifetime of carbon-14, which is related to the half-life $t_{1/2} = (5730 \pm 40)$ years by the relation $\tau = t_{1/2}/\ln(2)$.

```julia
using DifferentialEquations, Measurements, Plots

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

# Analytic solution
u = exp.(- sol.t / τ)

plot(sol.t, sol.u, label = "Numerical", xlabel = "Years", ylabel = "Fraction of Carbon-14")
plot!(sol.t, u, label = "Analytic")
```

The two curves are perfectly superimposed, indicating that the numerical solution matches the analytic one.  We can check that also the uncertainties are correctly propagated in the numerical solution:

```julia
println("Quantity of carbon-14 after ",  sol.t[11], " years:")
println("Numerical: ", sol[11])
println("Analytic:  ", u[11])
```

Both the value of the numerical solution and its uncertainty match the analytic solution within the requested tolerance.  We can also note that close to 5730 years after the beginning of the decay (half-life of the radioisotope), the fraction of carbon-14 that survived is about 0.5.

## Simple pendulum

### Small angles approximation

The next problem we are going to study is the simple pendulum in the approximation of small angles.  We address this simplified case because there exists an easy analytic solution to compare.

The differential equation we want to solve is

$$\ddot{\theta} + \frac{g}{L} \theta = 0$$

where $g = (9.79 \pm 0.02)~\mathrm{m}/\mathrm{s}^2$ is the gravitational acceleration measured where the experiment is carried out, and $L = (1.00 \pm 0.01)~\mathrm{m}$ is the length of the pendulum.

When you set up the problem for `DifferentialEquations.jl` remember to define the measurements as variables, as seen above.

```julia
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

# Analytic solution
u = u₀[2] .* cos.(sqrt(g / L) .* sol.t)

plot(sol.t, getindex.(sol.u, 2), label = "Numerical")
plot!(sol.t, u, label = "Analytic")
```

Also in this case there is a perfect superimposition between the two curves, including their uncertainties.

We can also have a look at the difference between the two solutions:

```julia
plot(sol.t, getindex.(sol.u, 2) .- u, label = "")
```

## Arbitrary amplitude

Now that we know how to solve differential equations involving numbers with uncertainties we can solve the simple pendulum problem without any approximation.  This time the differential equation to solve is the following:

$$\ddot{\theta} + \frac{g}{L} \sin(\theta) = 0$$

```julia
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
