# [Fit a simulation to a dataset](@id fit_simulation)

Running simulations is only half of the battle. Many times, in order to make the simulation
realistic, you need to fit the simulation to data. The SciML ecosystem has **integration with
automatic differentiation and adjoint methods** to automatically make the fitting process
stable and efficient. Let's see this in action.

## Required Dependencies

The following parts of the SciML Ecosystem will be used in this tutorial:

| Module                                                                                                           | Description                                               |
|:---------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------- |
| [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)                                             | The differential equation solvers                         |
| [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)                                                    | The numerical optimization package                        |
| [OptimizationPolyalgorithms.jl](https://github.com/SciML/Optimization.jl/blob/master/lib/OptimizationPolyalgorithms/src/OptimizationPolyalgorithms.jl) | The optimizers we will use                                |
| [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/dev/)                                               | The connection of the SciML ecosystems to differentiation |

Along with the following general ecosystem packages:

| Module                                                         | Description                            |
|:-------------------------------------------------------------- |:-------------------------------------- |
| [Plots.jl](https://docs.juliaplots.org/stable/)                | The plotting and visualization package |
| [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/) | The automatic differentiation package  |

## Problem Setup: Fitting Lotka-Volterra Data

Assume that we know that the dynamics of our system are given by the
[Lotka-Volterra dynamical system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations):
Let $x(t)$ be the number of rabbits in the environment and $y(t)$ be the number of wolves.
This is the same dynamical system as [the first tutorial!](@ref first_sim)
The equation that defines the evolution of the species is given as follows:

```math
\begin{align}
\frac{dx}{dt} &= \alpha x - \beta x y\\
\frac{dy}{dt} &= -\gamma y + \delta x y
\end{align}
```

where ``\alpha, \beta, \gamma, \delta`` are parameters. Starting from equal numbers of
rabbits and wolves, ``x(0) = 1`` and ``y(0) = 1``.

Now, in [the first tutorial](@ref first_sim), we assumed:

> Luckily, a local guide provided us with some parameters that seem to match the system!

Sadly, magical nymphs do not always show up and give us parameters. Thus in this case,
we will need to use Optimization.jl to optimize the model parameters to best fit some
experimental data.
We are given experimentally observed data of both rabbit and wolf populations
over a time span of ``t_0 = 0`` to ``t_f = 10`` at every ``\Delta t = 1``.
Can we figure out what the parameter values should be directly from the data?

## Solution as Copy-Pastable Code
```@example
using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using ForwardDiff, Plots

# Define experimental data
t_data = 0:10
x_data = [1.000 2.773 6.773 0.971 1.886 6.101 1.398 1.335 4.353 3.247 1.034]
y_data = [1.000 0.259 2.015 1.908 0.323 0.629 3.458 0.508 0.314 4.547 0.906]
xy_data = vcat(x_data, y_data)

# Plot the provided data
scatter(t_data, xy_data', label=["x Data" "y Data"])

# Setup the ODE function
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval
tspan = (0.0, 10.0)

# LV equation parameter. p = [α, β, δ, γ]
pguess = [1.0, 1.2, 2.5, 1.2]

# Set up the ODE problem with our guessed parameter values
prob = ODEProblem(lotka_volterra!, u0, tspan, pguess)

# Solve the ODE problem with our guessed parameter values
initial_sol = solve(prob, saveat = 1)

# View the guessed model solution
plt = plot(initial_sol, label = ["x Prediction" "y Prediction"])
scatter!(plt, t_data, xy_data', label = ["x Data" "y Data"])

# Define a loss metric function to be minimized
function loss(newp)
    newprob = remake(prob, p = newp)
    sol = solve(newprob, saveat = 1)
    loss = sum(abs2, sol .- xy_data)
    return loss, sol
end

# Define a callback function to monitor optimization progress
function callback(p, l, sol)
    display(l)
    plt = plot(sol, ylim = (0, 6), label = ["Current x Prediction" "Current y Prediction"])
    scatter!(plt, t_data, xy_data', label = ["x Data" "y Data"])
    display(plt)
    return false
end

# Set up the optimization problem with our loss function and initial guess
adtype = AutoForwardDiff()
pguess = [1.0, 1.2, 2.5, 1.2]
optf = OptimizationFunction((x, _) -> loss(x), adtype)
optprob = OptimizationProblem(optf, pguess)

# Optimize the ODE parameters for best fit to our data
pfinal = solve(optprob, PolyOpt(),
               callback = callback,
               maxiters = 200)
α, β, γ, δ = round.(pfinal, digits=1)
```

## Step-by-Step Solution

### Step 1: Install and Import the Required Packages

To do this tutorial, we will need a few components. This is done using the Julia Pkg REPL:

```julia
using Pkg
Pkg.add([
            "DifferentialEquations",
            "Optimization",
            "OptimizationPolyalgorithms",
            "SciMLSensitivity",
            "ForwardDiff",
            "Plots",
        ])
```

Now we're ready. Let's load in these packages:

```@example odefit
using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using ForwardDiff, Plots
```

### Step 2: View the Training Data

In our example,
we are given observed values for `x` and `y` populations at eleven instances in time.
Let's make that the training data for our Lotka-Volterra dynamical system model.

```@example odefit
# Define experimental data
t_data = 0:10
x_data = [1.000 2.773 6.773 0.971 1.886 6.101 1.398 1.335 4.353 3.247 1.034]
y_data = [1.000 0.259 2.015 1.908 0.323 0.629 3.458 0.508 0.314 4.547 0.906]
xy_data = vcat(x_data, y_data)

# Plot the provided data
scatter(t_data, xy_data', label=["x Data" "y Data"])
```

!!! note

    The `Array` `xy_data` above has been oriented with time instances as columns
    so that it can be directly compared with an `ODESolution` object. (See
    [Solution Handling](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/#solution)
    for more information on accessing DifferentialEquation.jl solution data.)
    However, plotting an `Array` with Plots.jl requires the variables to be columns
    and the time instances to be rows.
    Thus, whenever the experimental data is plotted,
    the transpose `xy_data'` will be used.

### Step 3: Set Up the ODE Model

We know that our system will behave according to the Lotka-Volterra ODE model,
so let's set up that model with an initial guess at the parameter values:
`\alpha`, `\beta`, `\gamma`, and `\delta`.
Unlike [the first tutorial](@ref first_sim), which used ModelingToolkit,
let's demonstrate using [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
to directly define the ODE for the numerical solvers.

To do this, we define a vector-based mutating function that calculates the derivatives for
our system. We will define our system as a vector `u = [x,y]`, and thus `u[1] = x` and
`u[2] = y`. This means that we need to calculate the derivative as `du = [dx,dy]`.
Our parameters will simply be the vector `p = [α, β, δ, γ]`.
Writing down the Lotka-Volterra equations in the
DifferentialEquations.jl direct form thus looks like the following:

```@example odefit
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end
```

Now we need to define the initial condition, time span, and parameter vector
in order to solve this differential equation.
We do not currently know the parameter values,
but we will guess some values to start with and optimize them later.
Following the problem setup, this looks like:

```@example odefit
# Initial condition
u0 = [1.0, 1.0]

# Simulation interval
tspan = (0.0, 10.0)

# LV equation parameter. p = [α, β, δ, γ]
pguess = [1.0, 1.2, 2.5, 1.2]
```

Now we bring these pieces all together to define the `ODEProblem` and solve it.
Note that we solve this equation with the keyword argument `saveat = 1`
so that it saves a point at every ``\Delta t = 1`` to match our experimental data.

```@example odefit
# Set up the ODE problem with our guessed parameter values
prob = ODEProblem(lotka_volterra!, u0, tspan, pguess)

# Solve the ODE problem with our guessed parameter values
initial_sol = solve(prob, saveat = 1)

# View the guessed model solution
plt = plot(initial_sol, label = ["x Prediction" "y Prediction"])
scatter!(plt, t_data, xy_data', label = ["x Data" "y Data"])
```

    Clearly the parameter values that we guessed are not correct to model this system.
    However, we can use Optimization.jl together with DifferentialEquations.jl
    to fit our parameters to our training data.

!!! note

    For more details on using DifferentialEquations.jl, check out the
    [getting started with DifferentialEquations.jl tutorial](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/).

### Step 4: Set Up the Loss Function for Optimization

Now let's start the optimization process.
First, let's define a loss function to be minimized.
(It is also sometimes referred to as a cost function.)
For our loss function, we want to take a set of parameters,
create a new ODE which has everything the same except for the changed parameters,
solve this ODE with new parameters, and compare the ODE solution against the provided data.
In this case, the *loss* returned from the loss function is a quantification
of the difference between the current solution and the desired solution.
When this difference is minimized, our model prediction will closely approximate the observed system data.

To change our parameter values,
there is a useful functionality in the
[SciML problems interface](https://docs.sciml.ai/DiffEqDocs/stable/basics/problem/#Modification-of-problem-types)
called `remake` which creates a new version of an existing `SciMLProblem` with the aspect you want changed.
For example, if we wanted to change the initial condition `u0` of our ODE, we could do `remake(prob, u0 = newu0)`
For our case, we want to change around just the parameters, so we can do `remake(prob, p = newp)`. It is faster to `remake` an existing `SciMLProblem` than to create a new problem every iteration.

!!! note

    `remake` can change multiple items at once by passing more keyword arguments, i.e., `remake(prob, u0 = newu0, p = newp)`.
    This can be used to extend the example to simultaneously learn the initial conditions and parameters!

Now use `remake` to build the loss function. After we solve the new problem,
we will calculate the sum of squared errors as our loss metric.
The sum of squares can be quickly written in Julia via `sum(abs2,x)`.
Using this information, our loss function looks like:

```@example odefit
function loss(newp)
    newprob = remake(prob, p = newp)
    sol = solve(newprob, saveat = 1)
    loss = sum(abs2, sol .- xy_data)
    return loss, sol
end
```

Notice that our loss function returns the loss value as the first return,
but returns extra information (the ODE solution with the new parameters)
as an extra return argument.
We will explain why this extra return information is helpful in the next section.

### Step 5: Solve the Optimization Problem

This step will look very similar to [the first optimization tutorial](@ref first_opt),
except now we have a new loss function `loss` which returns both the loss value
and the associated ODE solution.
(In the previous tutorial, `L` only returned the loss value.)
The `Optimization.solve` function can accept an optional callback function
to monitor the optimization process using extra arguments returned from `loss`.

The callback syntax is always:

```
callback(
    optimization variables,
    the current loss value,
    other arguments returned from the loss function, ...
)
```

In this case, we will provide the callback the arguments `(_, l, sol)`,
since there are no additional optimization function parameters.
The return value of the callback function should default to `false`.
`Optimization.solve` will halt if/when the callback function returns `true` instead.
Typically the `return` statement would monitor the loss value
and stop once some criteria is reached, e.g. `return loss < 0.0001`,
but we will stop after a set number of iterations instead.
More details about callbacks in Optimization.jl can be found
[here](https://docs.sciml.ai/Optimization/stable/API/solve/).

```@example odefit
function callback(_, l, sol)
    display(l)
    plt = plot(sol, ylim = (0, 6), label = ["Current x Prediction" "Current y Prediction"])
    scatter!(plt, t_data, xy_data', label = ["x Data" "y Data"])
    display(plt)
    return false
end
```

With this callback function, every step of the optimization will display both the loss value and a plot of how the solution compares to the training data.

Now, just like [the first optimization tutorial](@ref first_opt),
we set up our `OptimizationFunction` and `OptimizationProblem`,
and then `solve` the `OptimizationProblem`.
We will initialize the `OptimizationProblem` with the same `pguess` we used
when setting up the ODE Model in Step 3.
Observe how `Optimization.solve` brings the model closer to the experimental data as it iterates towards better ODE parameter values!

Note that we are using the `PolyOpt()` solver choice here which is discussed https://docs.sciml.ai/Optimization/dev/optimization_packages/polyopt/ since parameter estimation of non-linear differential equations is generally a non-convex problem so we want to run a stochastic algorithm (Adam) to get close to the minimum and then finish off with a quasi-newton method (L-BFGS) to find the optima.
Together, this looks like:

```@example odefit
# Set up the optimization problem with our loss function and initial guess
adtype = AutoForwardDiff()
pguess = [1.0, 1.2, 2.5, 1.2]
optf = OptimizationFunction((x, _) -> loss(x), adtype)
optprob = OptimizationProblem(optf, pguess)

# Optimize the ODE parameters for best fit to our data
pfinal = solve(optprob,
               PolyOpt(),
               callback = callback,
               maxiters = 200)
α, β, γ, δ = round.(pfinal, digits=1)
```

!!! note

    When referencing the documentation for DifferentialEquations.jl and Optimization.jl
    simultaneously, note that the variables `f`, `u`, and `p` will refer to different quantities.

    DifferentialEquations.jl:

    ```math
    \frac{du}{dt} = f(u,p,t)
    ```

    - `f` in `ODEProblem` is the function defining the derivative `du` in the ODE.

        Here: `lotka_volterra!`

    - `u` in `ODEProblem` contains the state variables of `f`.

        Here: `x` and `y`

    - `p` in `ODEProblem` contains the parameter variables of `f`.

        Here: `\alpha`, `\beta`, `\gamma`, and `\delta`

    - `t` is the independent (time) variable.

        Here: indirectly defined with `tspan` in `ODEProblem` and `saveat` in `solve`

    Optimization.jl:

    ```math
    \min_{u} f(u,p)
    ```

    - `f` in `OptimizationProblem` is the function to minimize (optimize).

        Here: the [anonymous function](https://docs.julialang.org/en/v1/manual/functions/#man-anonymous-functions) `(x, _) -> loss(x)`

    - `u` in `OptimizationProblem` contains the state variables of `f` to be optimized.

        Here: the ODE parameters `\alpha`, `\beta`, `\gamma`, and `\delta` stored in `p`

    - `p` in `OptimizationProblem` contains any fixed
    [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) of `f`.

        Here: our `loss` function does not require any hyperparameters, so we pass `_` for this `p`.
