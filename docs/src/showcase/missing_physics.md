# [Automatically Discover Missing Physics by Embedding Machine Learning into Differential Equations](@id autocomplete)

One of the most time consuming parts of modeling is building the model. How do you know
when your model is correct? When you solve an inverse problem to calibrate your model
to data, who you gonna call if there are no parameters that make the model the data?
This is the problem that the Universal Differential Equation (UDE) approach solves: the
ability to start from the model you have, and suggest minimal mechanistic extensions that
would allow the model to fit the data. In this showcase we will show how to take a partially
correct model and auto-complete it to find the missing physics.

!!! note

    For a scientific background on the universal differential equation approach, check out
    [Universal Differential Equations for Scientific Machine Learning](https://arxiv.org/abs/2001.04385)

## Starting Point: The Packages To Use

There are many packages which are used as part of this showcase. Let's detail what they
are and how they are used. For the neural network training:

| Module                                                                                                   | Description                                           |
|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| [OrdinaryDiffEq.jl](https://docs.sciml.ai/DiffEqDocs/stable/) (DifferentialEquations.jl)                 | The numerical differential equation solvers           |
| [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/stable/)                                    | The adjoint methods, defines gradients of ODE solvers |
| [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)                                            | The optimization library                              |
| [OptimizationOptimisers.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/optimisers/) | The optimization solver package with `Adam`           |
| [OptimizationOptimJL.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/optim/)         | The optimization solver package with `BFGS`           |

For the symbolic model discovery:

| Module                                                                                                        | Description                                       |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)                                           | The symbolic modeling environment                 |
| [DataDrivenDiffEq.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/)                                         | The symbolic regression interface                 |
| [DataDrivenSparse.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/sparse_regression/) | The sparse regression symbolic regression solvers |

Julia standard libraries:

| Module        | Description                      |
|---------------|----------------------------------|
| LinearAlgebra | Required for the `norm` function |
| Statistics    | Required for the `mean` function |

And external libraries:

| Module                                                                       | Description                                         |
|------------------------------------------------------------------------------|-----------------------------------------------------|
| [Lux.jl](http://lux.csail.mit.edu/stable/)                                   | The deep learning (neural network) framework        |
| [ComponentArrays.jl](https://jonniedie.github.io/ComponentArrays.jl/stable/) | For the `ComponentArray` type to match Lux to SciML |
| [Plots.jl](https://docs.juliaplots.org/stable/)                              | The plotting and visualization library              |
| [StableRNGs.jl](https://docs.juliaplots.org/stable/)                         | Stable random seeding                               |

!!! note
    The deep learning framework [Flux.jl](https://fluxml.ai/) could be used in place of Flux,
    though most tutorials in SciML generally prefer Lux.jl due to its explicit parameter
    interface leading to nicer code. Both share the same internal implementations of core
    kernels, and thus have very similar feature support and performance.

```@example ude
# SciML Tools
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)
```

## Problem Setup

In order to know that we have automatically discovered the correct model, we will use
generated data from a known model. This model will be the Lotka-Volterra equations. These
equations are given by:

```math
\begin{aligned}
\frac{dx}{dt} &= \alpha x - \beta x y      \\
\frac{dy}{dt} &= -\delta y + \gamma x y    \\
\end{aligned}
```

This is a model of rabbits and wolves. ``\alpha x`` is the exponential growth of rabbits
in isolation, ``-\beta x y`` and ``\gamma x y`` are the interaction effects of wolves
eating rabbits, and ``-\delta y`` is the term for how wolves die hungry in isolation.

Now assume that we have never seen rabbits and wolves in the same room. We only know the
two effects ``\alpha x`` and ``-\delta y``. Can we use Scientific Machine Learning to
automatically discover an extension to what we already know? That is what we will solve
with the universal differential equation.

## Generating the Training Data

First, let's generate training data from the Lotka-Volterra equations. This is
straightforward and standard DifferentialEquations.jl usage. Our sample data is thus
generated as follows:

```@example ude
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# Define the experimental parameter
tspan = (0.0,5.0)
u0 = 5f0 * rand(rng, 2)
p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0,tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.25)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t

x̄ = mean(X, dims = 2)
noise_magnitude = 5e-3
Xₙ = X .+ (noise_magnitude*x̄) .* randn(rng, eltype(X), size(X))

plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])
```

## Definition of the Universal Differential Equation

Now let's define our UDE. We will use Lux.jl to define the neural network as follows:

```@example ude
rbf(x) = exp.(-(x.^2))

# Multilayer FeedForward
U = Lux.Chain(
    Lux.Dense(2,5,rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,2)
)
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
```

We then define the UDE as a dynamical system that is `u' = known(u) + NN(u)` like:

```@example ude
# Define the hybrid model
function ude_dynamics!(du,u, p, t, p_true)
    û = U(u, p, st)[1] # Network prediction
    du[1] = p_true[1]*u[1] + û[1]
    du[2] = -p_true[4]*u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du,u,p,t) = ude_dynamics!(du,u,p,t,p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!,Xₙ[:, 1], tspan, p)
```

Notice that the most important part of this is that the neural network does not have
hardcoded weights. The weights of the neural network are the parameters of the ODE system.
This means that if we change the parameters of the ODE system, then we will have updated
the internal neural networks to new weights. Keep that in mind for the next part.

... and tada: now we have a neural network integrated into our dynamical system!

!!! note

    Even if the known physics is only approximate or correct, it can be helpful to improve
    the fitting process! Check out
    [this JuliaCon talk](https://www.youtube.com/watch?v=lCDrCqqnPto) which dives into this
    issue.

## Setting Up the Training Loop

Now let's build a training loop around our UDE. First, let's make a function `predict`
which runs our simulation at new neural network weights. Recall that the weights of the
neural network are the parameters of the ODE, so what we need to do in `predict` is
update our ODE to our new parameters and then run it.

For this update step, we will use
the `remake` function from the
[SciMLProblem interface](https://docs.sciml.ai/DiffEqDocs/stable/basics/problem/#Modification-of-problem-types).
`remake` works by specifying `key = value` pairs to update in the problem fields. Thus to
update `u0`, we would add a keyword argument `u0 = ...`. To update the parameters, we'd
do `p = ...`. The field names can be acquired from the
[problem documentation](https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/) (or
the docstrings!).

Knowing this, our `predict` function looks like:

```@example ude
function predict(θ, X = Xₙ[:,1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                ))
end
```

Now for our loss function we solve the ODE at our new parameters and check its L2 loss
against the dataset. Using our `predict` function, this looks like:

```@example ude
function loss(θ)
    X̂ = predict(θ)
    mean(abs2, Xₙ .- X̂)
end
```

Lastly, what we will need to track our optimization is to define a callback as
[defined by the OptimizationProblem's solve interface](https://docs.sciml.ai/Optimization/stable/API/solve/).
Because our function only returns one value, the loss `l`, the callback will be a function
of the current parameters `θ` and `l`. Let's setup a callback prints every 50 steps the
current loss:

```@example ude
losses = Float64[]

callback = function (p, l)
  push!(losses, l)
  if length(losses)%50==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  return false
end
```

## Training

Now we're ready to train! To run the training process, we will need to build an
[`OptimizationProblem`](https://docs.sciml.ai/Optimization/stable/API/optimization_problem/).
Because we have a lot of parameteres, we will use
[Zygote.jl](https://docs.sciml.ai/Zygote/stable/). Optimization.jl makes the choice of
automatic diffeerentiation easy just by specifying an `adtype` in the
[`OptimizationFunction` construction](https://docs.sciml.ai/Optimization/stable/API/optimization_function/)

Knowing this, we can build our `OptimizationProblem` as follows:

```@example ude
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
```

Now... we optimize it. We will use a mixed strategy. First, let's do some iterations of
ADAM because it's better at finding a good general area of parameter space, but then we
will move to BFGS which will quickly hone in on a local minima. Note that if we only use
ADAM it will take a ton of iterations, and if we only use BFGS we normally end up in a
bad local minima, so this combination tends to be a good one for UDEs.

Thus we first solve the optimization problem with ADAM. Choosing a learning rate of 0.1
(tuned to be as high as possible that doesn't tend to make the loss shoot up), we see:

```@example ude
res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
```

Now we use the optimization result of the first run as the initial condition of the
second optimization, and run it with BFGS. This looks like:

```@example ude
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters = 1000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate  
p_trained = res2.u
```

and bingo we have a trained UDE.

## Visualizing the Trained UDE

How well did our neural network do? Let's take a look:

```@example ude
# Plot the losses
pl_losses = plot(1:5000, losses[1:5000], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(5001:length(losses), losses[5001:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
```

Next, we compare the original data to the output of the UDE predictor. Note that we can even create more samples from the underlying model by simply adjusting the time steps!

```@example ude
## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):mean(diff(solution.t))/2:last(solution.t)
X̂ = predict(p_trained, Xₙ[:,1], ts)
# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])
```

Lets see how well the unknown term has been approximated:

```@example ude
# Ideal unknown interactions of the predictor
Ȳ = [-p_[2]*(X̂[1,:].*X̂[2,:])';p_[3]*(X̂[1,:].*X̂[2,:])']
# Neural network guess
Ŷ = U(X̂,p_trained,st)[1]

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
```

And have a nice look at all the information:

```@example ude
# Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ-Ŷ)), yaxis = :log, xlabel = "t", ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2,1))

pl_overall = plot(pl_trajectory, pl_missing)
```

That looks pretty good. And if we are happy with deep learning, we can leave it at that:
we have trained a neural network to capture our missing dynamics.

But...

Can we also make it print out the LaTeX for what the missing equations were? Find out
more after the break!

## Symbolic regression via sparse regression ( SINDy based )

Okay that was a quick break, and that's good because this next part is pretty cool. Let's
use [DataDrivenDiffEq.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/) to transform our
trained neural network from machine learning mumbo jumbo into predictions of missing
mechanistic equations. To do this, we first generate a symbolic basis that represents the
space of mechanistic functions we believe this neural network should map to. Let's choose
a bunch of polynomial functions:

```@example ude
@variables u[1:2]
b = polynomial_basis(u, 4)
basis = Basis(b,u);
```
Now let's define our `DataDrivenProblem`s for the sparse regressions. To assess the
capability of the sparse regression, we will look at 3 cases:

* What if we trained no neural network and tried to automatically uncover the equations
  from the original noisy data? This is the approach in the literature known as structural
  identification of dynamical systems (SINDy). We will call this the full problem. This
  will assess whether this incorporation of prior information was helpful.
* What if we trained the neural network using the ideal right hand side missing derivative
  functions? This is the value computed in the plots above as `Ȳ`. This will tell us whether
  the symbolic discovery could work in ideal situations.
* Do the symbolic regression directly on the function `y = NN(x)`, i.e. the trained learned
  neural network. This is what we really want, and will tell us how to extend our known
  equations.

To define the full problem, we need to define a `DataDrivenProblem` that has the time
series of the solution `X`, the time points of the solution `t`, and the derivative
at each time point of the solution (obtained by the ODE solution's interpolation. We can just use an interpolation to get the derivative:

```@example ude
full_problem = ContinuousDataDrivenProblem(Xₙ, t)
```

Now for the other two symbolic regressions, we are regressing input/outputs of the missing
terms and thus we directly define the datasets as the input/output mappings like:

```@example ude
ideal_problem = DirectDataDrivenProblem(X̂, Ȳ)
nn_problem = DirectDataDrivenProblem(X̂, Ŷ)
```

Let's solve the data driven problems using sparse regression. We will use the `ADMM`
method, which requires we define a set of shrinking cutoff values `λ`, and we do this like:

```@example ude
λ = exp10.(-3:0.01:3)
opt = ADMM(λ)
```

This is one of many methods for sparse regression, consult the
[DataDrivenDiffEq.jl documentation](https://docs.sciml.ai/DataDrivenDiffEq/stable/) for
more information on the algorithm choices. Taking this, let's solve each of the sparse
regressions:

```@example ude
options = DataDrivenCommonOptions(
    maxiters = 10_000, normalize = DataNormalization(ZScoreTransform), selector = bic, digits = 1,
    data_processing = DataProcessing(split = 0.9, batchsize = 30, shuffle = true, rng = StableRNG(1111)))

full_res = solve(full_problem, basis, opt, options = options)
full_eqs = get_basis(full_res)
println(full_res)
```

```@example ude
options = DataDrivenCommonOptions(
    maxiters = 10_000, normalize = DataNormalization(ZScoreTransform), selector = bic, digits = 1,
    data_processing = DataProcessing(split = 0.9, batchsize = 30, shuffle = true, rng = StableRNG(1111)))

ideal_res = solve(ideal_problem, basis, opt, options = options)
ideal_eqs = get_basis(ideal_res)
println(ideal_res)
```

```@example ude
options = DataDrivenCommonOptions(
    maxiters = 10_000, normalize = DataNormalization(ZScoreTransform), selector = bic, digits = 1,
    data_processing = DataProcessing(split = 0.9, batchsize = 30, shuffle = true, rng = StableRNG(1111)))

nn_res = solve(nn_problem, basis, opt, options = options)
nn_eqs = get_basis(nn_res)
println(nn_res)
```

Note that we passed the identical options into each of the `solve` calls to get the same data for each call.

We already saw that the full problem has failed to identify the correct equations of motion.
To have a closer look, we can inspect the corresponding equations:

```@example ude
for eqs in (full_eqs, ideal_eqs, nn_eqs)
    println(eqs)
    println(get_parameter_map(eqs))
    println()
end
```

Next, we want to predict with our model. To do so, we embedd the basis into a function like before:

```@example ude
# Define the recovered, hybrid model
function recovered_dynamics!(du,u, p, t)
    û = nn_eqs(u, p) # Recovered equations
    du[1] = p_[1]*u[1] + û[1]
    du[2] = -p_[4]*u[2] + û[2]
end

estimation_prob = ODEProblem(recovered_dynamics!, u0, tspan, get_parameter_values(nn_eqs))
estimate = solve(estimation_prob, Tsit5(), saveat = solution.t)

# Plot
plot(solution)
plot!(estimate)
```

We are still a bit off, so we fine tune the parameters by simply minimizing the residuals between the UDE predictor and our recovered parametrized equations:

```@example ude
function parameter_loss(p)
    Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂))) 
    sum(abs2, Ŷ .- Y)
end

optf = Optimization.OptimizationFunction((x,p)->parameter_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters = 1000)
```

## Simulation

```@example ude
# Look at long term prediction
t_long = (0.0, 50.0)
estimation_prob = ODEProblem(recovered_dynamics!, u0, t_long, parameter_res)
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.1) # Using higher tolerances here results in exit of julia
plot(estimate_long)
```

```@example ude
true_prob = ODEProblem(lotka!, u0, t_long, p_)
true_solution_long = solve(true_prob, Tsit5(), saveat = estimate_long.t)
plot!(true_solution_long)
```

## Post Processing and Plots

```@example ude
c1 = 3 # RGBA(174/255,192/255,201/255,1) # Maroon
c2 = :orange # RGBA(132/255,159/255,173/255,1) # Red
c3 = :blue # RGBA(255/255,90/255,0,1) # Orange
c4 = :purple # RGBA(153/255,50/255,204/255,1) # Purple

p1 = plot(t,abs.(Array(solution) .- estimate)' .+ eps(Float32),
          lw = 3, yaxis = :log, title = "Timeseries of UODE Error",
          color = [3 :orange], xlabel = "t",
          label = ["x(t)" "y(t)"],
          titlefont = "Helvetica", legendfont = "Helvetica",
          legend = :topright)

# Plot L₂
p2 = plot3d(X̂[1,:], X̂[2,:], Ŷ[2,:], lw = 3,
     title = "Neural Network Fit of U2(t)", color = c1,
     label = "Neural Network", xaxis = "x", yaxis="y",
     titlefont = "Helvetica", legendfont = "Helvetica",
     legend = :bottomright)
plot!(X̂[1,:], X̂[2,:], Ȳ[2,:], lw = 3, label = "True Missing Term", color=c2)

p3 = scatter(solution, color = [c1 c2], label = ["x data" "y data"],
             title = "Extrapolated Fit From Short Training Data",
             titlefont = "Helvetica", legendfont = "Helvetica",
             markersize = 5)

plot!(p3,true_solution_long, color = [c1 c2], linestyle = :dot, lw=5, label = ["True x(t)" "True y(t)"])
plot!(p3,estimate_long, color = [c3 c4], lw=1, label = ["Estimated x(t)" "Estimated y(t)"])
plot!(p3,[2.99,3.01],[0.0,10.0],lw=1,color=:black, label = nothing)
annotate!([(1.5,13,text("Training \nData", 10, :center, :top, :black, "Helvetica"))])
l = @layout [grid(1,2)
             grid(1,1)]
plot(p1,p2,p3,layout = l)
```
