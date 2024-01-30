# [Uncertainty Quantified Deep Bayesian Model Discovery](@id bnode)

In this tutorial, we show how SciML can combine the differential equation solvers seamlessly
with Bayesian estimation libraries like AdvancedHMC.jl and Turing.jl. This enables
converting Neural ODEs to Bayesian Neural ODEs, which enables us to estimate the error in
the Neural ODE estimation and forecasting. In this tutorial, a working example of the
Bayesian Neural ODE: NUTS sampler is shown.

!!! note
    
    For more details, have a look at this paper: https://arxiv.org/abs/2012.07244

## Step 1: Import Libraries

For this example, we will need the following libraries:

```@example bnode
# SciML Libraries
using DiffEqFlux, DifferentialEquations

# External Tools
using Random, Plots, Lux, Zygote, AdvancedHMC, MCMCChains, StatsPlots, ComponentArrays
```

## Setup: Get the data from the Spiral ODE example

We will also need data to fit against. As a demonstration, we will generate our data
using a simple cubic ODE `u' = A*u^3` as follows:

```@example bnode
u0 = [2.0; 0.0]
datasize = 40
tspan = (0.0, 1)
tsteps = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
```

We will want to train a neural network to capture the dynamics that fit `ode_data`.

## Step 2: Define the Neural ODE architecture.

Note that this step potentially offers a lot of flexibility in the number of layers/ number
of units in each layer. It may not necessarily be true that a 100 units architecture is
better at prediction/forecasting than a 50 unit architecture. On the other hand, a
complicated architecture can take a huge computational time without increasing performance.

```@example bnode
dudt2 = Lux.Chain(x -> x .^ 3,
                   Lux.Dense(2, 50, tanh),
                   Lux.Dense(50, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
rng = Random.default_rng()
p, st = Lux.setup(rng, dudt2)
p = ComponentArray{Float64}(p)
```

## Step 3: Define the loss function for the Neural ODE.

```@example bnode
function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end
```

## Step 4: Now we start integrating the Bayesian estimation workflow as prescribed by the AdvancedHMC interface with the NeuralODE defined above

The AdvancedHMC interface requires us to specify: (a) the Hamiltonian log density and its gradient , (b) the sampler and (c) the step size adaptor function.

For the Hamiltonian log density, we use the loss function. The θ*θ term denotes the use of Gaussian priors.

The user can make several modifications to Step 4. The user can try different acceptance ratios, warmup samples and posterior samples. One can also use the Variational Inference (ADVI) framework, which doesn't work quite as well as NUTS. The SGLD (Stochastic Gradient Langevin Descent) sampler is seen to have a better performance than NUTS. Have a look at https://sebastiancallh.github.io/post/langevin/ for a brief introduction to SGLD.

```@example bnode
l(θ) = -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ .* θ)
function dldθ(θ)
    x, lambda = Zygote.pullback(l, θ)
    grad = first(lambda(1))
    return x, grad
end

metric = DiagEuclideanMetric(length(p))
h = Hamiltonian(metric, l, dldθ)
```

We use the NUTS sampler with an acceptance ratio of δ= 0.45 in this example. In addition, we use Nesterov Dual Averaging for the Step Size adaptation.

We sample using 500 warmup samples and 500 posterior samples.

```@example bnode
integrator = Leapfrog(find_good_stepsize(h, p))
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.45, integrator))
samples, stats = sample(h, kernel, p, 500, adaptor, 500; progress = true)
```

## Step 5: Plot diagnostics

Now let's make sure the fit is good. This can be done by looking at the chain mixing plot
and the autocorrelation plot. First, let's create the chain mixing plot using the plot
recipes from ????

```@example bnode
samples = hcat(samples...)
samples_reduced = samples[1:5, :]
samples_reshape = reshape(samples_reduced, (500, 5, 1))
Chain_Spiral = Chains(samples_reshape)
plot(Chain_Spiral)
```

Now we check the autocorrelation plot:

```@example bnode
autocorplot(Chain_Spiral)
```

As another diagnostic, let's check the result on retrodicted data. To do this, we generate
solutions of the Neural ODE on samples of the neural network parameters, and check the
results of the predictions against the data. Let's start by looking at the time series:

```@example bnode
pl = scatter(tsteps, ode_data[1, :], color = :red, label = "Data: Var1", xlabel = "t",
             title = "Spiral Neural ODE")
scatter!(tsteps, ode_data[2, :], color = :blue, label = "Data: Var2")
for k in 1:300
    newp = typeof(p)(samples[:, 100:end][:, rand(1:400)])
    resol = predict_neuralode(newp)
    plot!(tsteps, resol[1, :], alpha = 0.04, color = :red, label = "")
    plot!(tsteps, resol[2, :], alpha = 0.04, color = :blue, label = "")
end

losses = map(x -> loss_neuralode(x)[1], eachcol(samples))
idx = findmin(losses)[2]
prediction = predict_neuralode(samples[:, idx])
plot!(tsteps, prediction[1, :], color = :black, w = 2, label = "")
plot!(tsteps, prediction[2, :], color = :black, w = 2, label = "Best fit prediction",
      ylims = (-2.5, 3.5))
```

That showed the time series form. We can similarly do a phase-space plot:

```@example bnode
pl = scatter(ode_data[1, :], ode_data[2, :], color = :red, label = "Data", xlabel = "Var1",
             ylabel = "Var2", title = "Spiral Neural ODE")
for k in 1:300
    newp = typeof(p)(samples[:, 100:end][:, rand(1:400)])
    resol = predict_neuralode(newp)
    plot!(resol[1, :], resol[2, :], alpha = 0.04, color = :red, label = "")
end
plot!(prediction[1, :], prediction[2, :], color = :black, w = 2,
      label = "Best fit prediction", ylims = (-2.5, 3))
```
