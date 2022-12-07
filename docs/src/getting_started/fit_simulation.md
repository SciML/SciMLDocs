# [Fit a simulation to a dataset](@id fit_simulation)

```@example odefit
using DifferentialEquations, Optimization, OptimizationPolyalgorithms, OptimizationOptimJL,
      SciMLSensitivity, Zygote, Plots

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5())

# Plot the solution
using Plots
plot(sol)
savefig("LV_ode.png")

function loss(p)
  sol = solve(prob, Tsit5(), p=p, saveat = tsteps)
  loss = sum(abs2, sol.-1)
  return loss, sol
end

callback = function (p, l, pred)
  display(l)
  plt = plot(pred, ylim = (0, 6))
  display(plt)
  # Tell Optimization.solve to not halt the optimization. If return true, then
  # optimization stops.
  return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)

result_ode = Optimization.solve(optprob, PolyOpt(),
                                callback = callback,
                                maxiters = 100)
```
