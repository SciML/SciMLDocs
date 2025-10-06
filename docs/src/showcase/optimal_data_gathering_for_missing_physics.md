# [Optimal Data Gathering for Missing Physics](@id srmf)

[The missing physics showcase](@ref autocomplete)
teaches how to discover the missing parts of a dynamic model, using universal differential equations (UDE) and symbolic regression (SR).

High quality data is needed to ensure the true dynamics are recovered.
In this tutorial, we look at an efficient data gathering technique for SciML models,
using a bioreactor example.
To this end, we will rely on the following packages:
```@example DoE
using Random; Random.seed!(984519674645)
using StableRNGs; rng = StableRNG(845652695)
import ModelingToolkit as MTK
import ModelingToolkit: t_nounits as t, D_nounits as D, @mtkmodel, @mtkcompile, mtkcompile
using ModelingToolkit
import ModelingToolkitNeuralNets
import OrdinaryDiffEqRosenbrock as ODE
import SymbolicIndexingInterface
using Plots
import Optimization as OPT
import OptimizationOptimisers as OptOptim
import OptimizationBBO as OptBBO
import OptimizationNLopt as OptNL
import SciMLStructures
import SciMLStructures: Tunable
import SciMLSensitivity as SMS
using Statistics
using SymbolicRegression
using LuxCore
using LuxCore: stateless_apply
using Lux
using Statistics
using DataFrames
nothing # hide
```

The bioreactor consists of 3 states: substrate concentration $C_s(t)$, biomass concentration $C_x(t)$ and volume $V(t)$.
```math
\begin{aligned}
\frac{dC_s}{dt} &= -\left(\frac{\mu(C_s)}{y_{x,s}} + m\right) C_x + \frac{Q_{in}(t)}{V}(C_{S,in} - C_s),\\
\frac{dC_x}{dt} &= \mu(C_s) C_x - \frac{Q_{in}(t)}{V}C_x,\\
\frac{dV}{dt} &= Q_{in}(t).
\end{aligned}
```
The substrate is eaten by the biomass, causing the biomass to grow.
The rate by which the biomass grows $μ(t)$ is an unknown function (missing physics),
which must be estimated from experimental data.
The rate by which the substrate is consumed $σ(t)$ is dependent on $μ(t)$, trough a yield factor $y_{x:s}$ and a maintenance term $m$,
where are assumed to be known parameters.
More substrate can be pumped into the reactor  with pumping speed $Q_{in}(t)$.
This pumped substrate has known concentration $C_{s_{in}}$.
The goal is to optimize the control action $Q_{in}(t)$, such that $μ(t)$ can be estimated as precisely as possible.
We restrict $Q_{in}(t)$ to piecewise constant functions.
This can be implemented in MTK as:
```@example DoE
@mtkmodel Bioreactor begin
    @constants begin
        C_s_in = 50.0
        y_x_s = 0.777
        m = 0.0
    end
    @parameters begin
        controls[1:length(optimization_state)-1] = optimization_state[2:end], [tunable = false] # optimization_state is defined further below
        Q_in = optimization_initial, [tunable = false] # similar for optimization state
    end
    @variables begin
        C_s(t) = 1.0
        C_x(t) = 1.0
        V(t) = 7.0
        μ(t)
        σ(t)
    end
    @equations begin
        σ ~ μ / y_x_s + m
        D(C_s) ~ -σ * C_x + Q_in / V * (C_s_in - C_s)
        D(C_x) ~ μ * C_x - Q_in / V * C_x
        D(V) ~ Q_in
    end
    @discrete_events begin
        (t == 1.0) => [Q_in ~ controls[1]]
        (t == 2.0) => [Q_in ~ controls[2]]
        (t == 3.0) => [Q_in ~ controls[3]]
        (t == 4.0) => [Q_in ~ controls[4]]
        (t == 5.0) => [Q_in ~ controls[5]]
        (t == 6.0) => [Q_in ~ controls[6]]
        (t == 7.0) => [Q_in ~ controls[7]]
        (t == 8.0) => [Q_in ~ controls[8]]
        (t == 9.0) => [Q_in ~ controls[9]]
        (t == 10.0) => [Q_in ~ controls[10]]
        (t == 11.0) => [Q_in ~ controls[11]]
        (t == 12.0) => [Q_in ~ controls[12]]
        (t == 13.0) => [Q_in ~ controls[13]]
        (t == 14.0) => [Q_in ~ controls[14]]
        (t == 15.0) => [Q_in ~ optimization_initial] # HACK TO GET Q_IN BACK TO ITS ORIGINAL VALUE
    end
end
nothing # hide
```

The true value of $μ(t)$, which must be recovered is the Monod equation.
```math
\begin{equation*}
\mu(C_s) = \frac{\mu_{max}C_s}{K_s + C_s}.
\end{equation*}
```
We thus extend the bioreactor MTK model with this equation:
```@example DoE
@mtkmodel TrueBioreactor begin
    @extend Bioreactor()
    @parameters begin
        μ_max = 0.421
        K_s = 0.439*10
    end
    @equations begin
        μ ~ μ_max * C_s / (K_s + C_s) 
    end
end
nothing # hide
```

Similarly, we can extend the bioreactor with a neural network to represent this missing physics.
```@example DoE
@mtkmodel UDEBioreactor begin
    @extend Bioreactor()
    @structural_parameters begin
        chain = Lux.Chain(Lux.Dense(1, 5, tanh),
                          Lux.Dense(5, 5, tanh),
                          Lux.Dense(5, 1, x->1*sigmoid(x)))
    end
    @components begin
        nn = NeuralNetworkBlock(; n_input=1, n_output=1, chain, rng)
    end
    @equations begin
        nn.output.u[1] ~ μ
        nn.input.u[1] ~ C_s
    end
end
nothing # hide
```

We start by gathering some initial data.
Because we don't yet know anything about the missing physics,
we arbitrarily pick the zero control action.
The only state we measure is $C_s$
We also add some noise to the simulated data, to make it more realistic:
```@example DoE
optimization_state =  zeros(15)
optimization_initial = optimization_state[1] # HACK CAN'T GET THIS TO WORK WITHOUT SEPARATE SCALAR
@mtkcompile true_bioreactor = TrueBioreactor()
prob = ODE.ODEProblem(true_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
sol = ODE.solve(prob, ODE.Rodas5P())

@mtkcompile  ude_bioreactor = UDEBioreactor()
ude_prob = ODE.ODEProblem(ude_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
ude_sol = ODE.solve(ude_prob, ODE.Rodas5P())

data = DataFrame(sol)
data = data[1:2:end, :] # HACK TO GET ONLY THE MEASUREMENTS WE NEED; MTK ALWAYS SAVES BEFORE AND AFTER CALLBACK; WITH NO OPTION TO DISABLE

sd_cs = 0.1
data[!, "C_s(t)"] += sd_cs * randn(size(data, 1))

plts = plot(), plot(), plot(), plot()
plot!(plts[1], sol, idxs=:C_s, lw=3,c=1)
plot!(plts[1], ylabel="Cₛ(g/L)", xlabel="t(h)")
scatter!(plts[1], data[!, "timestamp"], data[!, "C_s(t)"]; ms=3,c=1)
plot!(plts[2], sol, idxs=:C_x, lw=3,c=1)
plot!(plts[2], ylabel="Cₓ(g/L)", xlabel="t(h)")
plot!(plts[3], sol, idxs=:V, ylabel="V(L)", xlabel="t(h)", lw=3, color=:black, ylims=(6.0,8.0))
C_s_range_plot = 0.0:0.01:50.0
μ_max = 0.421; K_s = 0.439*10 # TODO extract the  values from the model.
plot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)
plot!(plts[4], ylabel="μ(1/h)", xlabel="Cₛ(g/L)",ylims=(0,0.5))
plot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
```

Now we can train the neural network to match this data:
```@example DoE
function loss(x, (probs, get_varss, datas))
    loss = zero(eltype(x))
    for i in eachindex(probs)
        prob = probs[i]
        get_vars = get_varss[i]
        data = datas[i]
        new_p = SciMLStructures.replace(Tunable(), prob.p, x)
        new_prob = remake(prob, p=new_p, u0=eltype(x).(prob.u0))
        new_sol = ODE.solve(new_prob, ODE.Rodas5P())
        for (i, j) in enumerate(1:2:length(new_sol.t)) # HACK TO DEAL WITH DOUBLE SAVE
            loss += sum(abs2.(get_vars(new_sol, j) .- data[!, "C_s(t)"][i]))
        end
        if !(SciMLBase.successful_retcode(new_sol))
            println("failed")
            return Inf
        end
    end
    loss
end
of = OPT.OptimizationFunction{true}(loss, SMS.AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))
get_vars = getu(ude_bioreactor, [ude_bioreactor.C_s])
ps = ([ude_prob], [get_vars], [data]);
op = OPT.OptimizationProblem(of, x0, ps)
res = OPT.solve(op, OptOptim.LBFGS(), maxiters=1000)

new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob = remake(ude_prob, p=new_p)
res_sol = ODE.solve(res_prob, ODE.Rodas5P())

extracted_chain = arguments(equations(ude_bioreactor.nn)[1].rhs)[1]
T = defaults(ude_bioreactor)[ude_bioreactor.nn.T]
μ_predicted_plot = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]
μ_predicted_data = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]

plts = plot(), plot(), plot(), plot()
plot!(plts[1], sol, idxs=:C_s, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)
plot!(plts[1], ylabel="Cₛ(g/L)", xlabel="t(h)")
scatter!(plts[1], data[!, "timestamp"], data[!, "C_s(t)"]; ms=3,c=1)
plot!(plts[2], sol, idxs=:C_x, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)
plot!(plts[2], ylabel="Cₓ(g/L)", xlabel="t(h)")
plot!(plts[3], sol, idxs=:V, ylabel="V(L)", xlabel="t(h)", lw=3, color=:black, ylims=(6.0,8.0))
plot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)
plot!(plts[4], C_s_range_plot, μ_predicted_plot, lw=3, c=2)
scatter!(plts[4], data[!, "C_s(t)"], μ_predicted_data, ms=3, c=2)
plot!(plts[4], ylabel="μ(1/h)", xlabel="Cₛ(g/L)",ylims=(0,0.5))
plot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
```
On the above figure we see that the neural network predicts $C_s$ well, except during the final hours of the experiment,
where we have multiple positive realizations of the noise in a row.
The neural network also predicts $µ$ well in the low substrate concentration region,
where we have data available.
However, the fit is poor at higher substrate concentrations,
where we do not have data.

We continue by making the neural network interpretable using symbolic regression.
```@example DoE
options = SymbolicRegression.Options(
    unary_operators=(exp, sin, cos),
    binary_operators=(+, *, /, -),
    seed=123,
    deterministic=true,
    save_to_file=false,
    defaults=v"0.24.5"
)
hall_of_fame = equation_search(collect(data[!, "C_s(t)"])', μ_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)
```
Next, we extract the 10 model structures which symbolic regression thinks are best,
and predict the system with them.
```@example DoE
n_best = 10
function get_model_structures(hall_of_fame, options, n_best)
    best_models = []
    best_models_scores = []
    i = 1
    round(hall_of_fame.members[i].loss,sigdigits=5)
    while length(best_models) <= n_best
        member = hall_of_fame.members[i]
        rounded_score = round(member.loss, sigdigits=5)
        if !(rounded_score in best_models_scores)
            push!(best_models,member)
            push!(best_models_scores, rounded_score)
        end
        i += 1
    end
    model_structures = []
    @syms x
    for i = 1:n_best
        eqn = node_to_symbolic(best_models[i].tree, options, varMap=["x"])
        fi = build_function(eqn, x, expression=Val{false})
        push!(model_structures, fi)
    end
    return model_structures
end

function get_probs_and_caches(model_structures)
    probs_plausible = Array{Any}(undef, length(model_structures))
    syms_cache = Array{Any}(undef, length(model_structures))
    i = 1
    for i in 1:length(model_structures)
        @mtkmodel PlausibleBioreactor begin
            @extend Bioreactor()
            @equations begin
                μ ~ model_structures[i](C_s)
            end
        end
        @mtkcompile plausible_bioreactor = PlausibleBioreactor()
        plausible_prob = ODE.ODEProblem(plausible_bioreactor, [], (0.0, 15.0), [], tstops=0:15, saveat=0:15)
        probs_plausible[i] = plausible_prob

        callback_controls = plausible_bioreactor.controls
        initial_control = plausible_bioreactor.Q_in

        syms_cache[i] = (callback_controls, initial_control, plausible_bioreactor.C_s)
    end
    probs_plausible, syms_cache
end
model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures)

plts = plot(), plot(), plot(), plot()
for i in 1:length(model_structures)
    plot!(plts[4],  C_s_range_plot, model_structures[i].( C_s_range_plot);c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = ODE.solve(plausible_prob, ODE.Rodas5P())
    # plot!(sol_plausible; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    plot!(plts[1], sol_plausible, idxs=:C_s, lw=1,ls=:dash,c=i+2)
    plot!(plts[2], sol_plausible, idxs=:C_x, lw=1,ls=:dash,c=i+2)
end
plot!(plts[1], sol, idxs=:C_s, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)
plot!(plts[1], ylabel="Cₛ(g/L)", xlabel="t(h)")
scatter!(plts[1], data[!, "timestamp"], data[!, "C_s(t)"]; ms=3,c=1)
plot!(plts[2], sol, idxs=:C_x, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)
plot!(plts[2], ylabel="Cₓ(g/L)", xlabel="t(h)")
plot!(plts[3], sol, idxs=:V, ylabel="V(L)", xlabel="t(h)", lw=3, color=:black, ylims=(6.0,8.0))
μ_max = 0.421; K_s = 0.439*10 # TODO extract the  values from the model.
plot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)
plot!(plts[4], C_s_range_plot, μ_predicted_plot, lw=3, c=2)
scatter!(plts[4], data[!, "C_s(t)"], μ_predicted_data, ms=3, c=2)
plot!(plts[4], ylabel="μ(1/h)", xlabel="Cₛ(g/L)",ylims=(0,0.5))
plot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
```
On the figure, we see that most plausible model structures predict the states $C_s$ and $C_x$ well, similar to the neural network.
The plausible model structures also fit $\mu$ well in the low $C_s$ region, but not outside this region.
One group of the structures predicts that $\mu$ keeps increasing as $C_s$ becomes large
while another group predicts that $\mu$ stays below $0.1$ $1/\mathrm{h}$.

We now design a second experiment to start discriminating between these plausible model structures,
using the following criterion:
```math
\begin{equation*}
\argmax_{\bm Q_{in}} \frac{2!(10-2)!}{10!}\sum_{i=1}^{10} \sum_{j=i+1}^{10} \max_{t_k} (\bm C_s^i(t_k) - \bm C_s^j(t_k))^2.
\end{equation*}
```
In this equation, $C_s^i$ denotes the predicted substrate concentration for the i'th plausible model structure.
The distance between two model structures is scored by the maximal squared difference between the two structures at the measurement times.
The criterion then calculates the average distance between all model structures.
Collecting measurements where the plausible model structures differ greatly in predictions,
will cause at least some of the model structures to become unlikely,
and thus cause new model structures to enter the top 10 plausible model structures.
```@example DoE
function S_criterion(optimization_state, (probs_plausible, syms_cache))
    n_structures = length(probs_plausible)
    sols = Array{Any}(undef, n_structures)
    for i in 1:n_structures
        plausible_prob = probs_plausible[i]
        callback_controls, initial_control, C_s = syms_cache[i]
        plausible_prob.ps[callback_controls] = optimization_state[2:end]
        plausible_prob.ps[initial_control] = optimization_state[1]
        sol_plausible = ODE.solve(plausible_prob, ODE.Rodas5P())
        if !(SciMLBase.successful_retcode(sol_plausible))
            return 0.0
        end
    loss
        sols[i] = sol_plausible
    end
    squared_differences = Float64[]
    for i in 1:n_structures
        callback_controls, initial_control, C_s = syms_cache[i]
        for j in i+1:n_structures
            push!(squared_differences, maximum((sols[i][C_s] .- sols[j][C_s]) .^ 2))
        end
    end
    ret = -mean(squared_differences)
    println(ret)
    return ret
end
lb = zeros(15)
ub = 10 * ones(15)

design_prob = OPT.OptimizationProblem(S_criterion, optimization_state, (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = OPT.solve(design_prob, OptBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=100.0)

optimization_state = control_pars_opt.u
optimization_initial = optimization_initial2 = optimization_state[1]

plts = plot(), plot()
t_pwc = []
pwc = []
for i in 0:14
    push!(t_pwc,i)
    push!(t_pwc,i+1)
    push!(pwc,optimization_state[i+1])
    push!(pwc,optimization_state[i+1])
end
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t(h)",ylabel="Qin(L/h)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, C_s = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = ODE.solve(plausible_prob, ODE.Rodas5P())
    plot!(plts[2], sol_plausible, idxs=:C_s, lw=3,ls=:dash,c=i+2)
end
plot!(plts[2],xlabel="t(h)",ylabel="Cₛ(g/L)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
```
The above figure shows that a maximal control action is generally preferred.
This causes the two aforementioned groups in the model structures to be easily discriminated from one another.

We now gather a second dataset and perform the same exercise.
```@example DoE
@mtkcompile true_bioreactor2 = TrueBioreactor()
prob2 = ODE.ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol2 = ODE.solve(prob2, ODE.Rodas5P())
@mtkcompile ude_bioreactor2 = UDEBioreactor()
ude_prob2 = ODE.ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [ude_bioreactor2.Q_in => optimization_initial], tstops=0:15, save_everystep=false)
ude_sol2 = ODE.solve(ude_prob2, ODE.Rodas5P())
plot(ude_sol2[3,:])
ude_prob_remake = remake(ude_prob, p=ude_prob2.p)
sol_remake = ODE.solve(ude_prob_remake, ODE.Rodas5P())
plot(sol_remake[3,:])
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))

get_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])

data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "C_s(t)"] += sd_cs * randn(size(data2, 1))

ps = ([ude_prob, ude_prob2], [get_vars, get_vars2], [data, data2]);
op = OPT.OptimizationProblem(of, x0, ps)
res = OPT.solve(op, OptNL.NLopt.LN_BOBYQA, maxiters=5_000)

new_p = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)
res_prob = remake(ude_prob2, p=new_p)
callback_controls, initial_control, C_s = syms_cache[1]
res_prob.ps[initial_control] = optimization_initial2
res_sol = ODE.solve(res_prob, ODE.Rodas5P())
extracted_chain = arguments(equations(ude_bioreactor2.nn)[1].rhs)[1]
T = defaults(ude_bioreactor2)[ude_bioreactor2.nn.T]
μ_predicted_plot2 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]

μ_predicted_data = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]
μ_predicted_data2 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]

total_data = hcat(collect(data[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data, μ_predicted_data2)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)
model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot(), plot()
for i in 1:length(model_structures)
    plot!(plts[4],  C_s_range_plot, model_structures[i].( C_s_range_plot);c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = ODE.solve(plausible_prob, ODE.Rodas5P())
    # plot!(sol_plausible; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    plot!(plts[1], sol_plausible, idxs=:C_s, lw=1,ls=:dash,c=i+2)
    plot!(plts[2], sol_plausible, idxs=:C_x, lw=1,ls=:dash,c=i+2)
end
plot!(plts[1], sol2, idxs=:C_s, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)
plot!(plts[1], ylabel="Cₛ(g/L)", xlabel="t(h)")
scatter!(plts[1], data2[!, "timestamp"], data2[!, "C_s(t)"]; ms=3,c=1)
plot!(plts[2], sol2, idxs=:C_x, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)
plot!(plts[2], ylabel="Cₓ(g/L)", xlabel="t(h)")
plot!(plts[3], sol2, idxs=:V, ylabel="V(L)", xlabel="t(h)", lw=3, color=:black)
plot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)
plot!(plts[4], C_s_range_plot, μ_predicted_plot2, lw=3, c=2)
scatter!(plts[4], data[!, "C_s(t)"], μ_predicted_data, ms=3, c=2)
scatter!(plts[4], data2[!, "C_s(t)"], μ_predicted_data2, ms=3, c=2)
plot!(plts[4], ylabel="μ(1/h)", xlabel="Cₛ(g/L)",ylims=(0,0.5))
plot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
```
The above shows the data analysis corresponding to this second experiment.
Both the UDE and most of the plausible model structures predict the states well,

The UDE and the plausible model structures also approximate the missing physics $\mu$ well in the region where we have gathered data.
This means in the regions of low substrate concentration,
with data coming primarily from the first experiment,
and high substrate concentration, coming from the second experiment.
However, we do not have any measurements at substrate concentrations between these two groups.
This causes there to be substantial disagreement between the plausible model structures in the medium substrate concentration range.

We now optimize the controls for a third experiment:
```@example DoE
prob = OPT.OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = OPT.solve(prob, OptBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)

optimization_state = control_pars_opt.u
optimization_initial = optimization_state[1]

plts = plot(), plot()
t_pwc = []
pwc = []
for i in 0:14
    push!(t_pwc,i)
    push!(t_pwc,i+1)
    push!(pwc,optimization_state[i+1])
    push!(pwc,optimization_state[i+1])
end
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t(h)",ylabel="Qin(L/h)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, C_s = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = ODE.solve(plausible_prob, ODE.Rodas5P())
    plot!(plts[2], sol_plausible, idxs=:C_s, lw=3,ls=:dash,c=i+2)
end
plot!(plts[2],xlabel="t(h)",ylabel="Cₛ(g/L)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
```

The optimal design algorithm is also aware of this uncertainty at the medium concentration range,
and aims to remedy this in the next experiment, as can be seen on the above figure.
Using the first control action, the bioreactor substrate concentration gets pumped from a low substrate concentration level to a medium level.
At this level, there is substantial disagreement between the plausible model structures, leading to substantial disagreement in predicted substrate concentrations.
To keep the reactor at the medium substrate concentration range, while the biomass concentration increases rapidly,
an increasing amount of substrate has to be pumped into the reactor every hour.
This explains the staircase with increasing step heights form of the control function.
After the staircase reaches the maximal control value, a zero control is used.
Some model structures decrease more rapidly in substrate concentration than others.

```@example DoE
@mtkcompile true_bioreactor3 = TrueBioreactor()
prob3 = ODE.ODEProblem(true_bioreactor3, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol3 = ODE.solve(prob3, ODE.Rodas5P())
@mtkcompile ude_bioreactor3 = UDEBioreactor()
ude_prob3 = ODE.ODEProblem(ude_bioreactor3, [], (0.0, 15.0), tstops=0:15, save_everystep=false)

x0 = reduce(vcat, getindex.((default_values(ude_bioreactor3),), tunable_parameters(ude_bioreactor3)))

get_vars3 = getu(ude_bioreactor3, [ude_bioreactor3.C_s])

data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "C_s(t)"] += sd_cs * randn(size(data3, 1))

ps = ([ude_prob, ude_prob2, ude_prob3], [get_vars, get_vars2, get_vars3], [data, data2, data3]);
op = OPT.OptimizationProblem(of, x0, ps)
res = OPT.solve(op, OptOptim.LBFGS(), maxiters=10_000)
extracted_chain = arguments(equations(ude_bioreactor3.nn)[1].rhs)[1]
T = defaults(ude_bioreactor3)[ude_bioreactor3.nn.T]

μ_predicted_data = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]
μ_predicted_data2 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
μ_predicted_data3 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data3[!, "C_s(t)"]]

total_data = hcat(collect(data[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'), collect(data3[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data, μ_predicted_data2, μ_predicted_data3)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)
bar(i->hall_of_fame.members[i].loss, 1:10, ylabel="loss", xlabel="hall of fame member", xticks=1:10)
plot!(tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
```
The Monod equation $(0.419 / ((x1 + 4.300) / x1))$ is member 7 of the hall of fame.
All hall of fame members before it have visually higher loss,
while all the members after it are indiscernible from it.
This indicates that it is a good candidate for the true model structure.

Symbolic regression sometimes finds the true model structure in a somewhat unusual form,
like with a double division.
This is because symbolic regression considers multiplication and division to have the same complexity.

In this tutorial, we have shown that experimental design can be used to explore the state space of a dynamic system in a thoughtful way,
such that missing physics can be recovered in an efficient manner.
