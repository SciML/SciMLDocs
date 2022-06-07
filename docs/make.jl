using SciMLDocs, Documenter, LibGit2, Pkg

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

#Pkg.develop(url="https://github.com/SciML/DiffEqDocs.jl")

# Ordering Matters!
docsmodules = [
              "Equation Solvers" => ["LinearSolve", "NonlinearSolve", "Integrals", "Optimization"],
              "Partial Differential Equation Solvers" => ["MethodOfLines", "NeuralPDE", "NeuralOperators", 
                                                          "FEniCS", "DiffEqOperators"],
              "Modeling Tools" => ["ModelingToolkit", "ModelingToolkitStandardLibrary", "Catalyst", 
                                   "NBodySimulator", "ParameterizedFunctions"],
              "Inverse Problems" => ["DiffEqSensitivity", "DiffEqParamEstim"],
              "AbstractArray Libraries" => ["RecursiveArrayTools", "LabelledArrays", "MultiScaleArrays"],
              "Uncertainty Quantification" => ["PolyChaos"],
              "Simulation Analysis" => ["GlobalSensitivity"],
              "Symbolic Analysis" => ["SymbolicNumericIntegration"],
              "Interfaces" => ["SciMLBase", "SciMLOperators", "CommonSolve"],              
              "Numerical Utilities" => ["Surrogates", "ExponentialUtilities", "DiffEqNoiseProcess", 
                                        "PoissonRandom", "QuasiMonteCarlo"],
              "Machine Learning" => ["DiffEqFlux"],
              "Learning Resources" => [],
              "Developer Documentation" => ["SciMLStyle", "COLPRAC"],
]

usereadme = ["FEniCS", "NBodySimulator", "SymbolicNumericIntegration", "SciMLStyle", "COLPRAC"]

catpagestarts = [
    Any["highlevels/equation_solvers.md"],
    Any["highlevels/partial_differential_equation_solvers.md"],
    Any["highlevels/modeling_tools.md"],
    Any["highlevels/inverse_problems.md"],
    Any["highlevels/abstractarray_libraries.md"],
    Any["highlevels/uncertainty_quantification.md"],
    Any["highlevels/simulation_analysis.md"],
    Any["highlevels/symbolic_analysis.md"],
    Any["highlevels/interfaces.md"],
    Any["highlevels/numerical_utilities.md"],
    Any["highlevels/machine_learning.md"],
    Any["highlevels/learning_resources.md"],
    Any["highlevels/developer_documentation.md"],
]

# Omitted for now:

# Interfaces => SciMLParameters
# Solvers => DifferentialEquations DiffEqJump
# Partial Differential Equation Solvers =>  HighDimPDE 
# Inverse Problems =>  DiffEqBayes
# Simulation Analysis => MinimallyDisruptiveCurves
# Uncertainty Quantification => DiffEqUncertainty 
# Symbolic Analysis => StructuralIdentifiability 
# Machine Learning => ReservoirComputing DeepEquilibriumNetworks
# DiffEqDevDocs

fullpages = Any["The SciML Open Souce Software Ecosystem" => "index.md"]
allmods = Vector{Any}()

function recursive_append(pages::AbstractArray{<:AbstractArray},str)
    map(recursive_append,pages,str)
end

function recursive_append(pages::AbstractArray{<:Pair{String,Any}},str)
    for i in eachindex(pages)
        if pages[i][2] isa AbstractArray
            pages[i] = pages[i][1] => recursive_append(pages[i][2],str)
        elseif pages[i][2] isa String
            pages[i] = pages[i][1] => joinpath(str,pages[i][2])
        end
    end
    pages
end

function recursive_append(pages::AbstractArray{<:String},str)
    for i in eachindex(pages)
        pages[i] = joinpath(str,pages[i])
    end
    pages
end

function recursive_append(pages::AbstractArray{<:Pair{String,String}},str)
    for i in eachindex(pages)
        pages[i] = pages[i][1] => joinpath(str,pages[i][2])
    end
    pages
end

function recursive_append(pages::AbstractArray{<:Any},str)
    for i in eachindex(pages)
        if pages[i] isa Pair && pages[i][2] isa String
            pages[i] = pages[i][1] => joinpath(str,pages[i][2])
        elseif pages[i] isa Pair && pages[i][2] isa AbstractArray
            pages[i] = pages[i][1] => recursive_append(pages[i][2],str)
        elseif pages[i] isa String
            pages[i] = joinpath(str,pages[i])
        else
            error("wait what?")
        end
    end
    pages
end

for (i,cat) in enumerate(docsmodules)
    global catpage
    catpage = catpagestarts[i]

    for mod in cat[2]
        if mod in usereadme
            dir = joinpath(pkgdir(SciMLDocs),"docs","src","modules",mod)
            mkdir(dir)
            mkdir(mod)
            LibGit2.clone("https://github.com/SciML/$mod", mod)
            cp(joinpath(mod,"README.md"),joinpath(dir,"index.md"),force=true)
            push!(catpage,mod => Any[joinpath("modules",mod,"index.md")])
        else
            ex = quote
                using $(Symbol(mod))
                cp(joinpath(pkgdir($(Symbol(mod))),"docs","src"),joinpath(pkgdir(SciMLDocs),"docs","src","modules",$mod),force=true)
                include(joinpath(pkgdir($(Symbol(mod))),"docs","pages.jl"))
                push!(allmods,$(Symbol(mod)))

                if $mod == "DiffEqDocs"
                    push!(catpage,"DifferentialEquations" => recursive_append(pages,joinpath("modules",$mod)))
                elseif $mod == "DiffEqDevDocs"
                    push!(catpage,"*DiffEq Developer Documentation" => recursive_append(pages,joinpath("modules",$mod)))
                else
                    push!(catpage,$mod => recursive_append(pages,joinpath("modules",$mod)))
                end
            end
            @eval $ex
        end
    end
    push!(fullpages, cat[1] => catpage)
end

@show fullpages

push!(allmods,Plots)

mathengine = MathJax3(Dict(
    :loader => Dict("load" => ["[tex]/require","[tex]/mathtools"]),
    :tex => Dict(
        "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
        "packages" => ["base", "ams", "autoload", "mathtools", "require"],
    ),
))

makedocs(
    sitename="SciML",
    authors="The SciML Open Source Software Organization Contributors",
    modules=identity.(allmods),
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             mathengine = mathengine,
                             canonical="https://docs.sciml.ai/stable/"),
    pages=fullpages
)

deploydocs(;
    repo="github.com/SciML/SciMLDocs.jl",
    devbranch="main",
)
