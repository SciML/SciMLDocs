using SciMLDocs, Documenter
using Plots

# Ordering Matters!
docsmodules = [
              "Equation Solvers" => ["LinearSolve", "NonlinearSolve", "Integrals", "Optimization", "NeuralOperators"],
              "Modeling Tools" => ["ModelingToolkit", "ModelingToolkitStandardLibrary", "Catalyst"],
              "Inverse Problems" => ["DiffEqSensitivity", "DiffEqParamEstim", ],
              "AbstractArray Libraries" => [],
              "Uncertainty Quantification" => [],
              "Symbolic Analysis" => [],
              "Interfaces" => ["SciMLBase"],
              "Utilities" => ["GlobalSensitivity", "Surrogates"],
              "Machine Learning" => ["DiffEqFlux"],
              "Developer Documentation" => [],
]

catpagestarts = [
    Any["highlevels/equation_solvers.md"],
    Any["highlevels/modeling_tools.md"],
    Any["highlevels/inverse_problems.md"],
    Any["highlevels/abstractarray_libraries.md"],
    Any["highlevels/uncertainty_quantification.md"],
    Any["highlevels/symbolic_analysis.md"],
    Any["highlevels/interfaces.md"],
    Any["highlevels/utilities.md"],
    Any["highlevels/machine_learning.md"],
    Any["highlevels/developer_documentation.md"],
]

# Omitted for now:

# Interfaces => CommonSolve SciMLOperators
# Solvers => DifferentialEquations FEniCS DiffEqOperators HighDimPDE NeuralPDE, MethodOfLines DiffEqJump
# ModelingTools => NBodySimulator ParameterizedFunctions 
# Inverse Problems =>  DiffEqBayes MinimallyDisruptiveCurves
# AbstractArray Libraries => MultiScaleArrays, LabelledArrays, RecursiveArrayTools
# Utilities => ExponentialUtilities QuasiMonteCarlo PoissonRandom DiffEqNoiseProcess
# Uncertainty Quantification => DiffEqUncertainty PolyChaos
# Symbolic Analysis => StructuralIdentifiability SymbolicNumericIntegration
# Machine Learning => ReservoirComputing  FastDEQ
# Extra Documentation => SciMLStyle COLPRAC 

fullpages = Any[["The SciML Open Souce Software Ecosystem" => "index.md"]]
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
        ex = quote
            using $(Symbol(mod))
            cp(joinpath(pkgdir($(Symbol(mod))),"docs","src"),joinpath(pkgdir(SciMLDocs),"docs","src","modules",$mod),force=true)
            include(joinpath(pkgdir($(Symbol(mod))),"docs","pages.jl"))
            push!(allmods,$(Symbol(mod)))
            push!(catpage,$mod => recursive_append(pages,joinpath("modules",$mod)))
        end
        @eval $ex
    end
    push!(fullpages, cat[1] => catpage)
end

push!(allmods,Plots)

makedocs(
    sitename="SciML",
    authors="Chris Rackauckas",
    modules=identity.(allmods),
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://docs.sciml.ai/stable/"),
    pages=fullpages
)

deploydocs(;
    repo="github.com/SciML/SciMLDocs.jl",
    devbranch="main",
)
