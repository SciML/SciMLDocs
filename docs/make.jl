using SciMLDocs, Documenter, LibGit2, Pkg

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

using DiffEqProblemLibrary, OrdinaryDiffEq

ODEProblemLibrary = DiffEqProblemLibrary.ODEProblemLibrary
ODEProblemLibrary.importodeproblems()

SDEProblemLibrary = DiffEqProblemLibrary.SDEProblemLibrary
SDEProblemLibrary.importsdeproblems()

DDEProblemLibrary = DiffEqProblemLibrary.DDEProblemLibrary
DDEProblemLibrary.importddeproblems()

DAEProblemLibrary = DiffEqProblemLibrary.DAEProblemLibrary
DAEProblemLibrary.importdaeproblems()

using DiffEqDevTools # Needed for tableaus
using DiffEqBase

# Ordering Matters!
docsmodules = [
    "Equation Solvers" => ["LinearSolve", "NonlinearSolve", "DiffEqDocs", "Integrals",
                           "Optimization", "JumpProcesses"],
    "Partial Differential Equation Solvers" => ["MethodOfLines", "NeuralPDE",
                                                "NeuralOperators", "FEniCS",
                                                "DiffEqOperators"],
    "Modeling Tools" => ["ModelingToolkit", "ModelingToolkitStandardLibrary", "Catalyst",
        "NBodySimulator", "ParameterizedFunctions"],
    "Inverse Problems" => ["SciMLSensitivity", "DiffEqParamEstim", "DiffEqBayes"],
    "AbstractArray Libraries" => ["RecursiveArrayTools", "LabelledArrays", "MultiScaleArrays"],
    "Uncertainty Quantification" => ["PolyChaos"],
    "Simulation Analysis" => ["GlobalSensitivity"],
    "Symbolic Analysis" => ["StructuralIdentifiability", "SymbolicNumericIntegration"],
    "Interfaces" => ["SciMLBase", "SciMLOperators", "CommonSolve"],
    "Numerical Utilities" => ["Surrogates", "ExponentialUtilities", "DiffEqNoiseProcess",
        "PoissonRandom", "QuasiMonteCarlo", "DataInterpolations",
        "FFTW", "RuntimeGeneratedFunctions", "MuladdMacro",],
    "Machine Learning" => ["DiffEqFlux","DeepEquilibriumNetworks"],
    "Learning Resources" => [],
    "Developer Documentation" => ["SciMLStyle", "COLPRAC", "DiffEqDevDocs"],
]

docspackage = ["DiffEqDocs", "DiffEqDevDocs"]
docspackagenames = Dict("DiffEqDocs" => "DifferentialEquations",
                        "DiffEqDevDocs" => "DiffEq Developer Documentation")

usereadme = ["FEniCS", "SciMLStyle", "COLPRAC",
    "DataInterpolations", "FFTW", "RuntimeGeneratedFunctions", "MuladdMacro",
    "SBMLToolkit", "CellMLToolkit"]

readmeurls = Dict(
    "FEniCS" => "https://github.com/SciML/FEniCS.jl",
    "SciMLStyle" => "https://github.com/SciML/SciMLStyle",
    "COLPRAC" => "https://github.com/SciML/ColPrac",
    "DataInterpolations" => "https://github.com/PumasAI/DataInterpolations.jl",
    "RuntimeGeneratedFunctions" => "https://github.com/SciML/RuntimeGeneratedFunctions.jl",
    "MuladdMacro" => "https://github.com/SciML/MuladdMacro.jl",
    "FFTW" => "https://github.com/JuliaMath/FFTW.jl",
    "SBMLToolkit" => "https://github.com/SciML/SBMLToolkit.jl",
    "CellMLToolkit" => "https://github.com/SciML/CellMLToolkit.jl"
)


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
# Partial Differential Equation Solvers =>  HighDimPDE
# Simulation Analysis => MinimallyDisruptiveCurves
# Uncertainty Quantification => DiffEqUncertainty 
# Machine Learning => ReservoirComputing

fullpages = Any["The SciML Open Souce Software Ecosystem"=>"index.md"]
allmods = Vector{Any}()

function recursive_append(pages::AbstractArray{<:AbstractArray}, str)
    map(recursive_append, pages, str)
end

function recursive_append(pages::AbstractArray{<:Pair{String,Any}}, str)
    for i in eachindex(pages)
        if pages[i][2] isa AbstractArray
            pages[i] = pages[i][1] => recursive_append(pages[i][2], str)
        elseif pages[i][2] isa String
            pages[i] = pages[i][1] => joinpath(str, pages[i][2])
        end
    end
    pages
end

function recursive_append(pages::AbstractArray{<:String}, str)
    for i in eachindex(pages)
        pages[i] = joinpath(str, pages[i])
    end
    pages
end

function recursive_append(pages::AbstractArray{<:Pair{String,String}}, str)
    for i in eachindex(pages)
        pages[i] = pages[i][1] => joinpath(str, pages[i][2])
    end
    pages
end

function recursive_append(pages::AbstractArray{<:Any}, str)
    for i in eachindex(pages)
        if pages[i] isa Pair && pages[i][2] isa String
            pages[i] = pages[i][1] => joinpath(str, pages[i][2])
        elseif pages[i] isa Pair && pages[i][2] isa AbstractArray
            pages[i] = pages[i][1] => recursive_append(pages[i][2], str)
        elseif pages[i] isa String
            pages[i] = joinpath(str, pages[i])
        else
            error("wait what?")
        end
    end
    pages
end

for (i, cat) in enumerate(docsmodules)
    global catpage
    catpage = catpagestarts[i]

    for mod in cat[2]
        if mod in usereadme
            dir = joinpath(pkgdir(SciMLDocs), "docs", "src", "modules", mod)
            mkdir(dir)
            mkdir(mod)
            LibGit2.clone(readmeurls[mod], mod)
            cp(joinpath(mod, "README.md"), joinpath(dir, "index.md"), force=true)
            push!(catpage, mod => Any[joinpath("modules", mod, "index.md")])
        elseif mod in docspackage
            dir = joinpath(pkgdir(SciMLDocs), "docs", "src", "modules", mod)
            mkdir(dir)
            mkdir(mod)
            LibGit2.clone("https://github.com/SciML/$mod.jl", mod)

            cp(joinpath(mod, "docs", "pages.jl"), dir, force=true)
            include(joinpath(pwd(), mod, "docs", "pages.jl"))

            cp(joinpath(mod, "docs", "src"), dir, force=true)
            @show readdir(dir)
            push!(catpage, docspackagenames[mod] => recursive_append(pages, joinpath("modules", mod)))
        else
            ex = quote
                using $(Symbol(mod))
                cp(joinpath(pkgdir($(Symbol(mod))), "docs", "src"), joinpath(pkgdir(SciMLDocs), "docs", "src", "modules", $mod), force=true)
                include(joinpath(pkgdir($(Symbol(mod))), "docs", "pages.jl"))
                push!(allmods, $(Symbol(mod)))
                push!(catpage, $mod => recursive_append(pages, joinpath("modules", $mod)))
            end
            @eval $ex
        end
    end
    push!(fullpages, cat[1] => catpage)
end

@show fullpages

append!(allmods, [Plots, DiffEqBase, DiffEqDevTools, DiffEqProblemLibrary, ODEProblemLibrary,
                  SDEProblemLibrary, DDEProblemLibrary, DAEProblemLibrary, OrdinaryDiffEq])

mathengine = MathJax3(Dict(
    :loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
    :tex => Dict(
        "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
        "packages" => ["base", "ams", "autoload", "mathtools", "require"],
    ),
))

makedocs(
    sitename="SciML",
    authors="The SciML Open Source Software Organization Contributors",
    modules=identity.(allmods),
    clean=true, doctest=false,
    format=Documenter.HTML(analytics="UA-90474609-3",
        assets=["assets/favicon.ico"],
        mathengine=mathengine,
        canonical="https://docs.sciml.ai/stable/"),
    pages=fullpages
)

deploydocs(;
    repo="github.com/SciML/SciMLDocs",
    devbranch="main"
)
