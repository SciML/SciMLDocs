using SciMLDocs, Documenter, LibGit2, Pkg
using MultiDocumenter

# Ordering Matters!
docsmodules = [
    "Equation Solvers" => ["LinearSolve", "NonlinearSolve", "DiffEqDocs", "Integrals",
                           "Optimization", "JumpProcesses"],
    "Partial Differential Equation Solvers" => ["MethodOfLines", "NeuralPDE",
                                                "NeuralOperators", "FEniCS",
                                                "HighDimPDE", "DiffEqOperators"],
    "Modeling Tools" => ["DiffEqCallbacks", "ModelingToolkit", "ModelingToolkitStandardLibrary", 
        "Catalyst", "NBodySimulator", "ParameterizedFunctions"],
    "Inverse Problems" => ["SciMLSensitivity", "DiffEqParamEstim", "DiffEqBayes"],
    "AbstractArray Libraries" => ["RecursiveArrayTools", "LabelledArrays", "MultiScaleArrays"],
    "Uncertainty Quantification" => ["PolyChaos"],
    "Simulation Analysis" => ["GlobalSensitivity"],
    "Symbolic Analysis" => ["StructuralIdentifiability", "SymbolicNumericIntegration"],
    "Interfaces" => ["SciMLBase", "SciMLOperators", "CommonSolve"],
    "Numerical Utilities" => ["Surrogates", "ExponentialUtilities", "DiffEqNoiseProcess",
    "PoissonRandom", "QuasiMonteCarlo", "DataInterpolations",
    "FFTW", "RuntimeGeneratedFunctions", "MuladdMacro",],
    "Machine Learning" => ["DiffEqFlux","DeepEquilibriumNetworks","ReservoirComputing"],
    "Extra Resources" => ["SciMLTutorialsOutput", "SciMLBenchmarksOutput"],
    "Developer Documentation" => ["SciMLStyle", "COLPRAC", "DiffEqDevDocs"],
]

docspackage = ["DiffEqDocs", "DiffEqDevDocs", "SciMLBenchmarksOutput", "SciMLTutorialsOutput"]
docspackagenames = Dict("DiffEqDocs" => "DifferentialEquations",
                        "DiffEqDevDocs" => "DiffEq Developer Documentation",
                        "SciMLBenchmarksOutput" => "The SciML Benchmarks",
                        "SciMLTutorialsOutput" => "Extended SciML Tutorials and Learning Materials")
docspackage_hasjl = Dict("DiffEqDocs" => true,
                         "DiffEqDevDocs" => true,
                         "SciMLBenchmarksOutput" => false,
                         "SciMLTutorialsOutput" => false)

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

highlevelpages = [
    "The SciML Open Souce Software Ecosystem" => "index.md",
    "highlevels/equation_solvers.md",
    "highlevels/partial_differential_equation_solvers.md",
    "highlevels/modeling_tools.md",
    "highlevels/inverse_problems.md",
    "highlevels/abstractarray_libraries.md",
    "highlevels/uncertainty_quantification.md",
    "highlevels/simulation_analysis.md",
    "highlevels/symbolic_analysis.md",
    "highlevels/interfaces.md",
    "highlevels/numerical_utilities.md",
    "highlevels/machine_learning.md",
    "highlevels/learning_resources.md",
    "highlevels/developer_documentation.md",
]

docs = []

for (i, cat) in enumerate(docsmodules)
    for mod in cat[2]
        if mod in usereadme
            dir = joinpath(pkgdir(SciMLDocs), "docs", "src", "modules", mod)
            mkdir(dir)
            mkdir(mod)
            LibGit2.clone(readmeurls[mod], mod)
            cp(joinpath(mod, "README.md"), joinpath(dir, "index.md"), force=true)
            push!(catpage, mod => Any[joinpath("modules", mod, "index.md")])
        elseif mod in docspackage
            if docspackage_hasjl[mod]
                push!(docs,("https://github.com/SciML/$mod.jl.git", "gh-pages") => MultiDocumenter.MultiDocRef(
                    upstream = joinpath(clonedir, "Documenter"),
                    path = "doc",
                    name = docspackagenames[mod],
                ))
            else
                push!(docs,("https://github.com/SciML/$mod.git", "gh-pages") => MultiDocumenter.MultiDocRef(
                    upstream = joinpath(clonedir, "Documenter"),
                    path = "doc",
                    name = mod,
                ))
            end
        else
            push!(docs,("https://github.com/SciML/$mod.jl.git", "gh-pages") => MultiDocumenter.MultiDocRef(
                upstream = joinpath(clonedir, "Documenter"),
                path = "doc",
                name = mod,
            ))
        end
    end
end

clonedir = mktempdir()

# using SSH for cloning is suggested when you're dealing with private repos, because
# an ssh-agent will handle your keys for you
# prefix = "git@github.com:"
prefix = "https://github.com/"

for ((remote, branch), docref) in docs
    run(`git clone --depth 1 $prefix$remote --branch $branch --single-branch $(docref.upstream)`)
end

outpath = joinpath(@__DIR__, "out")

MultiDocumenter.make(
    outpath,
    collect(last.(docs));
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["stable"],
        engine = MultiDocumenter.FlexSearch
    )
)