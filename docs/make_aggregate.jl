using Documenter, LibGit2, Pkg
using MultiDocumenter

clonedir = mktempdir()

# Ordering Matters!
docsmodules = [
    "Solvers" => [
    "Equation Solvers" => ["LinearSolve", "NonlinearSolve", #="DiffEqDocs",=# "Integrals",
                           "Optimization", "JumpProcesses"],
    #"Inverse Problems / Estimation" => [
    #                                "SciMLSensitivity", "DiffEqParamEstim", "DiffEqBayes"],
    "PDE Solvers" => ["MethodOfLines", "NeuralPDE",
                      "NeuralOperators", "FEniCS",
                      "HighDimPDE", "DiffEqOperators"],
    #"Third-Party PDE Solvers" => ["Trixi", "Gridap", "ApproxFun", "VoronoiFVM"]
    ],

    "Modeling" => [
    "Modeling Languages" => ["ModelingToolkit", "Catalyst", "NBodySimulator",
                             "ParameterizedFunctions"],
    #"Model Libraries and Importers" => ["ModelingToolkitStandardLibrary", "DiffEqCallbacks",
    #                                    "CellMLToolkit", "SBMLToolkit",
    #                                    "ReactionNetworkImporters"],
    #"Symbolic Tools" => ["Symbolics", "SymbolicUtils", "MetaTheory"],
    #"Array Libraries" => ["RecursiveArrayTools", "LabelledArrays", "MultiScaleArrays"],
    #"Third-Party Array Libraries" => ["ComponentArrays", "StaticArrays", "BandedArrays",
    #                                  "FillArrays", "BlockBandedArrays"]
    ],

    "Analysis" => [
    #"Plots and Visualization" => ["Plots", "Makie"]
    #"Parameter Analysis" => ["GlobalSensitivity", "StructuralIdentifiability"],
    "Uncertainty Quantification" => ["PolyChaos", #= "SciMLExpectations" =#],
    #"Third-Party Uncertainty Quantification" => ["Measurements", "MonteCarloMeasurements",
    #                                             "ProbNumDiffEq", "TaylorIntegration",
    #                                             "IntervalArithmetic"],
    #"Third-Party Parameter Analysis => ["BifurcationKit", "ReachabilityAnalysis",
    #                                   "ControlSystems", "DynamicalSystems"],
    ],

    "Machine Learning" => [
     #   "Implicit Layer Deep Learning" => ["DiffEqFlux","DeepEquilibriumNetworks"],
        "Function Approximation" => ["Surrogates", "ReservoirComputing"],
     #   "Symbolic Learning" => ["DataDrivenDiffEq", "SymbolicNumericIntegration"],
     # "Third-Party Deep Learning" => ["Flux", "Lux", "SimpleChains", "NNLib"],
     # "Third-Party Symbolic Learning" => ["SymbolicRegression"]
    ],

    "Developer Tools" => [
    #"Numerical Utilities" => ["ExponentialUtilities", "DiffEqNoiseProcess",
    #    "PreallocationTools", "EllipsisNotation",
    #    "PoissonRandom", "QuasiMonteCarlo", "RuntimeGeneratedFunctions", "MuladdMacro"],
    #"Third-Party Numerical Utilities" => ["FFTW", "DataInterpolations", "Distributions",
    #                                      "SpecialFunctions", "LoopVectorization",
    #                                      "Polyester", "Tullio"]
    "High-Level Interfaces" => ["SciMLBase", "SciMLOperators", "CommonSolve"],
    #"Third-Party Interfaces" => ["ArrayInterface", "Adapt", "AbstractFFTs", "GPUArrays",
    #                             "RecipesBase", "Tables", ]
    "Developer Documentation" => ["SciMLStyle", "COLPRAC", "DiffEqDevDocs"],
    "Extra Resources" => ["SciMLTutorialsOutput", "SciMLBenchmarksOutput"],
    ],
]

fixnames = Dict("SciMLDocs" => "The SciML Open Souce Software Ecosystem",
                "DiffEqDocs" => "DifferentialEquations",
                "DiffEqDevDocs" => "DiffEq Developer Documentation",
                "SciMLBenchmarksOutput" => "The SciML Benchmarks",
                "SciMLTutorialsOutput" => "Extended SciML Tutorials")
hasnojl = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput", "COLPRAC", "SciMLStyle"]
usemain = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput"]

docs = Any[
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, "Home"),
        path = "SciMLDocs",
        name = "Home",
        giturl = "https://github.com/SciML/SciMLDocs.git",
    )
]

for group in docsmodules
    docgroups = []
    for cat in group[2]
        docsites = []
        for mod in cat[2]
            url = if mod in hasnojl
                "https://github.com/SciML/$mod.git"
            else
                "https://github.com/SciML/$mod.jl.git"
            end
            push!(docsites,MultiDocumenter.MultiDocRef(
                upstream = joinpath(clonedir, mod),
                path = mod,
                name = mod in keys(fixnames) ? fixnames[mod] : mod,
                giturl = url,
                branch = mod ∈ usemain ? "main" : "gh-pages"
            ))
        end
        push!(docgroups, MultiDocumenter.Column(cat[1], docsites))
    end
    push!(docs, MultiDocumenter.MegaDropdownNav(group[1], docgroups))
end

outpath = mktempdir()

MultiDocumenter.make(
    outpath, docs;
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["stable"],
        engine = MultiDocumenter.FlexSearch
    ),
    brand_image = MultiDocumenter.BrandImage("https://docs.sciml.ai",
                                             joinpath(@__DIR__,"assets","logo.png"))
)

gitroot = normpath(joinpath(@__DIR__, ".."))
run(`git pull`)
outbranch = "aggregate-pages"
has_outbranch = true
if !success(`git checkout $outbranch`)
    has_outbranch = false
    if !success(`git switch --orphan $outbranch`)
        @error "Cannot create new orphaned branch $outbranch."
        exit(1)
    end
end
for file in readdir(gitroot; join = true)
    endswith(file, ".git") && continue
    rm(file; force = true, recursive = true)
end
for file in readdir(outpath)
    cp(joinpath(outpath, file), joinpath(gitroot, file))
end
run(`git add .`)
if success(`git commit -m 'Aggregate documentation'`)
    @info "Pushing updated documentation."
    if has_outbranch
        run(`git push`)
    else
        run(`git push -u origin $outbranch`)
    end
    run(`git checkout main`)
else
    @info "No changes to aggregated documentation."
end
