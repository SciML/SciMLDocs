using Documenter, LibGit2, Pkg
using MultiDocumenter

clonedir = joinpath(@__DIR__, "clones")
# clonedir = mktempdir()

# Ordering Matters!
docsmodules = [
    "Solvers" => [
    "Equation Solvers" => ["LinearSolve", "NonlinearSolve", "DiffEqDocs", "Integrals",
                           "Optimization", "JumpProcesses"],
    "Inverse Problems / Estimation" => [
                                   "SciMLSensitivity", "DiffEqParamEstim", "DiffEqBayes"],
    "PDE Solvers" => ["MethodOfLines", "NeuralPDE",
                     "NeuralOperators", "FEniCS",
                      "HighDimPDE", "DiffEqOperators"],
    "Third-Party PDE Solvers" => ["Trixi", "Gridap", "ApproxFun", "VoronoiFVM"]
    ],

    "Modeling" => [
    "Modeling Languages" => ["ModelingToolkit", "Catalyst", "NBodySimulator",
                             "ParameterizedFunctions"],
    "Model Libraries and Importers" => ["ModelingToolkitStandardLibrary", "DiffEqCallbacks",
                                       #="CellMLToolkit", =# "SBMLToolkit",
                                       #="ReactionNetworkImporters"=#],
    "Symbolic Tools" => ["Symbolics", "SymbolicUtils", "MetaTheory"],
    "Array Libraries" => ["RecursiveArrayTools", "LabelledArrays", "MultiScaleArrays"],
    "Third-Party Array Libraries" => ["ComponentArrays", "StaticArrays", #"FillArrays",
                                     "BandedMatrices", "BlockBandedMatrices"]
    ],

    "Analysis" => [
    "Plots and Visualization" => [
        # "PlotDocs",
        "Makie"],
    "Parameter Analysis" => ["GlobalSensitivity", "StructuralIdentifiability"],
    "Uncertainty Quantification" => ["PolyChaos",  "SciMLExpectations" ],
    "Third-Party Uncertainty Quantification" => ["Measurements", "MonteCarloMeasurements",
                                                "ProbNumDiffEq", "TaylorIntegration",
                                                "IntervalArithmetic"],
    "Third-Party Parameter Analysis" => ["DynamicalSystems", "BifurcationKit",
                                      "ControlSystems", "ReachabilityAnalysis"],
    ],

    "Machine Learning" => [
       "Implicit Layer Deep Learning" => ["DiffEqFlux","DeepEquilibriumNetworks"],
        "Function Approximation" => ["Surrogates", "ReservoirComputing"],
       "Symbolic Learning" => ["DataDrivenDiffEq", "SymbolicNumericIntegration"],
     "Third-Party Deep Learning" => ["Flux", "Lux", "SimpleChains", #="NNlib"=#],
     "Third-Party Symbolic Learning" => ["SymbolicRegression"]
    ],

    "Developer Tools" => [
    "Numerical Utilities" => ["ExponentialUtilities", "DiffEqNoiseProcess",
       #"PreallocationTools", "EllipsisNotation",
       "PoissonRandom", "QuasiMonteCarlo", "RuntimeGeneratedFunctions", "MuladdMacro"],
    "Third-Party Numerical Utilities" => ["FFTW",
                                          # "DataInterpolations",
                                         "Distributions",
                                         "SpecialFunctions", "LoopVectorization",
                                         "Polyester",
                                         # "Tullio"
                                         ],
    "High-Level Interfaces" => ["SciMLBase", "SciMLOperators", "CommonSolve"],
    "Third-Party Interfaces" => ["ArrayInterface",
                                # "Adapt",
                                "AbstractFFTs", "GPUArrays",
                                #"RecipesBase",
                                 "Tables", ],
    "Developer Documentation" => ["SciMLStyle", "COLPRAC", "DiffEqDevDocs"],
    "Extra Resources" => ["SciMLTutorialsOutput", "SciMLBenchmarksOutput"],
    ],
]

fixnames = Dict("SciMLDocs" => "The SciML Open Souce Software Ecosystem",
                "DiffEqDocs" => "DifferentialEquations",
                "DiffEqDevDocs" => "DiffEq Developer Documentation",
                "PlotDocs" => "Plots",
                "SciMLBenchmarksOutput" => "The SciML Benchmarks",
                "SciMLTutorialsOutput" => "Extended SciML Tutorials")
hasnojl = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput", "COLPRAC", "SciMLStyle"]
usemain = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput"]

external_urls = Dict(
    "Trixi" => "https://github.com/trixi-framework/Trixi.jl",
    "Gridap" => "https://github.com/gridap/Gridap.jl",
    "ApproxFun" => "https://github.com/JuliaApproximation/ApproxFun.jl",
    "VoronoiFVM" => "https://github.com/j-fu/VoronoiFVM.jl",
    "Symbolics" => "https://github.com/JuliaSymbolics/Symbolics.jl",
    "SymbolicUtils" => "https://github.com/JuliaSymbolics/SymbolicUtils.jl",
    "MetaTheory" => "https://github.com/JuliaSymbolics/MetaTheory.jl",
    "ComponentArrays" => "https://github.com/jonniedie/ComponentArrays.jl",
    "StaticArrays" => "https://github.com/JuliaArrays/StaticArrays.jl",
    "FillArrays" => "https://github.com/JuliaArrays/FillArrays.jl",
    "BandedMatrices" => "https://github.com/JuliaMatrices/BandedMatrices.jl",
    "BlockBandedMatrices" => "https://github.com/JuliaMatrices/BlockBandedMatrices.jl",
    "PlotDocs" => "https://github.com/JuliaPlots/PlotDocs.jl",
    "Makie" => "https://github.com/MakieOrg/Makie.jl",
    "Measurements" => "https://github.com/JuliaPhysics/Measurements.jl",
    "MonteCarloMeasurements" => "https://github.com/baggepinnen/MonteCarloMeasurements.jl",
    "ProbNumDiffEq" => "https://github.com/nathanaelbosch/ProbNumDiffEq.jl",
    "TaylorIntegration" => "https://github.com/PerezHz/TaylorIntegration.jl",
    "IntervalArithmetic" => "https://github.com/JuliaIntervals/IntervalArithmetic.jl",
    "DynamicalSystems" => "https://github.com/JuliaDynamics/DynamicalSystems.jl",
    "BifurcationKit" => "https://github.com/bifurcationkit/BifurcationKit.jl",
    "ReachabilityAnalysis" => "https://github.com/JuliaReach/ReachabilityAnalysis.jl",
    "ControlSystems" => "https://github.com/JuliaControl/ControlSystems.jl",
    "Flux" => "https://github.com/FluxML/Flux.jl",
    "Lux" => "https://github.com/avik-pal/Lux.jl",
    "SimpleChains" => "https://github.com/PumasAI/SimpleChains.jl",
    "NNlib" => "https://github.com/FluxML/NNlib.jl",
    "SymbolicRegression" => "https://github.com/MilesCranmer/SymbolicRegression.jl",
    "FFTW" => "https://github.com/JuliaMath/FFTW.jl",
    "DataInterpolations" => "https://github.com/PumasAI/DataInterpolations.jl",
    "Distributions" => "https://github.com/JuliaStats/Distributions.jl",
    "SpecialFunctions" => "https://github.com/JuliaMath/SpecialFunctions.jl",
    "LoopVectorization" => "https://github.com/JuliaSIMD/LoopVectorization.jl",
    "Polyester" => "https://github.com/JuliaSIMD/Polyester.jl",
    "Tullio" => "https://github.com/mcabbott/Tullio.jl",
    "ArrayInterface" => "https://github.com/JuliaArrays/ArrayInterface.jl",
    "AbstractFFTs" => "https://github.com/JuliaMath/AbstractFFTs.jl",
    "GPUArrays" => "https://github.com/JuliaGPU/GPUArrays.jl",
    "Tables" => "https://github.com/JuliaData/Tables.jl",
)

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
            elseif mod in keys(external_urls)
                external_urls[mod]
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

outpath = joinpath(@__DIR__, "out")

MultiDocumenter.make(
    outpath, docs;
    assets_dir = joinpath(@__DIR__, "assets"),
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["stable"],
        engine = MultiDocumenter.FlexSearch,
        lowfi = true
    ),
    brand_image = MultiDocumenter.BrandImage("SciMLDocs", "assets/logo.png")
)
