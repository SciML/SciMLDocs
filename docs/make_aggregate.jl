using Documenter, LibGit2, Pkg
using MultiDocumenter

clonedir = mktempdir()

#=
Big ones:
DiffEqCallbacks
DiffEqDocs
JumpProcesses
NeuralPDE
Trixi
Makie
DynamicalSystems
Surrogates
LoopVectorization
SciMLTutorials
SciMLBenchmarks
=#

# Ordering Matters!
docsmodules = [

    "Modeling" => [
    "Modeling Languages" => ["ModelingToolkit", "Catalyst", "NBodySimulator",
                             "ParameterizedFunctions"],
    "Model Libraries and Importers" => ["ModelingToolkitStandardLibrary", "DiffEqCallbacks",
                                        "CellMLToolkit", "SBMLToolkit",
                                        "ReactionNetworkImporters"],
    "Symbolic Tools" => ["ModelOrderReduction", "Symbolics", #="SymbolicUtils", "MetaTheory"=#],
    "Third-Party Symbolic Tools" => ["MomentClosure"],
    "Array Libraries" => ["RecursiveArrayTools", "LabelledArrays", "MultiScaleArrays"],
    "Third-Party Array Libraries" => ["ComponentArrays", "StaticArrays", #="FillArrays",=#
                                      "BandedMatrices", "BlockBandedMatrices"]
    ],

    "Solvers" => [
    "Equation Solvers" => ["LinearSolve", "NonlinearSolve", "DiffEqDocs", "Integrals",
                           "Optimization", "JumpProcesses"],
    "Inverse Problems / Estimation" => [
                                    "SciMLSensitivity", "DiffEqParamEstim", "DiffEqBayes"],
    "PDE Solvers" => ["MethodOfLines", "NeuralPDE",
                      "NeuralOperators", "FEniCS",
                      "HighDimPDE", "DiffEqOperators"],
    "Third-Party PDE Solvers" => ["LowRankIntegrators", "Trixi", "Gridap", "ApproxFun", "VoronoiFVM"]
    ],
    # LowRankIntegrators.jl

    "Analysis" => [
    "Plots and Visualization" => ["PlotDocs", "Makie"],
    "Parameter Analysis" => ["GlobalSensitivity", "StructuralIdentifiability"],
    "Uncertainty Quantification" => ["PolyChaos", "SciMLExpectations"],
    "Third-Party Uncertainty Quantification" => ["Measurements", "MonteCarloMeasurements",
                                                 "ProbNumDiffEq", "TaylorIntegration",
                                                 "IntervalArithmetic"],
    "Third-Party Parameter Analysis" => ["DynamicalSystems", "BifurcationKit",
                                       "ControlSystems", "ReachabilityAnalysis"],
    ],

    "Machine Learning" => [
         "Function Approximation" => ["Surrogates", "ReservoirComputing"],
         "Implicit Layer Deep Learning" => ["DiffEqFlux","DeepEquilibriumNetworks"],
         "Symbolic Learning" => ["DataDrivenDiffEq", "SymbolicNumericIntegration"],
      "Third-Party Deep Learning" => ["Flux", "Lux", "SimpleChains"],
      "Third-Party Symbolic Learning" => ["SymbolicRegression"]
    ],

    "Developer Tools" => [
    "Numerical Utilities" => ["ExponentialUtilities", "DiffEqNoiseProcess",
        #="PreallocationTools", "EllipsisNotation",=#
        "PoissonRandom", "QuasiMonteCarlo", "RuntimeGeneratedFunctions", "MuladdMacro"],
    "Third-Party Numerical Utilities" => ["FFTW", #= "DataInterpolations",=# "Distributions",
                                          "SpecialFunctions", "LoopVectorization",
                                          "Polyester", #="Tullio"=#],
    "High-Level Interfaces" => ["SciMLBase", "SciMLOperators", "CommonSolve"],
    "Third-Party Interfaces" => ["ArrayInterface", #= "Adapt", =# "AbstractFFTs", "GPUArrays",
                                 #= "RecipesBase", =# "Tables", ],
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
    "LowRankIntegrators" => "https://github.com/FHoltorf/LowRankIntegrators.jl",
    "MomentClosure" => "https://github.com/augustinas1/MomentClosure.jl",
    "Trixi" => "https://github.com/trixi-framework/Trixi.jl",
    "Gridap" => "https://github.com/gridap/Gridap.jl",
    "ApproxFun" => "https://github.com/JuliaApproximation/ApproxFun.jl",
    "VoronoiFVM" => "https://github.com/j-fu/VoronoiFVM.jl",
    "Symbolics" => "https://github.com/JuliaSymbolics/Symbolics.jl",
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
        path = "Overview",
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

outpath = joinpath(@__DIR__, "build")

analytics_script = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-Q3FE4BYYHQ"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-Q3FE4BYYHQ');
</script>
"""

MultiDocumenter.make(
    outpath, docs;
    assets_dir = "docs/src/assets",
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["stable"],
        engine = MultiDocumenter.FlexSearch
    ),
    custom_scripts = [analytics_script],
    brand_image = MultiDocumenter.BrandImage("https://sciml.ai",
                                             joinpath("assets","logo.png"))
)
