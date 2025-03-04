using Documenter, LibGit2, Pkg
using MultiDocumenter

include("CommercialSupportComponent.jl")

clonedir = joinpath(@__DIR__, "cloned")

# Ordering Matters!
docsmodules = [
    "Modeling" => [
        "Modeling Languages" => ["ModelingToolkit", "Catalyst", "NBodySimulator",
            "ParameterizedFunctions"],
        "Model Libraries and Importers" => ["ModelingToolkitStandardLibrary",
            "ModelingToolkitNeuralNets",
            "DiffEqCallbacks",
            "FiniteStateProjection",
            "CellMLToolkit", "SBMLToolkit",
            "BaseModelica",
            "ReactionNetworkImporters"],
        "Symbolic Tools" => ["ModelOrderReduction", "Symbolics", "SymbolicUtils"], #="MetaTheory"=#
        #=
        "Third-Party Modeling Tools" => ["MomentClosure", "Agents", "Unitful",
            "ReactionMechanismSimulator",
            "AlgebraicPetri"],
        =#
        "Array Libraries" => ["RecursiveArrayTools", "ComponentArrays", "LabelledArrays", "MultiScaleArrays"],
        #=
        "Third-Party Array Libraries" => ["StaticArrays", #="FillArrays",=#
            "BandedMatrices", "BlockBandedMatrices"],
        =#
    ],
    "Solvers" => [
        "Equation Solvers" => [
            "LinearSolve", "NonlinearSolve", "DiffEqDocs", "Integrals",
            "DifferenceEquations", "Optimization", "JumpProcesses", "LineSearch"],
        #=
        "Third-Party Equation Solvers" => [
            "LowRankIntegrators",
            "FractionalDiffEq",
            "ManifoldDiffEq",
        ],
        =#
        "Inverse Problems / Estimation" => [
            "SciMLSensitivity", "DiffEqParamEstim", "DiffEqBayes"],
        "PDE Solvers" => ["MethodOfLines", "NeuralPDE",
            "NeuralOperators", "FEniCS",
            "HighDimPDE", "DiffEqOperators"],
        #=
        "Third-Party PDE Solvers" => [
            "Trixi",
            "Ferrite",
            "Gridap",
            "ApproxFun",
            "VoronoiFVM",
        ],
        =#
        "Advanced Solver APIs" => ["OrdinaryDiffEq", "BoundaryValueDiffEq", "DiffEqGPU"],
    ],
    "Analysis" => [
        #= "Plots and Visualization" => ["Makie"], #="PlotDocs",=# =#
        "Parameter Analysis" => ["EasyModelAnalysis", "GlobalSensitivity", "StructuralIdentifiability"],
        #=
        "Third-Party Parameter Analysis" => ["DynamicalSystems", "BifurcationKit",
            "ControlSystems", "ReachabilityAnalysis"],
        =#
        "Uncertainty Quantification" => ["PolyChaos", "SciMLExpectations"],
        #=
        "Third-Party Uncertainty Quantification" => ["Measurements",
            "MonteCarloMeasurements",
            "ProbNumDiffEq", "TaylorIntegration",
            "IntervalArithmetic"],
        =#
    ],
    "Machine Learning" => [
        "Function Approximation" => ["Surrogates", "ReservoirComputing"],
        "Implicit Layer Deep Learning" => ["DiffEqFlux", "DeepEquilibriumNetworks"],
        #= "Third-Party Implicit Layer Deep Learning" => ["Flux", "Lux", "SimpleChains"], =#
        "Symbolic Learning" => ["DataDrivenDiffEq", "SymbolicNumericIntegration"],
        #= "Third-Party Symbolic Learning" => ["SymbolicRegression"], =#
        "Third-Party Differentiation Tooling" => ["SparseDiffTools", "FiniteDiff",
            #= "ForwardDiff",
            "Zygote", "Enzyme" =#],
    ],
    "Developer Tools" => [
        "Numerical Utilities" => ["ExponentialUtilities", "DiffEqNoiseProcess",
            "PreallocationTools", "EllipsisNotation", "DataInterpolations",
            "PoissonRandom", "QuasiMonteCarlo", "RuntimeGeneratedFunctions", "MuladdMacro", "FindFirstFunctions",
            "SparseDiffTools"],
        #=
        "Third-Party Numerical Utilities" => ["FFTW", "Distributions",
            "SpecialFunctions", "LoopVectorization",
            "Polyester", ], #="Tullio"=#
        =#
        "High-Level Interfaces" => [
            "SciMLBase",
            "SciMLStructures",
            "SymbolicIndexingInterface",
            "TermInterface",
            "SciMLOperators",
            "SurrogatesBase",
            "CommonSolve",
        ],
        "Third-Party Interfaces" => ["ArrayInterface", "StaticArrayInterface", #= "AbstractFFTs",
            "GPUArrays", #= "Adapt", =#
            "Tables" =#],                                 #= "RecipesBase", =#
        "Developer Documentation" => ["SciMLStyle", "ColPrac", "DiffEqDevDocs"],
        "Extra Resources" => [
            "SciMLWorkshop",
            "SciMLTutorialsOutput",
            "SciMLBenchmarksOutput",
            "ModelingToolkitCourse",
        ],
    ],
]

fixnames = Dict("SciMLDocs" => "The SciML Open Source Software Ecosystem",
                "DiffEqDocs" => "DifferentialEquations",
                "DiffEqDevDocs" => "DiffEq Developer Documentation",
                "PlotDocs" => "Plots",
                "SciMLBenchmarksOutput" => "The SciML Benchmarks",
                "SciMLTutorialsOutput" => "Extended SciML Tutorials")
hasnojl = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput", "ColPrac", "SciMLStyle", "ModelingToolkitCourse"]
usemain = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput"]

external_urls = Dict("Enzyme" => "https://github.com/EnzymeAD/Enzyme.jl",
                     "Zygote" => "https://github.com/FluxML/Zygote.jl",
                     "FiniteDiff" => "https://github.com/JuliaDiff/FiniteDiff.jl",
                     "ForwardDiff" => "https://github.com/JuliaDiff/ForwardDiff.jl",
                     "SparseDiffTools" => "https://github.com/JuliaDiff/SparseDiffTools.jl",
                     "ManifoldDiffEq" => "https://github.com/JuliaManifolds/ManifoldDiffEq.jl",
                     "FractionalDiffEq" => "https://github.com/SciFracX/FractionalDiffEq.jl",
                     "Agents" => "https://github.com/JuliaDynamics/Agents.jl",
                     "LowRankIntegrators" => "https://github.com/FHoltorf/LowRankIntegrators.jl",
                     "MomentClosure" => "https://github.com/augustinas1/MomentClosure.jl",
                     "Trixi" => "https://github.com/trixi-framework/Trixi.jl",
                     "Gridap" => "https://github.com/gridap/Gridap.jl",
                     "Ferrite" => "https://github.com/Ferrite-FEM/Ferrite.jl",
                     "ApproxFun" => "https://github.com/JuliaApproximation/ApproxFun.jl",
                     "VoronoiFVM" => "https://github.com/j-fu/VoronoiFVM.jl",
                     "Symbolics" => "https://github.com/JuliaSymbolics/Symbolics.jl",
                     "SymbolicUtils" => "https://github.com/JuliaSymbolics/SymbolicUtils.jl",
                     "TermInterface" => "https://github.com/JuliaSymbolics/TermInterface.jl",
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
                     "BifurcationKit" => "https://github.com/bifurcationkit/BifurcationKitDocs.jl",
                     "ReachabilityAnalysis" => "https://github.com/JuliaReach/ReachabilityAnalysis.jl",
                     "ControlSystems" => "https://github.com/JuliaControl/ControlSystems.jl",
                     "Flux" => "https://github.com/FluxML/Flux.jl",
                     "Lux" => "https://github.com/LuxDL/Lux.jl",
                     "SimpleChains" => "https://github.com/PumasAI/SimpleChains.jl",
                     "NNlib" => "https://github.com/FluxML/NNlib.jl",
                     "SymbolicRegression" => "https://github.com/MilesCranmer/SymbolicRegression.jl",
                     "FFTW" => "https://github.com/JuliaMath/FFTW.jl",
                     "Distributions" => "https://github.com/JuliaStats/Distributions.jl",
                     "SpecialFunctions" => "https://github.com/JuliaMath/SpecialFunctions.jl",
                     "LoopVectorization" => "https://github.com/JuliaSIMD/LoopVectorization.jl",
                     "Polyester" => "https://github.com/JuliaSIMD/Polyester.jl",
                     "Tullio" => "https://github.com/mcabbott/Tullio.jl",
                     "ArrayInterface" => "https://github.com/JuliaArrays/ArrayInterface.jl",
                     "StaticArrayInterface" => "https://github.com/JuliaArrays/StaticArrayInterface.jl",
                     "AbstractFFTs" => "https://github.com/JuliaMath/AbstractFFTs.jl",
                     "GPUArrays" => "https://github.com/JuliaGPU/GPUArrays.jl",
                     "Tables" => "https://github.com/JuliaData/Tables.jl",
                     "Unitful" => "https://github.com/PainterQubits/Unitful.jl",
                     "ReactionMechanismSimulator" => "https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl",
                     "FiniteStateProjection" => "https://github.com/kaandocal/FiniteStateProjection.jl",
                     "AlgebraicPetri" => "https://github.com/AlgebraicJulia/AlgebraicPetri.jl")

docs = Any[MultiDocumenter.MultiDocRef(upstream = joinpath(clonedir, "Home"),
                                       path = "Overview",
                                       name = "Home",
                                       giturl = "https://github.com/SciML/SciMLDocs.git")]

for group in docsmodules
    docgroups = []
    for cat in group[2]
        docsites = MultiDocumenter.DropdownComponent[]
        for mod in cat[2]
            url = if mod in hasnojl
                "https://github.com/SciML/$mod.git"
            elseif mod in keys(external_urls)
                external_urls[mod]
            else
                "https://github.com/SciML/$mod.jl.git"
            end
            push!(docsites,
                  MultiDocumenter.MultiDocRef(upstream = joinpath(clonedir, mod),
                                              path = mod,
                                              name = mod in keys(fixnames) ? fixnames[mod] :
                                                     mod,
                                              giturl = url,
                                              branch = mod ∈ usemain ? "main" : "gh-pages"))
        end
        push!(docgroups, MultiDocumenter.Column(cat[1], docsites))
    end
    push!(docs, MultiDocumenter.MegaDropdownNav(group[1], docgroups))
end

push!(docs, MultiDocumenter.MegaDropdownNav(
    "Commercial Support",
    [
        MultiDocumenter.Column("Commercial Support", [JuliaHubCommercialSupportComponent("https://juliahub.com/company/contact-us-sciml-docs")]),
        MultiDocumenter.Column("Products built with SciML", [ProductsUsedComponent()]),
    ]
))

outpath = joinpath(@__DIR__, "build")

MultiDocumenter.make(outpath, docs;
                     assets_dir = "docs/src/assets",
                     search_engine = MultiDocumenter.SearchConfig(index_versions = [
                                                                      "stable",
                                                                  ],
                                                                  engine = MultiDocumenter.FlexSearch),
                     custom_scripts = [
                         "https://www.googletagmanager.com/gtag/js?id=G-Q3FE4BYYHQ",
                         Docs.HTML("""
                         window.dataLayer = window.dataLayer || [];
                         function gtag(){dataLayer.push(arguments);}
                         gtag('js', new Date());
                         gtag('config', 'G-Q3FE4BYYHQ');
                         """),
                     ],
                     brand_image = MultiDocumenter.BrandImage("https://sciml.ai",
                                                              joinpath("assets",
                                                                       "logo.png")))
