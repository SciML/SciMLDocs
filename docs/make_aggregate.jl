using Documenter, LibGit2, Pkg
using MultiDocumenter

clonedir = mktempdir()

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
        "Array Libraries" => ["RecursiveArrayTools", "LabelledArrays", "MultiScaleArrays"],
    ],
    "Solvers" => [
        "Equation Solvers" => ["LinearSolve", "NonlinearSolve", "DiffEqDocs", "Integrals", "DifferenceEquations",
            "Optimization", "JumpProcesses"],
        "Inverse Problems / Estimation" => [
            "SciMLSensitivity", "DiffEqParamEstim", "DiffEqBayes"],
        "PDE Solvers" => ["MethodOfLines", "NeuralPDE",
            "NeuralOperators", "FEniCS",
            "HighDimPDE", "DiffEqOperators"],
        "Advanced Solver APIs" => ["OrdinaryDiffEq", "DiffEqGPU"],
    ],
    "Analysis" => [
        "Plots and Visualization" => ["Makie"], #="PlotDocs",=#
        "Parameter Analysis" => ["EasyModelAnalysis", "GlobalSensitivity", "StructuralIdentifiability"],
        "Uncertainty Quantification" => ["PolyChaos", "SciMLExpectations"],
    ],
    "Machine Learning" => [
        "Function Approximation" => ["Surrogates", "ReservoirComputing"],
        "Implicit Layer Deep Learning" => ["DiffEqFlux", "DeepEquilibriumNetworks"],
        "Symbolic Learning" => ["DataDrivenDiffEq", "SymbolicNumericIntegration"],
    ],
    "Developer Tools" => [
        "Numerical Utilities" => ["ExponentialUtilities", "DiffEqNoiseProcess",
            "PreallocationTools", "EllipsisNotation", "DataInterpolations",
            "PoissonRandom", "QuasiMonteCarlo", "RuntimeGeneratedFunctions", "MuladdMacro", "FindFirstFunctions"],
        "High-Level Interfaces" => [
            "SciMLBase",
            "SciMLStructures",
            "SymbolicIndexingInterface",
            "TermInterface",
            "SciMLOperators",
            "SurrogatesBase",
            "CommonSolve",
        ],
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
                "SciMLBenchmarksOutput" => "The SciML Benchmarks",
                "SciMLTutorialsOutput" => "Extended SciML Tutorials")
hasnojl = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput", "ColPrac", "SciMLStyle", "ModelingToolkitCourse"]
usemain = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput"]

docs = Any[MultiDocumenter.MultiDocRef(upstream = joinpath(clonedir, "Home"),
                                       path = "Overview",
                                       name = "Home",
                                       giturl = "https://github.com/SciML/SciMLDocs.git")]

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
