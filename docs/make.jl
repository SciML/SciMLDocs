using SciMLDocs, Documenter, LibGit2, Pkg
using MultiDocumenter

clonedir = mktempdir()

# Ordering Matters!
docsmodules = [
    "SciML" => ["SciMLDocs"],
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
    "PoissonRandom", "QuasiMonteCarlo", "RuntimeGeneratedFunctions", "MuladdMacro",],
    "Machine Learning" => ["DiffEqFlux","DeepEquilibriumNetworks","ReservoirComputing"],
    "Extra Resources" => ["SciMLTutorialsOutput", "SciMLBenchmarksOutput"],
    "Developer Documentation" => ["SciMLStyle", "COLPRAC", "DiffEqDevDocs"],
]

fixnames = Dict("SciMLDocs" => "The SciML Open Souce Software Ecosystem",
                "DiffEqDocs" => "DifferentialEquations",
                "DiffEqDevDocs" => "DiffEq Developer Documentation",
                "SciMLBenchmarksOutput" => "The SciML Benchmarks",
                "SciMLTutorialsOutput" => "Extended SciML Tutorials and Learning Materials")
hasnojl = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput", "COLPRAC", "SciMLStyle"]

makedocs(sitename = "SciMLDocs.jl",
         authors = "Chris Rackauckas",
         modules = Module[],
         clean = true, doctest = false,
         strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://scimldocs.sciml.ai/stable/"),
         pages = [
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
        ])

deploydocs(repo = "github.com/SciML/SciMLDocs")

docs = []

for (i, cat) in enumerate(docsmodules)
    docsites = []
    for mod in cat[2]
        url = if mod in hasnojl
            "https://github.com/SciML/$mod.git"
        else
            "https://github.com/SciML/$mod.jl.git"
        end
        push!(docsites,MultiDocumenter.MultiDocRef(
            upstream = joinpath(clonedir, "Documenter"),
            path = mod,
            name = mod in keys(fixnames) ? fixnames[mod] : mod,
            giturl = url,
        ))
    end
    push!(docs, MultiDocumenter.DropdownNav(cat[1], docsites))
end

outpath = joinpath(@__DIR__, "out")

MultiDocumenter.make(
    outpath,
    docs;
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["stable"],
        engine = MultiDocumenter.FlexSearch
    )
)
