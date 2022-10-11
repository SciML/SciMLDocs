using Documenter, LibGit2, Pkg
using MultiDocumenter

clonedir = mktempdir()

# Ordering Matters!
docsmodules = [
    "Solvers" => [
    "Equation Solvers" => ["LinearSolve", "NonlinearSolve", #="DiffEqDocs",=# "Integrals",
                           "Optimization", "JumpProcesses"],
    #"Inverse Problems and Parameter Estimation" => [
    #                                "SciMLSensitivity", "DiffEqParamEstim", "DiffEqBayes"],
    #"Partial Differential Equations" => ["MethodOfLines", "NeuralPDE",
    #                                    "NeuralOperators", "FEniCS",
    #                                    "HighDimPDE", "DiffEqOperators"],

    ],

    #=


    "Modeling Tools" => [
    "Modeling Languages" => ["ModelingToolkit", "Catalyst", "NBodySimulator",
                             "ParameterizedFunctions"],
    "Pre-Built Model Libraries" => ["ModelingToolkitStandardLibrary", "DiffEqCallbacks"],
    "Array Libraries" => ["RecursiveArrayTools", "LabelledArrays", "MultiScaleArrays"],
    ],

    "Simulation Analysis" => [
    "Uncertainty Quantification" => ["PolyChaos"],
    "Parameter Analysis" => ["GlobalSensitivity", "StructuralIdentifiability"],
    ],

    "Machine Learning" => [
        "Implicit Layer Deep Learning" => ["DiffEqFlux","DeepEquilibriumNetworks"],
        "Robust Function Learning" => ["Surrogates", "ReservoirComputing"],
        "Symbolic Learning" => ["DataDrivenDiffEq", "SymbolicNumericIntegration"],
    ],

    "Developer Tools" => [
    "Numerical Utilities" => ["ExponentialUtilities", "DiffEqNoiseProcess",
        "PoissonRandom", "QuasiMonteCarlo", "RuntimeGeneratedFunctions", "MuladdMacro"],
    "High-Level Interfaces" => ["SciMLBase", "SciMLOperators", "CommonSolve"],
    "Developer Documentation" => ["SciMLStyle", "COLPRAC", "DiffEqDevDocs"],
    ],

    "Extra Resources" => ["SciMLTutorialsOutput", "SciMLBenchmarksOutput"],
    =#
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
    @show group
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
