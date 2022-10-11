using Documenter, LibGit2, Pkg
using MultiDocumenter

clonedir = mktempdir()

# Ordering Matters!
docsmodules = [
    #"Home" => ["SciMLDocs"],
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

docs = Any[
    push!(docsites,MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, "Documenter"),
        path = "SciMLDocs",
        name = "Home",
        giturl = "https://github.com/SciML/SciMLDocs.git",
    ))
]

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

outpath = mktempdir()

MultiDocumenter.make(
    outpath,
    docs;
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["stable"],
        engine = MultiDocumenter.FlexSearch
    )
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
