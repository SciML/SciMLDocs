using Documenter: Documenter
using LibGit2
using Pkg
using MultiDocumenter

include("CommercialSupportComponent.jl")

clonedir = joinpath(@__DIR__, "cloned")

# Ordering Matters!
docsmodules = [
    "Modeling" => [
        "Modeling Languages" => [
            "ModelingToolkit", "Catalyst", "NBodySimulator",
            "ParameterizedFunctions",
            "ProcessSimulator", "MomentClosure",
        ],
        "Model Libraries and Importers" => [
            "ModelingToolkitStandardLibrary",
            "ModelingToolkitNeuralNets",
            "DiffEqCallbacks",
            "FiniteStateProjection",
            "CellMLToolkit", "SBMLToolkit",
            "BaseModelica", "AudioPlugins",
            "ReactionNetworkImporters",
            "DiffEqPhysics", "DiffEqFinancial",
            "PubChem", "Pyomo", "MathML",
        ],
        "Symbolic Tools" => [
            "ModelOrderReduction", "Symbolics", "SymbolicUtils",
            "SymbolicIntegration", "SymbolicSMT", "SymbolicLimits",
            "SymbolicAnalysis", "FunctionProperties",
        ], #="MetaTheory"=#
        #=
        "Third-Party Modeling Tools" => ["Agents", "Unitful",
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
            "DifferenceEquations", "Optimization", "JumpProcesses", "LineSearch",
            "Evolutionary", "NeuralLinearSolve", "Corleone",
        ],
        #=
        "Third-Party Equation Solvers" => [
            "LowRankIntegrators",
            "FractionalDiffEq",
            "ManifoldDiffEq",
        ],
        =#
        "Inverse Problems / Estimation" => [
            "CurveFit", "SciMLSensitivity", "DiffEqParamEstim", "DiffEqBayes",
        ],
        "PDE Solvers" => [
            "MethodOfLines", "NeuralPDE",
            "NeuralOperators", "FEniCS",
            "HighDimPDE", "DiffEqOperators",
            "FiniteVolumeMethod", "FiniteVolumeMethod1D",
        ],
        #=
        "Third-Party PDE Solvers" => [
            "Trixi",
            "Ferrite",
            "Gridap",
            "ApproxFun",
            "VoronoiFVM",
        ],
        =#
        "Advanced Solver APIs" => [
            "OrdinaryDiffEq", "BoundaryValueDiffEq", "DiffEqGPU",
            "SteadyStateDiffEq", "OrdinaryDiffEqOperatorSplitting",
            "IRKGaussLegendre", "MATLABDiffEq", "QuantumNLDiffEq",
        ],
    ],
    "Analysis" => [
        #= "Plots and Visualization" => ["Makie"], #="PlotDocs",=# =#
        "Parameter Analysis" => [
            "EasyModelAnalysis", "GlobalSensitivity", "StructuralIdentifiability",
            "MinimallyDisruptiveCurves", "CatalystNetworkAnalysis",
        ],
        "Third-Party Parameter Analysis" => ["BifurcationKit"],
        #= "DynamicalSystems", "ControlSystems", "ReachabilityAnalysis"], =#
        "Uncertainty Quantification" => [
            "PolyChaos", "SciMLExpectations", "OptimalUncertaintyQuantification",
        ],
        #=
        "Third-Party Uncertainty Quantification" => ["Measurements",
            "MonteCarloMeasurements",
            "ProbNumDiffEq", "TaylorIntegration",
            "IntervalArithmetic"],
        =#
    ],
    "Machine Learning" => [
        "Function Approximation" => ["Surrogates", "ReservoirComputing"],
        "Implicit Layer Deep Learning" => ["DiffEqFlux", "DeepEquilibriumNetworks", "NeuralLyapunov"],
        #= "Third-Party Implicit Layer Deep Learning" => ["Flux", "Lux", "SimpleChains"], =#
        "Symbolic Learning" => ["DataDrivenDiffEq", "SymbolicNumericIntegration"],
        #= "Third-Party Symbolic Learning" => ["SymbolicRegression"], =#
        "Third-Party Differentiation Tooling" => [
            "SparseDiffTools", "FiniteDiff",
            #= "ForwardDiff",
            "Zygote", "Enzyme" =#
        ],
    ],
    "Developer Tools" => [
        "Numerical Utilities" => [
            "ExponentialUtilities", "DiffEqNoiseProcess",
            "PreallocationTools", "EllipsisNotation", "DataInterpolations", "DataInterpolationsND",
            "PoissonRandom", "QuasiMonteCarlo", "RuntimeGeneratedFunctions", "MuladdMacro", "FindFirstFunctions",
            "SparseDiffTools", "BipartiteGraphs",
            "FastAlmostBandedMatrices", "FastBroadcast", "FunctionWrappersWrappers",
            "LHLFactorization", "LightweightStats",
            "PureGebal", "PureKLU", "PureUMFPACK",
            "RootedTrees", "SparseBandedMatrices", "RespecializeParams",
            "ConcreteStructs",
        ],
        #=
        "Third-Party Numerical Utilities" => ["FFTW", "Distributions",
            "SpecialFunctions", "LoopVectorization",
            "Polyester", ], #="Tullio"=#
        =#
        "High-Level Interfaces" => [
            "SciMLBase",
            "SciMLStructures",
            "SciMLLogging",
            "ADTypes",
            "SymbolicIndexingInterface",
            "TermInterface",
            "SciMLOperators",
            "SurrogatesBase",
            "CommonSolve",
            "SciMLIterators",
            "Static",
        ],
        "Third-Party Interfaces" => ["ArrayInterface", "StaticArrayInterface" #= "AbstractFFTs",
            "GPUArrays", #= "Adapt", =#
            "Tables" =#],                                 #= "RecipesBase", =#
        "Developer Documentation" => [
            "SciMLStyle", "ColPrac", "DiffEqDevDocs", "OrgMaintenanceScripts",
        ],
        "Extra Resources" => [
            "SciMLWorkshop",
            "SciMLTutorialsOutput",
            "SciMLBenchmarksOutput",
            "ModelingToolkitCourse",
        ],
    ],
]

fixnames = Dict(
    "SciMLDocs" => "The SciML Open Source Software Ecosystem",
    "DiffEqDocs" => "DifferentialEquations",
    "DiffEqDevDocs" => "DiffEq Developer Documentation",
    "PlotDocs" => "Plots",
    "SciMLBenchmarksOutput" => "The SciML Benchmarks",
    "SciMLTutorialsOutput" => "Extended SciML Tutorials"
)
hasnojl = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput", "ColPrac", "SciMLStyle", "ModelingToolkitCourse"]
usemain = ["SciMLBenchmarksOutput", "SciMLTutorialsOutput"]

external_urls = Dict(
    "Enzyme" => "https://github.com/EnzymeAD/Enzyme.jl",
    "Zygote" => "https://github.com/FluxML/Zygote.jl",
    "FiniteDiff" => "https://github.com/JuliaDiff/FiniteDiff.jl",
    "ForwardDiff" => "https://github.com/JuliaDiff/ForwardDiff.jl",
    "SparseDiffTools" => "https://github.com/JuliaDiff/SparseDiffTools.jl",
    "ManifoldDiffEq" => "https://github.com/JuliaManifolds/ManifoldDiffEq.jl",
    "FractionalDiffEq" => "https://github.com/SciFracX/FractionalDiffEq.jl",
    "Agents" => "https://github.com/JuliaDynamics/Agents.jl",
    "LowRankIntegrators" => "https://github.com/FHoltorf/LowRankIntegrators.jl",
    "Trixi" => "https://github.com/trixi-framework/Trixi.jl",
    "Gridap" => "https://github.com/gridap/Gridap.jl",
    "Ferrite" => "https://github.com/Ferrite-FEM/Ferrite.jl",
    "ApproxFun" => "https://github.com/JuliaApproximation/ApproxFun.jl",
    "VoronoiFVM" => "https://github.com/j-fu/VoronoiFVM.jl",
    "Symbolics" => "https://github.com/JuliaSymbolics/Symbolics.jl",
    "SymbolicSMT" => "https://github.com/JuliaSymbolics/SymbolicSMT.jl",
    "SymbolicIntegration" => "https://github.com/JuliaSymbolics/SymbolicIntegration.jl",
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
    "AlgebraicPetri" => "https://github.com/AlgebraicJulia/AlgebraicPetri.jl"
)

# The runner pods are evicted once their ephemeral storage grows past a few
# tens of GiB, and `git clone --depth 1` of ~126 gh-pages branches lands ~54
# GiB on disk between worktrees and .git packs, which is what gets the pod
# killed mid-clone. Fetching a codeload tarball per repository and keeping
# only the root files and the version directories each MultiDocRef selects
# shrinks that to ~10 GiB and skips .git entirely; maybe_clone uses an
# existing upstream directory as-is when it has no .git.
function site_url(giturl)
    m = match(r"^https://github\.com/([^/]+)/(.+?)(?:\.git)?$", giturl)
    return m === nothing ? nothing : "https://$(lowercase(m[1])).github.io/$(m[2])/"
end

function fetch_docs(ref::MultiDocumenter.MultiDocRef)
    (isempty(ref.giturl) || isdir(ref.upstream)) && return
    m = match(r"^https://github\.com/(.+?)(?:\.git)?$", ref.giturl)
    m === nothing && return
    wanted = ref.versions === nothing ? nothing : Set(ref.versions.versions)
    mkpath(clonedir)
    tmp = mktempdir(clonedir)
    try
        tarball = "https://codeload.github.com/$(m[1])/tar.gz/refs/heads/$(ref.branch)"
        @info "Fetching $(tarball) for $(ref.name)"
        run(pipeline(`curl -fsSL --retry 3 $(tarball)`, `tar -xz -C $(tmp)`))
        src = only(readdir(tmp; join = true))
        mkpath(ref.upstream)
        moved = String[]
        for entry in readdir(src)
            full = joinpath(src, entry)
            (isfile(full) || wanted === nothing || entry in wanted) || continue
            # stable is normally a symlink to a version directory; the link
            # would dangle once the target is not kept, so dereference it
            real = islink(full) ? normpath(joinpath(src, readlink(full))) : full
            ispath(real) || continue
            dir = isdir(real)
            mv(real, joinpath(ref.upstream, entry); force = true)
            dir && push!(moved, entry)
        end
        # a selection that matched nothing upstream would leave the ref empty;
        # keeping every directory degrades to what happens without a selection
        if wanted !== nothing && isempty(moved)
            for entry in readdir(src)
                full = joinpath(src, entry)
                isdir(full) || continue
                real = islink(full) ? normpath(joinpath(src, readlink(full))) : full
                ispath(real) && mv(real, joinpath(ref.upstream, entry); force = true)
            end
        end
    catch e
        rm(ref.upstream; force = true, recursive = true)
        @warn "Tarball fetch failed for $(ref.name); MultiDocumenter will `git clone` it instead" exception =
            (e, catch_backtrace())
    finally
        rm(tmp; force = true, recursive = true)
        run(`sync`)
    end
    return
end

home = MultiDocumenter.MultiDocRef(
    upstream = joinpath(clonedir, "Home"),
    path = "Overview",
    name = "Home",
    giturl = "https://github.com/SciML/SciMLDocs.git",
    versions = MultiDocumenter.VersionSelection(
        ["Overview"];
        all_versions_url = "https://docs.sciml.ai/"
    )
)
docs = Any[home]
refs = MultiDocumenter.MultiDocRef[home]

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
            ref = MultiDocumenter.MultiDocRef(
                upstream = joinpath(clonedir, mod),
                path = mod,
                name = mod in keys(fixnames) ? fixnames[mod] :
                    mod,
                giturl = url,
                branch = mod ∈ usemain ? "main" : "gh-pages",
                versions = mod ∈ usemain ? nothing :
                    MultiDocumenter.VersionSelection(
                        ["stable", "dev"];
                        all_versions_url = site_url(url)
                    )
            )
            push!(docsites, ref)
            push!(refs, ref)
        end
        push!(docgroups, MultiDocumenter.Column(cat[1], docsites))
    end
    push!(docs, MultiDocumenter.MegaDropdownNav(group[1], docgroups))
end

push!(
    docs, MultiDocumenter.MegaDropdownNav(
        "Commercial Support",
        [
            MultiDocumenter.Column("Commercial Support", [JuliaHubCommercialSupportComponent("https://juliahub.com/company/contact-us-sciml-docs")]),
            MultiDocumenter.Column("Products built with SciML", [ProductsUsedComponent()]),
        ]
    )
)

foreach(fetch_docs, refs)

outpath = joinpath(@__DIR__, "build")

MultiDocumenter.make(
    outpath, docs;
    assets_dir = "docs/src/assets",
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = [
            "stable",
        ],
        engine = MultiDocumenter.FlexSearch
    ),
    custom_scripts = [
        "https://www.googletagmanager.com/gtag/js?id=G-Q3FE4BYYHQ",
        Docs.HTML(
            """
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-Q3FE4BYYHQ');
            """
        ),
    ],
    brand_image = MultiDocumenter.BrandImage(
        "https://sciml.ai",
        joinpath(
            "assets",
            "logo.png"
        )
    )
)
