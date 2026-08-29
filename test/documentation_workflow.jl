@testset "Documentation workflow environments" begin
    root = pkgdir(SciMLDocs)
    workflow = read(joinpath(root, ".github", "workflows", "Documentation.yml"), String)
    aggregate_project_path = joinpath(root, "docs", "aggregate", "Project.toml")

    @test isfile(aggregate_project_path)
    if isfile(aggregate_project_path)
        aggregate_deps = keys(TOML.parsefile(aggregate_project_path)["deps"])
        @test Set(aggregate_deps) == Set(["Documenter", "LibGit2", "MultiDocumenter", "Pkg"])
    end
    @test occursin("--project=docs/aggregate", workflow)
    @test occursin("if: github.event_name != 'schedule'", workflow)
end

@testset "Executable documentation workload" begin
    root = pkgdir(SciMLDocs)
    bnode = read(joinpath(root, "docs", "src", "showcase", "bayesian_neural_ode.md"), String)
    sample_count = match(r"n_samples = (\d+)", bnode)

    @test sample_count !== nothing
    if sample_count !== nothing
        @test parse(Int, only(sample_count.captures)) <= 100
    end
end
