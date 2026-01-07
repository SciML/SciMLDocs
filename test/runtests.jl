using Test
using ExplicitImports
using SciMLDocs

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(SciMLDocs) === nothing
    @test check_no_stale_explicit_imports(SciMLDocs) === nothing
end
