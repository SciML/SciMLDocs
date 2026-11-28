function strip_julia_comments(s::AbstractString)
    out = IOBuffer()
    i = firstindex(s)
    n = lastindex(s)
    while i <= n
        if i < n && s[i] == '#' && s[nextind(s, i)] == '='
            depth = 1
            i = nextind(s, i, 2)
            while i < n && depth > 0
                nxt = nextind(s, i)
                if s[i] == '#' && s[nxt] == '='
                    depth += 1
                    i = nextind(s, i, 2)
                elseif s[i] == '=' && s[nxt] == '#'
                    depth -= 1
                    i = nextind(s, i, 2)
                else
                    i = nxt
                end
            end
        elseif s[i] == '#'
            while i <= n && s[i] != '\n'
                i = nextind(s, i)
            end
        else
            print(out, s[i])
            i = nextind(s, i)
        end
    end
    return String(take!(out))
end

@testset "Package inventory coverage" begin
    root = pkgdir(SciMLDocs)
    inventory_path = joinpath(root, "docs", "package_inventory.toml")
    aggregate_path = joinpath(root, "docs", "make_aggregate.jl")
    index_path = joinpath(root, "docs", "src", "highlevels", "package_index.md")
    make_path = joinpath(root, "docs", "make.jl")

    inventory = TOML.parsefile(inventory_path)["packages"]
    aggregate = read(aggregate_path, String)
    index = read(index_path, String)
    make = read(make_path, String)

    @test !isempty(inventory)
    @test occursin("highlevels/package_index.md", make)

    missing_from_index = String[]
    missing_from_dropdown = String[]
    inventoried_dropdown = Set{String}()
    for (repo, meta) in inventory
        occursin(repo, index) || push!(missing_from_index, repo)
        dropdown = get(meta, "dropdown", nothing)
        if dropdown !== nothing
            push!(inventoried_dropdown, dropdown)
            occursin("\"" * dropdown * "\"", aggregate) ||
                push!(missing_from_dropdown, dropdown)
        end
    end
    @test missing_from_index == String[]
    @test missing_from_dropdown == String[]

    ext_start = findfirst("external_urls = Dict(", aggregate)
    ext_stop = findfirst("docs = Any[", aggregate)
    @test ext_start !== nothing && ext_stop !== nothing
    ext_block = aggregate[first(ext_start):first(ext_stop)]
    third_party = Set{String}()
    for m in eachmatch(r"\"([A-Za-z0-9]+)\"\s*=>\s*\"https://", ext_block)
        push!(third_party, m.captures[1])
    end

    docs_start = findfirst("docsmodules = [", aggregate)
    docs_stop = findfirst("fixnames = Dict", aggregate)
    @test docs_start !== nothing && docs_stop !== nothing
    docsmodules = strip_julia_comments(aggregate[first(docs_start):first(docs_stop)])

    uninventoried = String[]
    seen = Set{String}()
    for m in eachmatch(r"\"([A-Za-z][A-Za-z0-9]*)\"(?!\s*=>)", docsmodules)
        name = m.captures[1]
        name in seen && continue
        push!(seen, name)
        name in third_party && continue
        name in inventoried_dropdown && continue
        push!(uninventoried, name)
    end
    @test uninventoried == String[]
end
