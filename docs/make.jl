using SciMLDocs, Documenter

# Ordering Matters!
docsmodules = ["SciMLBase", "LinearSolve", "NonlinearSolve"]
fullpages = Vector{Any}()

function recursive_append(pages::AbstractArray{<:AbstractArray},str)
    map(recursive_append,pages,str)
end

function recursive_append(pages::AbstractArray{<:Pair{String,Any}},str)
    for i in eachindex(pages)
        if pages[i][2] isa AbstractArray
            pages[i] = pages[i][1] => recursive_append(pages[i][2],str)
        elseif pages[i][2] isa String
            pages[i] = pages[i][1] => joinpath(str,pages[i][2])
        end
    end
    pages
end

function recursive_append(pages::AbstractArray{<:String},str)
    for i in eachindex(pages) 
        pages[i] = joinpath(str,pages[i])
    end
    pages
end

function recursive_append(pages::AbstractArray{<:Pair{String,String}},str)
    for i in eachindex(pages) 
        pages[i] = pages[i][1] => joinpath(str,pages[i][2])
    end
    pages
end

function recursive_append(pages::AbstractArray{<:Any},str)
    for i in eachindex(pages) 
        if pages[i] isa Pair
            pages[i] = pages[i][1] => joinpath(str,pages[i][2])
        elseif pages[i] isa String
            pages[i] = joinpath(str,pages[i])
        end
    end
    pages
end

for mod in docsmodules
    ex = quote
        using $(Symbol(mod))
        cp(joinpath(pkgdir($(Symbol(mod))),"docs","src"),joinpath(pkgdir(SciMLDocs),"docs","src","copies",$mod),force=true)
        include(joinpath(pkgdir($(Symbol(mod))),"docs","pages.jl"))
        push!(fullpages,$mod => recursive_append(pages,joinpath("copies",$mod)))
    end
    @eval $ex 
end

makedocs(
    sitename="The SciML Open Source Software Ecosystem",
    authors="Chris Rackauckas",
    modules=[SciMLBase,LinearSolve,NonlinearSolve],
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://docs.sciml.ai/stable/"),
    pages=fullpages
)

deploydocs(;
    repo="github.com/SciML/SciMLDocs.jl",
    devbranch="main",
)
