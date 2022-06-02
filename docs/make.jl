using SciMLDocs, Documenter
using Plots

# Ordering Matters!
docsmodules = ["Interfaces" => ["SciMLBase"],
              "Equation Solvers" => ["LinearSolve", "NonlinearSolve", "Integrals"],
              "Utilities" => ["Surrogates"]
]


fullpages = Vector{Any}()
allmods = Vector{Any}()

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
        if pages[i] isa Pair && pages[i][2] isa String
            pages[i] = pages[i][1] => joinpath(str,pages[i][2])
        elseif pages[i] isa Pair && pages[i][2] isa AbstractArray
            pages[i] = pages[i][1] => recursive_append(pages[i][2],str)
        elseif pages[i] isa String
            pages[i] = joinpath(str,pages[i])
        else
            error("wait what?")
        end
    end
    pages
end

global catpage

for cat in docsmodules
    catpage = Vector{Any}()
    for mod in cat[2]
        ex = quote
            using $(Symbol(mod))
            cp(joinpath(pkgdir($(Symbol(mod))),"docs","src"),joinpath(pkgdir(SciMLDocs),"docs","src","copies",$mod),force=true)
            include(joinpath(pkgdir($(Symbol(mod))),"docs","pages.jl"))
            push!(allmods,$(Symbol(mod)))
            push!(catpage,$mod => recursive_append(pages,joinpath("copies",$mod)))
        end
        @eval $ex
    end
    push!(fullpages, cat[1] => catpage)
end

push!(allmods,Plots)

makedocs(
    sitename="The SciML Open Source Software Ecosystem",
    authors="Chris Rackauckas",
    modules=identity.(allmods),
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
