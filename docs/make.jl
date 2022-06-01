using SciMLBase, LinearSolve, NonlinearSolve
using Documenter

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
        @show pages[i],1
        pages[i] = joinpath(str,pages[i])
    end
    pages
end

function recursive_append(pages::AbstractArray{<:Pair{String,String}},str)
    for i in eachindex(pages) 
        @show pages[i],2
        pages[i] = pages[i][1] => joinpath(str,pages[i][2])
    end
    pages
end

function recursive_append(pages::AbstractArray{<:Any},str)
    for i in eachindex(pages) 
        @show pages[i],3
        if pages[i] isa Pair
            pages[i] = pages[i][1] => joinpath(str,pages[i][2])
        elseif pages[i] isa String
            pages[i] = joinpath(str,pages[i])
        end
    end
    pages
end

SciMLBase_pages=[
    "Home" => "index.md",
    "Interfaces" => Any[
        "interfaces/Problems.md",
        "interfaces/SciMLFunctions.md",
        "interfaces/Algorithms.md",
        "interfaces/Solutions.md",
        "interfaces/Init_Solve.md",
        "interfaces/Common_Keywords.md",
        "interfaces/Differentiation.md",
        "interfaces/PDE.md",
    ],
    "Fundamentals" => Any[
        "fundamentals/FAQ.md"
    ]
]

LinearSolve_pages=[
    "Home" => "index.md",
    "Tutorials" => Any[
        "tutorials/linear.md"
        "tutorials/caching_interface.md"
    ],
    "Basics" => Any[
        "basics/LinearProblem.md",
        "basics/common_solver_opts.md",
        "basics/CachingAPI.md",
        "basics/Preconditioners.md",
        "basics/FAQ.md"
    ],
    "Solvers" => Any[
        "solvers/solvers.md"
    ],
    "Advanced" => Any[
        "advanced/developing.md"
        "advanced/custom.md"
    ]
]

NonlinearSolve_pages=[
    "Home" => "index.md",
    "Tutorials" => Any[
        "tutorials/nonlinear.md",
        "tutorials/iterator_interface.md"
    ],
    "Basics" => Any[
        "basics/NonlinearProblem.md",
        "basics/NonlinearFunctions.md",
        "basics/FAQ.md"
    ],
    "Solvers" => Any[
        "solvers/NonlinearSystemSolvers.md",
        "solvers/BracketingSolvers.md"
    ]
]

SciMLBase_pages = recursive_append(SciMLBase_pages,joinpath(pkgdir(SciMLBase),"docs","src"))
LinearSolve_pages = recursive_append(LinearSolve_pages,joinpath(pkgdir(LinearSolve),"docs","src"))
NonlinearSolve_pages = recursive_append(NonlinearSolve_pages,joinpath(pkgdir(NonlinearSolve),"docs","src"))

fullpages = ["SciMLBase" => SciMLBase_pages,
             "LinearSolve" => LinearSolve_pages,
             "NonlinearSolve" => NonlinearSolve_pages]

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
