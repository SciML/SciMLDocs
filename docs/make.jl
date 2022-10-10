using Documenter

makedocs(sitename = "Overview of SciML",
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
