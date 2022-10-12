using Documenter

makedocs(sitename = "Overview of Julia's SciML",
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
            "Solvers" => ["highlevels/equation_solvers.md",
                          "highlevels/inverse_problems.md",
                          "highlevels/partial_differential_equation_solvers.md"],
            "Modeling Tools" => ["highlevels/modeling_languages.md",
                                 "highlevels/model_libraries_and_importers.md",
                                 "highlevels/symbolic_tools.md",
                                 "highlevels/array_libraries.md"],
            "Simulation Analysis" => ["highlevels/parameter_analysis.md",
                                      "highlevels/uncertainty_quantification.md",
                                      "highlevels/plots_visualization.md",
                                      ],
            "Machine Learning" => ["highlevels/implicit_layers.md",
                                   "highlevels/function_approximation.md",
                                   "highlevels/symbolic_learning.md"],
            "Developer Tools" => ["highlevels/numerical_utilities.md",
                                  "highlevels/interfaces.md",
                                  "highlevels/developer_documentation.md"],
            "Extra Learning Resources" => ["highlevels/learning_resources.md"],
        ])

deploydocs(repo = "github.com/SciML/SciMLDocs")
