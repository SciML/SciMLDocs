using Documenter

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
                           :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                                        "packages" => [
                                            "base",
                                            "ams",
                                            "autoload",
                                            "mathtools",
                                            "require",
                                        ])))

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
                                  canonical = "https://docs.sciml.ai/stable/",
                                  mathengine = mathengine),
         pages = [
            "SciML: Open Source Software for Scientific Machine Learning with Julia" => "index.md",
            "Getting Started" => [
                "getting_started/getting_started.md",
                "New User Tutorials" => [
                    "getting_started/installation.md",
                    "getting_started/first_simulation.md",
                    "getting_started/first_optimization.md",
                    "getting_started/fit_simulation.md",
                    "getting_started/find_root.md",
                    "getting_started/integral_approx.md",
                ],
                "Comparison With Other Tools" => [
                    "comparisons/python.md",
                    "comparisons/matlab.md",
                    "comparisons/r.md",
                    "comparisons/cppfortran.md",
                ],
            ],
            "What is SciML?" => ["overview.md",
            "Solvers" => ["highlevels/equation_solvers.md",
                          "highlevels/inverse_problems.md",
                          "highlevels/partial_differential_equation_solvers.md"],
            "Modeling Tools" => ["highlevels/modeling_languages.md",
                                 "highlevels/model_libraries_and_importers.md",
                                 "highlevels/symbolic_tools.md",
                                 "highlevels/array_libraries.md"],
            "Simulation Analysis" => ["highlevels/parameter_analysis.md",
                                      "highlevels/uncertainty_quantification.md",
                                      "highlevels/plots_visualization.md"],
            "Machine Learning" => ["highlevels/function_approximation.md",
                                   "highlevels/implicit_layers.md",
                                   "highlevels/symbolic_learning.md"],
            "Developer Tools" => ["highlevels/numerical_utilities.md",
                                  "highlevels/interfaces.md",
                                  "highlevels/developer_documentation.md"],
            "Extra Learning Resources" => ["highlevels/learning_resources.md"],
        ]])

deploydocs(repo = "github.com/SciML/SciMLDocs")
