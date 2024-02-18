using Documenter

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

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
         clean = true, doctest = false, linkcheck = true,
         linkcheck_ignore = ["https://twitter.com/ChrisRackauckas/status/1477274812460449793",
                             "https://epubs.siam.org/doi/10.1137/0903023"],
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
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
                 ],
                 "Comparison With Other Tools" => [
                     "comparisons/python.md",
                     "comparisons/matlab.md",
                     "comparisons/r.md",
                     "comparisons/cppfortran.md",
                 ],
             ],
             "Showcase of Cool Examples" => Any["showcase/showcase.md",
                                                "Automated Model Discovery" => Any["showcase/missing_physics.md",
                                                                                   "showcase/bayesian_neural_ode.md",
                                                                                   "showcase/blackhole.md"],
                                                "Solving Difficult Equations Efficiently" => Any["showcase/brusselator.md",
                                                                                                 "showcase/pinngpu.md",
                                                                                                 "showcase/massively_parallel_gpu.md",
                                                                                                 "showcase/gpu_spde.md"],
                                                "Useful Cool Wonky Things" => Any["showcase/ode_types.md",
                                                                                  "showcase/symbolic_analysis.md",
                                                                                  "showcase/optimization_under_uncertainty.md"]],
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
