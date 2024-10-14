# [Installing SciML Software](@id installation)

## Step 1: Install Julia

Download Julia using [this website](https://julialang.org/downloads/).

!!! note
    
    Some Linux distributions do weird and incorrect things with Julia installations!
    Please install Julia using the binaries provided by the [official JuliaLang website!](https://julialang.org/downloads/)

To ensure that you have installed Julia correctly, open it up and type `versioninfo()` in
the REPL. It should look like the following:

![](https://user-images.githubusercontent.com/1814174/195772770-e8f7b8f8-a853-4a95-a5c5-c6ed40f0c8d9.PNG)

(with the CPU/OS/etc. details matching your computer!)

If you got stuck in this installation process, ask for help
[on the Julia Discourse](https://discourse.julialang.org/) or in
[the Julia Zulip chatrooms](https://julialang.zulipchat.com/#)

## Optional Step 1.5: Get VS Code Setup with the Julia Extension

You can run SciML with Julia in any development environment you please, but our recommended
environment is VS Code. For more information on using Julia with VS Code, check out the
[Julia VS Code Extension website](https://www.julia-vscode.org/). Let's install it!

First [download VS Code from the official website](https://code.visualstudio.com/).

Next, open Visual Studio Code and click Extensions.

![](https://user-images.githubusercontent.com/1814174/195773680-2226b2fc-5903-4eff-aae2-4d8689f16280.PNG)

Then, search for “Julia” in the search bar on the top of the extension tab, click on the
“Julia” extension, and click the install button on the tab that opens up.

![](https://user-images.githubusercontent.com/1814174/195773697-ede4edee-d479-46e8-acce-94a3ff884de8.PNG)

To make sure your installation is correct, try running some code. Open a new file by
either going to the top left navigation bar `File |> New Text File`, or hitting `Ctrl+n`.
Name your new file `test.jl` (**important: the Julia VS Code functionality only turns on
when using a `.jl` file!**). Next, type 1+1 and hit `Ctrl+Enter`. A Julia REPL should
pop up and the result 2 should be displayed. Your environment should look something like
this:

![](https://user-images.githubusercontent.com/1814174/195774555-5841918e-e9a5-443c-9eca-84ed932af355.PNG)

For more help on using the VS Code editor with Julia, check out the
[VS Code in Julia documentation](https://www.julia-vscode.org/docs/stable/). Useful
[keyboard commands can be found here](https://www.julia-vscode.org/docs/stable/userguide/keybindings/).

Once again, if you got stuck in this installation process, ask for help
[on the Julia Discourse](https://discourse.julialang.org/) or in
[the Julia Zulip chatrooms](https://julialang.zulipchat.com/#)

## Step 2: Install a SciML Package

SciML is [over 130 Julia packages](https://github.com/SciML). That's too much stuff to
give someone in a single download! Thus instead, the SciML organization divides its
functionality into composable modules that can be mixed and matched as required. Installing
SciML ecosystem functionality is equivalent to installation of such packages.

For example, do you need the differential equation solver? Then install DifferentialEquations
via the command:

```julia
using Pkg;
Pkg.add("DifferentialEquations");
```

in the Julia REPL. Or, for a more robust REPL experience, hit the `]` command to make the
blue `pkg>` REPL environment start, and type in `add DifferentialEquations`. The package
REPL environment will have nice extras like auto-complete that will be useful in the future.
This command should run an installation sequence and precompile all of the
packages (precompile = "run a bunch of performance optimizations!"). Don't be surprised
if this installation process takes ~10 minutes on older computers. During the installation,
it should look like this:

![](https://user-images.githubusercontent.com/1814174/195775465-9e80de11-0b1e-4229-9eba-f5e49c9c81a1.PNG)

And that's it!

## How do I test that my installed correctly?

The best way is to [build and run your first simulation!](@ref first_sim)
