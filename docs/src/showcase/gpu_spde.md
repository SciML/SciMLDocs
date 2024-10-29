# [GPU-Accelerated Stochastic Partial Differential Equations](@id gpuspde)

Let's solve stochastic PDEs in Julia using GPU parallelism. To do this, we will use the
type-genericness of the DifferentialEquations.jl library in order to write a code that uses
within-method GPU-parallelism on the system of PDEs. The OrdinaryDiffEq.jl solvers of
DifferentialEquations.jl, including implicit solvers with GMRES, etc., and the same for SDEs,
DAEs, DDEs, etc. are all GPU-compatible with a fast form of broadcast.

!!! note
    
    The non-native Julia solvers, like Sundials are incompatible with arbitrary input
    types and thus not compatible with GPUs.

Let's dive into showing how to accelerate ODE solving with GPUs!

## Before we start: the two ways to accelerate ODE solvers with GPUs

Before we dive deeper, let us remark that there are two very different ways that one can
accelerate an ODE solution with GPUs. There is one case where `u` is very big and `f`
is very expensive, but very structured, and you use GPUs to accelerate the computation
of said `f`. The other use case is where `u` is very small but you want to solve the ODE
`f` over many different initial conditions (`u0`) or parameters `p`. In that case, you can
use GPUs to parallelize over different parameters and initial conditions. In other words:

| Type of Problem                           | SciML Solution                                                                                           |
|:----------------------------------------- |:-------------------------------------------------------------------------------------------------------- |
| Accelerate a big ODE                      | Use [CUDA.jl's](https://cuda.juliagpu.org/stable/) CuArray as `u0`                                       |
| Solve the same ODE with many `u0` and `p` | Use [DiffEqGPU.jl's](https://docs.sciml.ai/DiffEqGPU/stable/) `EnsembleGPUArray` and `EnsembleGPUKernel` |

This showcase will focus on the former case. For the latter, see the
[massively parallel GPU ODE solving showcase](@ref datagpu).

## Our Problem: 2-dimensional Reaction-Diffusion Equations

The reaction-diffusion equation is a PDE commonly handled in systems biology, which is a diffusion equation plus a nonlinear reaction term. The dynamics are defined as:

```math
u_t = D \Delta u + f(t,u)
```

But this doesn't need to only have a single “reactant” u: this can be a vector of reactants
and the ``f`` is then the nonlinear vector equations describing how these different pieces
react together. Let's settle on a specific equation to make this easier to explain. Let's use
a simple model of a 3-component system where A can diffuse through space to bind with the
non-diffusive B to form the complex C (also non-diffusive, assume B is too big and gets stuck
in a cell which causes C=A+B to be stuck as well). Other than the binding, we make each of
these undergo a simple birth-death process, and we write down the equations which result from
mass-action kinetics. If this all is meaningless to you, just understand that it gives the
system of PDEs:

```math
\begin{align}
A_t &= D \Delta A + \alpha_A(x) - \beta_A  A - r_1 A B + r_2 C\\
B_t &= \alpha_B - \beta_B B - r_1 A B + r_2 C\\
C_t &= \alpha_C - \beta_C C + r_1 A B - r_2 C
\end{align}
```

One addition that was made to the model is that we let ``\alpha_A(x)`` be the production of
``A``, and we let that be a function of space so that way it only is produced on one side of
our equation. Let's make it a constant when x>80, and 0 otherwise, and let our spatial domain
be ``x \in [0,100]`` and ``y \in [0,100]``.

This model is spatial: each reactant ``u(t,x,y)`` is defined at each point in space, and all
of the reactions are local, meaning that ``f`` at spatial point ``(x,y)`` only uses
``u_i(t,x,y)``. This is an important fact which will come up later for parallelization.

## Discretizing the PDE into ODEs

In order to solve this via a method of lines (MOL) approach, we need to discretize the PDE
into a system of ODEs. Let's do a simple uniformly-spaced grid finite difference
discretization. Choose ``dx = 1`` and ``dy = 1`` so that we have `100*100=10000` points for
each reactant. Notice how fast that grows! Put the reactants in a matrix such that
`A[i,j] = A(x_j,y_i)`, i.e. the columns of the matrix are the ``x`` values and the rows are
the ``y`` values (this way looking at the matrix is essentially like looking at the
discretized space).

So now we have 3 matrices (`A`, `B`, and `C`) for our reactants. How do we discretize the
PDE? In this case, the diffusion term simply becomes a tridiagonal matrix ``M`` where
``[1,-2,1]`` is the central band. You can notice that ``MA`` performs diffusion along the
columns of ``A``, and so this is diffusion along the ``y``. Similarly, ``AM`` flips the
indices and thus does diffusion along the rows of ``A`` making this diffusion along ``x``.
Thus ``D(M_yA + AM_x)`` is the discretized Laplacian (we could have separate diffusion
constants and ``dx \neq dy`` if we want by using different constants on the ``M``, but let's
not do that for this simple example. We leave that as an exercise for the reader). We
enforced a Neumann boundary condition with zero derivative (also known as a no-flux boundary
condition) by reflecting the changes over the boundary. Thus the derivative operator is
generated as:

```@example spde
using LinearAlgebra

# Define the constants for the PDE
const α₂ = 1.0
const α₃ = 1.0
const β₁ = 1.0
const β₂ = 1.0
const β₃ = 1.0
const r₁ = 1.0
const r₂ = 1.0
const D = 100.0
const γ₁ = 0.1
const γ₂ = 0.1
const γ₃ = 0.1
const N = 100
const X = reshape([i for i in 1:100 for j in 1:100], N, N)
const Y = reshape([j for i in 1:100 for j in 1:100], N, N)
const α₁ = 1.0 .* (X .>= 80)

const Mx = Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
    [1.0 for i in 1:(N - 1)])
const My = copy(Mx)
# Do the reflections, different for x and y operators
Mx[2, 1] = 2.0
Mx[end - 1, end] = 2.0
My[1, 2] = 2.0
My[end, end - 1] = 2.0
```

!!! note
    
    We could have also done these discretization steps using
    [DiffEqOperators.jl](https://docs.sciml.ai/DiffEqOperators/stable/) or
    [MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/). However, we are
    going to keep it in this form, so we can show the full code, making it easier to see
    how to define GPU-ready code!

Since all of the reactions are local, we only have each point in space react separately.
Thus this represents itself as element-wise equations on the reactants. Thus we can write
it out quite simply. The ODE which then represents the PDE is thus in pseudo Julia code:

```julia
DA = D * (Mx * A + A * My)
@. DA + α₁ - β₁ * A - r₁ * A * B + r₂ * C
@. α₂ - β₂ * B - r₁ * A * B + r₂ * C
@. α₃ - β₃ * C + r₁ * A * B - r₂ * C
```

Note here that I am using α₁ as a matrix (or row-vector, since that will broadcast just
fine) where every point in space with x<80 has this zero, and all of the others have it as
a constant. The other coefficients are all scalars.

How do we do this with the ODE solver?

## Our Representation via Views of 3-Tensors

We can represent our problem with a 3-dimensional tensor, taking each 2-dimensional slice
as our (A,B,C). This means that we can define:

```@example spde
u0 = zeros(N, N, 3);
```

Now we can decompose it like:

```julia
A = @view u[:, :, 1]
B = @view u[:, :, 2]
C = @view u[:, :, 3]
dA = @view du[:, :, 1]
dB = @view du[:, :, 2]
dC = @view du[:, :, 3]
```

These views will not construct new arrays and will instead just be pointers to the
(contiguous) memory pieces, so this is a nice and efficient way to handle this. Together,
our ODE using this tensor as its container can be written as follows:

```@example spde
function f(du, u, p, t)
    A = @view u[:, :, 1]
    B = @view u[:, :, 2]
    C = @view u[:, :, 3]
    dA = @view du[:, :, 1]
    dB = @view du[:, :, 2]
    dC = @view du[:, :, 3]
    DA = D * (Mx * A + A * My)
    @. dA = DA + α₁ - β₁ * A - r₁ * A * B + r₂ * C
    @. dB = α₂ - β₂ * B - r₁ * A * B + r₂ * C
    @. dC = α₃ - β₃ * C + r₁ * A * B - r₂ * C
end
```

where this is using @. to do inplace updates on our du to say how the full tensor should
update in time. Note that we can make this more efficient by adding some cache variables to
the diffusion matrix multiplications and using mul!, but let's ignore that for now.

Together, the ODE which defines our PDE is thus:

```@example spde
using DifferentialEquations

prob = ODEProblem(f, u0, (0.0, 100.0))
@time sol = solve(prob, ROCK2());
```

```@example spde
@time sol = solve(prob, ROCK2());
```

if I want to solve it on ``t \in [0,100]``. Done! The solution gives back our tensors (and
interpolates to create new ones if you use sol(t)). We can plot it in Plots.jl:

```@example spde
using Plots
p1 = surface(X, Y, sol[end][:, :, 1], title = "[A]")
p2 = surface(X, Y, sol[end][:, :, 2], title = "[B]")
p3 = surface(X, Y, sol[end][:, :, 3], title = "[C]")
plot(p1, p2, p3, layout = grid(3, 1))
```

and see the pretty gradients. Using this 2nd order ROCK method we solve this equation in
about 2 seconds. That's okay.

## Some Optimizations

There are some optimizations that can still be done. When we do A*B as matrix multiplication,
we create another temporary matrix. These allocations can bog down the system. Instead, we can
pre-allocate the outputs and use the inplace functions mul! to make better use of memory. The
easiest way to store these cache arrays are constant globals, but you can use closures
(anonymous functions which capture data, i.e. (x)->f(x,y)) or call-overloaded types to do it
without globals. The globals way (the easy way) is simply:

```julia
const MyA = zeros(N, N)
const AMx = zeros(N, N)
const DA = zeros(N, N)
function f(du, u, p, t)
    A = @view u[:, :, 1]
    B = @view u[:, :, 2]
    C = @view u[:, :, 3]
    dA = @view du[:, :, 1]
    dB = @view du[:, :, 2]
    dC = @view du[:, :, 3]
    mul!(MyA, My, A)
    mul!(AMx, A, Mx)
    @. DA = D * (MyA + AMx)
    @. dA = DA + α₁ - β₁ * A - r₁ * A * B + r₂ * C
    @. dB = α₂ - β₂ * B - r₁ * A * B + r₂ * C
    @. dC = α₃ - β₃ * C + r₁ * A * B - r₂ * C
end
```

For reference, closures looks like:

```julia
MyA = zeros(N, N)
AMx = zeros(N, N)
DA = zeros(N, N)
function f_full(du, u, p, t, MyA, AMx, DA)
    A = @view u[:, :, 1]
    B = @view u[:, :, 2]
    C = @view u[:, :, 3]
    dA = @view du[:, :, 1]
    dB = @view du[:, :, 2]
    dC = @view du[:, :, 3]
    mul!(MyA, My, A)
    mul!(AMx, A, Mx)
    @. DA = D * (MyA + AMx)
    @. dA = DA + α₁ - β₁ * A - r₁ * A * B + r₂ * C
    @. dB = α₂ - β₂ * B - r₁ * A * B + r₂ * C
    @. dC = α₃ - β₃ * C + r₁ * A * B - r₂ * C
end
f(du, u, p, t) = f_full(du, u, p, t, MyA, AMx, DA)
```

and a call overloaded type looks like:

```julia
struct MyFunction{T} <: Function
    MyA::T
    AMx::T
    DA::T
end

# Now define the overload
function (ff::MyFunction)(du, u, p, t)
    # This is a function which references itself via ff
    A = @view u[:, :, 1]
    B = @view u[:, :, 2]
    C = @view u[:, :, 3]
    dA = @view du[:, :, 1]
    dB = @view du[:, :, 2]
    dC = @view du[:, :, 3]
    mul!(ff.MyA, My, A)
    mul!(ff.AMx, A, Mx)
    @. ff.DA = D * (ff.MyA + ff.AMx)
    @. dA = f.DA + α₁ - β₁ * A - r₁ * A * B + r₂ * C
    @. dB = α₂ - β₂ * B - r₁ * A * B + r₂ * C
    @. dC = α₃ - β₃ * C + r₁ * A * B - r₂ * C
end

MyA = zeros(N, N)
AMx = zeros(N, N)
DA = zeros(N, N)

f = MyFunction(MyA, AMx, DA)
# Now f(du,u,p,t) is our function!
```

These last two ways enclose the pointer to our cache arrays locally but still present a
function f(du,u,p,t) to the ODE solver.

Now, since PDEs are large, many times we don't care about getting the whole timeseries. Using
the [output controls from DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/), we can make it only output the final timepoint.

```julia
prob = ODEProblem(f, u0, (0.0, 100.0))
@time sol = solve(prob, ROCK2(), progress = true, save_everystep = false,
    save_start = false);
@time sol = solve(prob, ROCK2(), progress = true, save_everystep = false,
    save_start = false);
```

Around 0.4 seconds. Much better. Also, if you're using VS Code, this'll give you a nice
progress bar, so you can track how it's going.

## Quick Note About Performance

!!! note
    
    We are using the ROCK2 method here because it's a method for stiff equations with
    eigenvalues that are real-dominated (as opposed to dominated by the imaginary parts).
    If we wanted to use a more conventional implicit ODE solver, we would need to make use
    of the sparsity pattern. This is covered in
    [the advanced ODE tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/)
    It turns out that ROCK2 is more efficient anyway (and doesn't require sparsity
    handling), so we will keep this setup.

### Quick Summary: full PDE ODE Code

As a summary, here's a full PDE code:

```@example
using OrdinaryDiffEq, LinearAlgebra

# Define the constants for the PDE
const α₂ = 1.0
const α₃ = 1.0
const β₁ = 1.0
const β₂ = 1.0
const β₃ = 1.0
const r₁ = 1.0
const r₂ = 1.0
const D = 100.0
const γ₁ = 0.1
const γ₂ = 0.1
const γ₃ = 0.1
const N = 100
const X = reshape([i for i in 1:100 for j in 1:100], N, N)
const Y = reshape([j for i in 1:100 for j in 1:100], N, N)
const α₁ = 1.0 .* (X .>= 80)

const Mx = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
    [1.0 for i in 1:(N - 1)]))
const My = copy(Mx)
Mx[2, 1] = 2.0
Mx[end - 1, end] = 2.0
My[1, 2] = 2.0
My[end, end - 1] = 2.0

# Define the initial condition as normal arrays
u0 = zeros(N, N, 3)

const MyA = zeros(N, N);
const AMx = zeros(N, N);
const DA = zeros(N, N)
# Define the discretized PDE as an ODE function
function f(du, u, p, t)
    A = @view u[:, :, 1]
    B = @view u[:, :, 2]
    C = @view u[:, :, 3]
    dA = @view du[:, :, 1]
    dB = @view du[:, :, 2]
    dC = @view du[:, :, 3]
    mul!(MyA, My, A)
    mul!(AMx, A, Mx)
    @. DA = D * (MyA + AMx)
    @. dA = DA + α₁ - β₁ * A - r₁ * A * B + r₂ * C
    @. dB = α₂ - β₂ * B - r₁ * A * B + r₂ * C
    @. dC = α₃ - β₃ * C + r₁ * A * B - r₂ * C
end

# Solve the ODE
prob = ODEProblem(f, u0, (0.0, 100.0))
sol = solve(prob, ROCK2(), progress = true, save_everystep = false, save_start = false)

using Plots;
gr();
p1 = surface(X, Y, sol[end][:, :, 1], title = "[A]")
p2 = surface(X, Y, sol[end][:, :, 2], title = "[B]")
p3 = surface(X, Y, sol[end][:, :, 3], title = "[C]")
plot(p1, p2, p3, layout = grid(3, 1))
```

## Making Use of GPU Parallelism

That was all using the CPU. How do we turn on GPU parallelism with
DifferentialEquations.jl? Well, you don't. DifferentialEquations.jl "doesn't have GPU bits".
So, wait... can we not do GPU parallelism? No, this is the glory of type-genericness,
especially in broadcasted operations. To make things use the GPU, we simply use a CuArray
from [CUDA.jl](https://cuda.juliagpu.org/stable/). If instead of `zeros(N,M)` we used
`CuArray(zeros(N,M))`, then the array lives on the GPU. CuArray naturally overrides
broadcast such that dotted operations are performed on the GPU. DifferentialEquations.jl
uses broadcast internally, and thus just by putting the array as a `CuArray`, the array-type
will take over how all internal updates are performed and turn this algorithm into a fully
GPU-parallelized algorithm that doesn't require copying to the CPU. Wasn't that simple?

From that you can probably also see how to multithread everything, or how to set everything
up with distributed parallelism. You can make the ODE solvers do whatever you want by
defining an array type where the broadcast does whatever special behavior you want.

So to recap, the entire difference from above is changing to:

```@example spde
using CUDA
const gMx = CuArray(Float32.(Mx))
const gMy = CuArray(Float32.(My))
const gα₁ = CuArray(Float32.(α₁))
gu0 = CuArray(Float32.(u0))

const gMyA = CuArray(zeros(Float32, N, N))
const AgMx = CuArray(zeros(Float32, N, N))
const gDA = CuArray(zeros(Float32, N, N))
function gf(du, u, p, t)
    A = @view u[:, :, 1]
    B = @view u[:, :, 2]
    C = @view u[:, :, 3]
    dA = @view du[:, :, 1]
    dB = @view du[:, :, 2]
    dC = @view du[:, :, 3]
    mul!(gMyA, gMy, A)
    mul!(AgMx, A, gMx)
    @. gDA = D * (gMyA + AgMx)
    @. dA = gDA + gα₁ - β₁ * A - r₁ * A * B + r₂ * C
    @. dB = α₂ - β₂ * B - r₁ * A * B + r₂ * C
    @. dC = α₃ - β₃ * C + r₁ * A * B - r₂ * C
end

prob2 = ODEProblem(gf, gu0, (0.0, 100.0))
CUDA.allowscalar(false) # makes sure none of the slow fallbacks are used
@time sol = solve(prob2, ROCK2(), progress = true, dt = 0.003, save_everystep = false,
    save_start = false);
```

```@example spde
@time sol = solve(prob2, ROCK2(), progress = true, dt = 0.003, save_everystep = false,
    save_start = false);
```

Go have fun.

## And Stochastic PDEs?

Why not make it an SPDE? All that we need to do is extend each of the PDE equations to have
a noise function. In this case, let's use multiplicative noise on each reactant. This means
that our noise update equation is:

```@example spde
function g(du, u, p, t)
    A = @view u[:, :, 1]
    B = @view u[:, :, 2]
    C = @view u[:, :, 3]
    dA = @view du[:, :, 1]
    dB = @view du[:, :, 2]
    dC = @view du[:, :, 3]
    @. dA = γ₁ * A
    @. dB = γ₂ * A
    @. dC = γ₃ * A
end
```

Now we just define and solve the system of SDEs:

```@example spde
prob = SDEProblem(f, g, u0, (0.0, 100.0))
@time sol = solve(prob, SRIW1());
```

```@example spde
using Plots;
gr();

# Use `Array` to transform the result back into a CPU-based `Array` for plotting
p1 = surface(X, Y, Array(sol[end][:, :, 1]), title = "[A]")
p2 = surface(X, Y, Array(sol[end][:, :, 2]), title = "[B]")
p3 = surface(X, Y, Array(sol[end][:, :, 3]), title = "[C]")
plot(p1, p2, p3, layout = grid(3, 1))
```

We can see the cool effect that diffusion dampens the noise in [A] but is unable to dampen
the noise in [B] which results in a very noisy [C]. The stiff SPDE takes much longer to
solve even using high order plus adaptivity because stochastic problems are just that much
more difficult (current research topic is to make new algorithms for this!). It gets GPU'd
just by using CuArray like before. But there we go: solving systems of stochastic PDEs using
high order adaptive algorithms with within-method GPU parallelism. That's gotta be a first?
The cool thing is that nobody ever had to implement the GPU-parallelism either, it just
exists by virtue of the Julia type system.

(Note: We can also use one of the SROCK methods for better performance here, but they will
require a choice of dt. This is left to the reader to try.)

!!! note
    
    This can take a while to solve! An explicit Runge-Kutta algorithm isn't necessarily great
    here, though to use a stiff solver on a problem of this size requires once again smartly
    choosing sparse linear solvers. The high order adaptive method is pretty much necessary
    though, since something like Euler-Maruyama is simply not stable enough to solve this at
    a reasonable dt. Also, the current algorithms are not so great at handling this problem.
    Good thing there's a publication coming along with some new stuff...
