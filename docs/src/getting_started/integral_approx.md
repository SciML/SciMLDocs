# [Numerically approximate an integral](@id integral_approx)

```@example integral
using Integrals
f(x,p) = sum(sin.(x))
prob = IntegralProblem(f,ones(2),3ones(2))
sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
sol.u
```
