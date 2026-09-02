# The B-spline Galerkin vertical operators carry their own Dirichlet flags (SplineFactor)
# instead of registering them in a global Dict keyed by objectid(F). That registry was read
# without its lock from every thread of the `@threads :static` column loop while the
# first-step per-column assemblies inserted under it, and an objectid can be reused after
# the factorization is freed. The solve itself must be unchanged.
using Test
using LinearAlgebra
using Springsteel
using Springsteel.CubicBSpline
using Scythe

@testset "SplineFactor carries the Dirichlet flags; no global registry" begin
    @test !isdefined(Scythe, :_RIRK_DIRICHLET)
    sp = CubicBSpline.SplineParameters(xmin = 0.0, xmax = 10_000.0, num_cells = 12,
                                       BCL = CubicBSpline.R1T0, BCR = CubicBSpline.R1T0)
    kcol = CubicBSpline.Spline1D(sp)
    d = Scythe._rirk_solve_data(kcol)
    dirichlet = Springsteel.BoundaryConditions(0.0, nothing, nothing, nothing)
    neumann   = Springsteel.BoundaryConditions(nothing, 0.0, nothing, nothing)
    n = sp.bDim
    b = [sin(0.37k) + 0.01k for k in 1:n]
    x = zeros(n)
    for (db, dt) in ((false, false), (true, false), (false, true), (true, true))
        bcB = db ? dirichlet : neumann
        bcT = dt ? dirichlet : neumann
        # Scalar-alpha and profile-alpha assemblies
        for F in (Scythe._assemble_spline_matrix(d, 1.0e5, -1.0, bcB, bcT),
                  Scythe._assemble_spline_matrix(d, fill(1.0e5, sp.mishDim), -1.0, bcB, bcT))
            @test F isa Scythe.SplineFactor
            @test (F.db, F.dt) == (db, dt)
            ldiv!(x, F, b)
            @test x == F.F \ b                     # the wrapper forwards, bit for bit
        end
    end
    # The state-dependent assembly stays a bare factorization (its callers pass the flags).
    Fsd = Scythe._assemble_sd_helmholtz(d, fill(1.0e5, sp.mishDim), true, true)
    @test !(Fsd isa Scythe.SplineFactor)
end
