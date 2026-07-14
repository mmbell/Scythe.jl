using Test
using Scythe
using Springsteel
using SparseArrays
using Statistics

# Stochastic forcing of the Twoway_PV_mixing mass sink.
#
# `S1` is the diabatic mass-sink coefficient: the h tendency carries -(Hfree + h)*w_b*S1, removing
# mass where air leaves the boundary layer and acting as a material source of PV. `:S1_sigma`
# perturbs it with a zero-mean Gaussian drawn per gridpoint per timestep, so the forcing carries the
# variable buoyancy of the rising motion rather than tracking w_b alone.
#
# What has to hold:
#   * sigma = 0 (the default) is the deterministic model, bit-for-bit — otherwise every existing run
#     and the bitwise regression benchmark silently change meaning.
#   * the noise is zero-mean with the prescribed standard deviation, and scales with |(Hfree+h)*w_b|
#     since it multiplies the sink rather than being added to it.
#   * a seed reproduces a run exactly (ensembles are then just a seed sweep).
@testset "Twoway_PV_mixing stochastic S1" begin

    Hfree = 2000.0
    Hb = 1000.0
    S1 = 1.0e-5

    function make_model(; S1_sigma = nothing, noise_seed = nothing, num_cells = 20)
        vars = ["h", "u", "v", "ub", "vb", "wb"]
        gp = SpringsteelGridParameters(
            geometry = "RL", iMin = 0.0, iMax = 3.0e5, num_cells = num_cells,
            max_wavenumber = Dict(v => 8 for v in vars),
            BCL = Dict("h"  => Springsteel.CubicBSpline.R1T1,
                       "u"  => Springsteel.CubicBSpline.R1T0,
                       "v"  => Springsteel.CubicBSpline.R1T0,
                       "ub" => Springsteel.CubicBSpline.R1T0,
                       "vb" => Springsteel.CubicBSpline.R1T0,
                       "wb" => Springsteel.CubicBSpline.R1T1),
            BCR = Dict("h"  => Springsteel.CubicBSpline.R0,
                       "u"  => Springsteel.CubicBSpline.R1T1,
                       "v"  => Springsteel.CubicBSpline.R0,
                       "ub" => Springsteel.CubicBSpline.R1T1,
                       "vb" => Springsteel.CubicBSpline.R0,
                       "wb" => Springsteel.CubicBSpline.R0),
            vars = Dict(v => i for (i, v) in enumerate(vars)))

        pp = Dict(:g => 9.81, :Ls_free => 500.0, :Ls_bl => 2000.0,
                  :K_min_free => 0.0, :K_min_bl => 1000.0, :Cd => 2.4e-3,
                  :Hfree => Hfree, :Hb => Hb, :f => 5.0e-5, :S1 => S1)
        S1_sigma === nothing || (pp[:S1_sigma] = S1_sigma)

        opts = Dict{Symbol,Any}(:semiimplicit => false, :exact_reference_state => false)
        noise_seed === nothing || (opts[:noise_seed] = noise_seed)

        return ModelParameters(
            ts = 3.0, integration_time = 3.0, output_interval = 3.0,
            equation_set = "Twoway_PV_mixing", initial_conditions = "",
            grid_params = gp, physical_params = pp, options = opts)
    end

    """A convergent vortex, so w_b = -Hb*div(u_b) is substantially nonzero and the mass sink bites."""
    function make_tile(model)
        patch = createGrid(model.grid_params)
        gridpoints = Scythe.getGridpoints(patch)
        for i in 1:size(patch.physical, 1)
            r_m = gridpoints[i, 1]
            vmax = 30.0 * (r_m / 50.0e3) * exp(1.0 - (r_m / 50.0e3))
            patch.physical[i, 1, 1] = 100.0 * exp(-(r_m / 100.0e3)^2)   # h
            patch.physical[i, 3, 1] = vmax                               # v
            patch.physical[i, 5, 1] = 0.8 * vmax                         # vb
            patch.physical[i, 4, 1] = -0.15 * vmax                       # ub — radial inflow
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        return createModelTile(patch, patch, model, spzeros(1, 1))
    end

    """The h tendency (expdot slot 1) from one call of the equation set."""
    function h_tendency(mtile)
        n = size(mtile.tile.physical, 1)
        Scythe.Twoway_PV_mixing(mtile, 1, n, 2)
        return copy(mtile.expdot_n[:, 1])
    end

    @testset "sigma = 0 is exactly the deterministic model" begin
        # The default (key absent) and an explicit zero must both take the untouched branch, and
        # must agree with each other BIT-FOR-BIT — not merely to a tolerance. If this ever fails,
        # the deterministic runs and the bitwise regression reference have changed meaning.
        absent = h_tendency(make_tile(make_model()))
        zero   = h_tendency(make_tile(make_model(S1_sigma = 0.0, noise_seed = 7)))
        @test absent == zero

        # ...and adding noise must actually change something (guards against a no-op branch).
        noisy = h_tendency(make_tile(make_model(S1_sigma = 0.5e-5, noise_seed = 7)))
        @test noisy != absent
    end

    @testset "a seed reproduces the run; a different seed is a different member" begin
        a1 = h_tendency(make_tile(make_model(S1_sigma = 0.5e-5, noise_seed = 2006)))
        a2 = h_tendency(make_tile(make_model(S1_sigma = 0.5e-5, noise_seed = 2006)))
        b  = h_tendency(make_tile(make_model(S1_sigma = 0.5e-5, noise_seed = 2007)))
        @test a1 == a2          # same seed, bit-for-bit
        @test a1 != b           # different seed, different member
    end

    @testset "successive timesteps draw fresh noise" begin
        # White in TIME: the perturbation must be resampled every call, not fixed at construction.
        mtile = make_tile(make_model(S1_sigma = 0.5e-5, noise_seed = 11))
        @test h_tendency(mtile) != h_tendency(mtile)
    end

    @testset "noise is zero-mean with the prescribed standard deviation" begin
        # The sink is -(Hfree + h)*w*(S1 + eps), so over many draws the h tendency at each point has
        # mean = the deterministic tendency and sd = |(Hfree + h)*w| * sigma. That sd is the real
        # assertion here: it pins BOTH the amplitude and the fact that the noise multiplies w_b
        # (localizing the forcing to convergent BL regions) rather than being added to the tendency.
        sigma = 0.5e-5
        nsamp = 4000

        deterministic = h_tendency(make_tile(make_model()))

        mtile = make_tile(make_model(S1_sigma = sigma, noise_seed = 99))
        n = size(mtile.tile.physical, 1)
        samples = Matrix{Float64}(undef, n, nsamp)
        for k in 1:nsamp
            samples[:, k] = h_tendency(mtile)
        end

        h = mtile.tile.physical[:, 1, 1]
        w = mtile.tile.physical[:, 6, 1]
        expected_sd = abs.((Hfree .+ h) .* w) .* sigma

        # Only judge where the forcing is actually active; where w_b ~ 0 the noise is ~0 by
        # construction and the relative test is meaningless.
        active = expected_sd .> 0.05 * maximum(expected_sd)
        @test count(active) > 20

        got_mean = vec(mean(samples; dims = 2))[active]
        got_sd = vec(std(samples; dims = 2))[active]
        want_sd = expected_sd[active]

        # Mean converges to the deterministic tendency; tolerance is 5 standard errors.
        stderr_mean = want_sd ./ sqrt(nsamp)
        @test all(abs.(got_mean .- deterministic[active]) .< 5 .* stderr_mean)

        # Sample sd of nsamp normal draws has relative error ~1/sqrt(2*nsamp) = 1.1%; allow 6x.
        @test all(isapprox.(got_sd, want_sd; rtol = 0.07))
    end

    @testset "stochastic path is still allocation-free" begin
        # randn! into the preallocated :eps slot — the whole point of doing Part 1 first.
        mtile = make_tile(make_model(S1_sigma = 0.5e-5, noise_seed = 3))
        n = size(mtile.tile.physical, 1)
        Scythe.Twoway_PV_mixing(mtile, 1, n, 2)      # compile
        @test (@allocations Scythe.Twoway_PV_mixing(mtile, 1, n, 2)) == 0
    end
end
