# ── Tests for src/radiation_rrtmgp.jl (the RRTMGP half of the radiation driver) ─
#
# Three tiers, by what they need:
#
#   1. PURE. `standard_extension`, `model_ozone`, `default_gases` and
#      `radiation_divergence!` need no solver at all (the standard atmosphere is
#      analytic and ships in the package). Always run.
#   2. GRAY. `GrayRadiation` uses no lookup tables and downloads no artifacts,
#      so the whole construct-write-solve-store path can be exercised on a
#      machine with no network. Always run.
#   3. SPECTRAL. `:clearsky`/`:allsky` need the RRTMGP artifacts, which download
#      lazily on first use. Every such testset is gated on a probe build and
#      SKIPPED with a clear message if the tables cannot be built, so a compute
#      node with no network reports "skipped", not "failed".
#
# Run standalone with:
#   julia -t 4 --project=. -e 'using Test, Scythe, Springsteel; include("test/test_radiation_rrtmgp.jl")'

using Test
using Scythe
using Springsteel

# `src/radiation_rrtmgp.jl` is included by `src/Scythe.jl` in a normal build.
# Loading it here as well would redefine its `const`s; the guard makes the file
# runnable both before and after that include line lands.
if !isdefined(Scythe, :rrtmgp_solver)
    Scythe.include(joinpath(dirname(pathof(Scythe)), "radiation_rrtmgp.jl"))
end

const SR = Scythe

# ── Artifact probe ────────────────────────────────────────────────────────────
# One attempt, cached: `rrtmgp_lookups` memoizes, so a success here costs the
# gated testsets nothing and a failure costs one try/catch.
const RRTMGP_LOOKUPS_OK = try
    SR.rrtmgp_lookups(:clearsky)
    true
catch err
    @info("test_radiation_rrtmgp: the RRTMGP spectral lookup tables could not " *
          "be built, so every artifact-dependent testset is SKIPPED. Run " *
          "tools/rrtmgp_prewarm.jl on a machine with network access to " *
          "populate the artifact cache.",
          exception = (err, catch_backtrace()))
    false
end

@testset "radiation_rrtmgp" begin

    # ── 1. Pure ───────────────────────────────────────────────────────────────

    @testset "rrtmgp_context is cached and CPU-multithreaded" begin
        ctx = SR.rrtmgp_context()
        @test ctx === SR.rrtmgp_context()
        @test Scythe.ClimaComms.device(ctx) isa Scythe.ClimaComms.CPUMultiThreaded
    end

    @testset "rrtmgp_method mapping" begin
        @test SR.rrtmgp_method(:gray) isa Scythe.RRTMGP.GrayRadiation
        @test SR.rrtmgp_method(:clearsky) isa Scythe.RRTMGP.ClearSkyRadiation
        @test SR.rrtmgp_method(:allsky) isa Scythe.RRTMGP.AllSkyRadiation
        @test SR.rrtmgp_method(:allsky_clear) isa
              Scythe.RRTMGP.AllSkyRadiationWithClearSkyDiagnostics
        @test_throws ErrorException SR.rrtmgp_method(:nonsense)
    end

    @testset "default_gases" begin
        g = SR.default_gases()
        @test g["co2"] ≈ 420.0e-6
        @test g["ch4"] ≈ 1.92e-6
        @test g["n2o"] ≈ 336.0e-9
        @test g["o2"] ≈ 0.209
        @test g["n2"] ≈ 0.781
        @test g["cfc11"] ≈ 233.0e-12
        @test g["cfc12"] ≈ 520.0e-12
        # physical_params overrides, in the units the key names promise
        g2 = SR.default_gases(Dict(:co2_ppm => 280.0, :ch4_ppb => 700.0))
        @test g2["co2"] ≈ 280.0e-6
        @test g2["ch4"] ≈ 700.0e-9
        @test g2["n2o"] ≈ 336.0e-9   # untouched keys keep their defaults
    end

    @testset "standard_extension geometry and values" begin
        z_top = 25.0e3
        n_ext = 15
        ext = SR.standard_extension(:tropical, n_ext, z_top)

        @test ext.nlay == n_ext
        @test length(ext.z) == n_ext
        @test length(ext.dz) == n_ext
        @test length(ext.T) == n_ext
        @test length(ext.p) == n_ext
        @test length(ext.vmr_h2o) == n_ext
        @test length(ext.vmr_o3) == n_ext
        @test length(ext.z_face) == n_ext + 1
        @test length(ext.T_face) == n_ext + 1
        @test length(ext.p_face) == n_ext + 1

        # The seam is exact: face 1 IS the model top, and the extension reaches
        # the fixed 70 km ceiling.
        @test ext.z_face[1] == z_top
        @test ext.z_face[end] == SR.RRTMGP_EXTENSION_TOP

        # Monotone in both z and p, and the faces tile the column exactly.
        @test all(diff(ext.z_face) .> 0)
        @test all(diff(ext.z) .> 0)
        @test all(diff(ext.p_face) .< 0)
        @test all(diff(ext.p) .< 0)
        @test ext.dz ≈ diff(ext.z_face)
        @test sum(ext.dz) ≈ SR.RRTMGP_EXTENSION_TOP - z_top rtol = 1e-14
        @test ext.z ≈ 0.5 .* (ext.z_face[1:(end - 1)] .+ ext.z_face[2:end])

        # Layer values sit between their faces.
        @test all(ext.p_face[2:end] .< ext.p .< ext.p_face[1:(end - 1)])

        # Spacing is uniform in ln p by construction; the inverse interpolation
        # z(ln p) and the forward ln p(z) share the same piecewise-linear
        # segments, so the round trip is exact to floating point.
        dlnp = diff(log.(ext.p_face))
        @test maximum(abs, dlnp .- dlnp[1]) / abs(dlnp[1]) < 1e-9

        # Physical ranges.
        @test all(ext.T .> 150.0)
        @test all(ext.T .< 350.0)
        @test all(ext.p .> 0.0)
        @test all(ext.vmr_h2o .>= 0.0)
        @test all(ext.vmr_o3 .> 0.0)
        # The ozone maximum (~30 km) lies inside the extension, not at its edge.
        @test 1 < argmax(ext.vmr_o3) < n_ext

        # The empty configurations.
        @test SR.standard_extension(:none, 15, z_top) === Scythe.EMPTY_EXTENSION
        @test SR.standard_extension(:tropical, 0, z_top) === Scythe.EMPTY_EXTENSION

        # A model top at or above the extension ceiling has nothing to extend into.
        @test_throws ErrorException SR.standard_extension(:tropical, 15, 80.0e3)
        @test_throws ErrorException SR.standard_extension(:tropical, -1, z_top)

        # Other spacings and climatologies still come out well formed.
        for (kind, n) in ((:tropical, 5), (:midlatitude_summer, 15), (:tropical, 40))
            e = SR.standard_extension(kind, n, 20.0e3)
            @test e.nlay == n
            @test e.z_face[1] == 20.0e3
            @test all(diff(e.z_face) .> 0)
            @test all(diff(e.p_face) .< 0)
        end
    end

    @testset "model_ozone" begin
        z_top = 25.0e3
        nlay = 60
        z = [(k - 0.5) * (z_top / nlay) for k in 1:nlay]

        o3 = SR.model_ozone(:tropical, z)
        @test length(o3) == nlay
        @test all(o3 .> 0.0)
        # The layer peaks near 30 km, ABOVE a 25 km model top, so ozone on the
        # model layers is monotone increasing and its maximum is the top layer.
        @test argmax(o3) == nlay
        @test all(diff(o3) .> 0.0)
        # Tropospheric background is the 30 ppbv floor; the top of the model is
        # already well into the layer.
        @test o3[1] < 1.0e-7
        @test o3[end] > 1.0e-6

        @test SR.model_ozone(:none, z) == zeros(nlay)

        # Continuous across the seam: the last model layer and the extension's
        # first layer are 200-1200 m apart and must not differ by a factor.
        ext = SR.standard_extension(:tropical, 15, z_top)
        @test 0.5 < o3[end] / ext.vmr_o3[1] < 2.0
    end

    @testset "radiation_divergence! telescopes exactly" begin
        rng_state = 20260902
        for (nlay, ncol) in ((30, 1), (12, 7), (100, 3))
            # Deterministic pseudo-random inputs (no RNG dependency).
            F = [sin(0.37 * k + 1.1 * c) * 40.0 + 0.3 * k - 2.0 * c
                 for k in 1:(nlay + 1), c in 1:ncol]
            dz = [50.0 + 40.0 * (1.0 + sin(0.21 * k + rng_state % 7)) for k in 1:nlay]
            q = fill(NaN, ncol * nlay)
            SR.radiation_divergence!(q, F, dz, nlay, ncol, nlay, 1)

            for c in 1:ncol
                # Per-element definition.
                for k in 1:nlay
                    @test q[(c - 1) * nlay + k] ≈ (F[k, c] - F[k + 1, c]) / dz[k]
                end
                # Telescoping: the discrete heating integrates to the column's
                # net flux difference, to round-off, for any flux profile.
                col = @view q[((c - 1) * nlay + 1):(c * nlay)]
                lhs = sum(col .* dz)
                rhs = F[1, c] - F[nlay + 1, c]
                @test isapprox(lhs, rhs; rtol = 1e-12, atol = 1e-10)
            end
        end
    end

    @testset "radiation_divergence! stride 3" begin
        nlay, ncol, stride = 8, 4, 3
        kDim = nlay * stride
        F = [cos(0.29 * k - 0.7 * c) * 25.0 + 1.5 * k for k in 1:(nlay + 1), c in 1:ncol]
        # dz is the STRIDED thickness: the sum of the three sub-layers it covers,
        # which is what makes the strided store exactly conservative.
        dz = [180.0 + 30.0 * sin(0.4 * k) for k in 1:nlay]

        q3 = fill(NaN, ncol * kDim)
        SR.radiation_divergence!(q3, F, dz, nlay, ncol, kDim, stride)

        q1 = fill(NaN, ncol * nlay)
        SR.radiation_divergence!(q1, F, dz, nlay, ncol, nlay, 1)

        for c in 1:ncol, k in 1:nlay
            val = q1[(c - 1) * nlay + k]
            off = (c - 1) * kDim + (k - 1) * stride
            # The whole stride block carries the layer's single value.
            @test q3[off + 1] == val
            @test q3[off + 2] == val
            @test q3[off + 3] == val
        end
        @test !any(isnan, q3)

        # Still telescoping, with the strided weights.
        for c in 1:ncol
            acc = 0.0
            for k in 1:nlay
                acc += q3[(c - 1) * kDim + (k - 1) * stride + 1] * dz[k]
            end
            @test isapprox(acc, F[1, c] - F[nlay + 1, c]; rtol = 1e-12, atol = 1e-10)
        end

        # A stride that does not tile kDim is a configuration error, not a
        # silently truncated column.
        @test_throws ErrorException SR.radiation_divergence!(q3, F, dz, nlay, ncol,
                                                             kDim + 1, stride)
        @test_throws ErrorException SR.radiation_divergence!(q3, F, dz, nlay, ncol,
                                                             kDim, 0)
    end

    # ── 2. Gray: the whole path, no artifacts ─────────────────────────────────

    @testset "gray offline column (artifact-free)" begin
        nlay, n_ext = 40, 10
        r = SR.radiation_offline_column(; method = :gray, solar = :fixed,
                                        nlay = nlay, n_ext = n_ext,
                                        check_values = true)

        @test length(r.z) == nlay
        @test length(r.q_lw) == nlay
        @test size(r.fluxes.lw_net) == (nlay + n_ext + 1, 1)
        @test sum(r.dz) ≈ 25.0e3 rtol = 1e-14

        # A gray atmosphere still emits to space and still cools the LOWER
        # troposphere. Only the lower troposphere: the gray optical depth is an
        # analytic function of p/p_sfc (Frierson 2006 / O'Gorman 2008) that knows
        # nothing about this column's water vapor, and applied to a standard
        # atmosphere it gives LW WARMING above about 12 km, where the real
        # profile's stratospheric inversion is not in gray radiative equilibrium.
        # That is gray radiation being gray, not a sign error -- the spectral
        # testsets below are where the cooling profile is actually checked.
        @test 100.0 < r.olr < 350.0
        @test all(r.q_lw[r.z .< 10.0e3] .< 0.0)

        # Exact energy consistency between the store and the fluxes.
        @test isapprox(sum(r.q_lw .* r.dz),
                       r.fluxes.lw_net[1, 1] - r.fluxes.lw_net[nlay + 1, 1];
                       rtol = 1e-12)

        # Gray radiation is a single band with solar_frac = 1, so the top-of-
        # atmosphere downward shortwave is exactly toa_flux * cos_zenith.
        @test r.fluxes.sw_dn[end, 1] ≈ r.toa_flux * r.cos_zenith rtol = 1e-8
        @test all(isfinite, r.q_sw)
    end

    @testset "gray solver: unwritten inputs fail loudly" begin
        # The NaN placeholders exist so that a driver that forgets to write an
        # input gets an immediate, named error instead of a plausible answer.
        ext = SR.standard_extension(:tropical, 5, 25.0e3)
        z_face = collect(range(0.0, 25.0e3; length = 21))
        was = Scythe.RRTMGP.check_values[]
        try
            sol = SR.rrtmgp_solver(20, ext, 1, :gray, :none, true;
                                   z_face = z_face)
            @test_throws ErrorException Scythe.RRTMGP.update_fluxes!(sol)
        finally
            Scythe.RRTMGP.check_values[] = was
        end
    end

    # ── 3. Spectral: artifact-gated ───────────────────────────────────────────

    if !RRTMGP_LOOKUPS_OK
        @info "SKIPPED: clear-sky offline column (RRTMGP artifacts unavailable)"
        @info "SKIPPED: shortwave boundary condition (RRTMGP artifacts unavailable)"
        @info "SKIPPED: all-sky cloud path (RRTMGP artifacts unavailable)"
    else
        @testset "clear-sky offline column" begin
            nlay, n_ext = 60, 15
            r = SR.radiation_offline_column(; kind = :tropical, nlay = nlay,
                                            n_ext = n_ext, method = :clearsky,
                                            solar = :fixed, check_values = true)

            @info "clear-sky offline column" OLR_W_m2 = r.olr lw_dn_sfc = r.fluxes.lw_dn[1, 1] sw_dn_sfc = r.fluxes.sw_dn[1, 1]

            # Clear-sky tropical OLR. S0's independent 60-layer, 45 km column
            # gave 281.86 W/m^2; a 25 km model plus a 15-layer extension to
            # 70 km must land in the same place.
            @test 270.0 <= r.olr <= 295.0

            # Longwave cools the whole troposphere. Sign convention: q < 0.
            @test all(r.q_lw[r.z .< 15.0e3] .< 0.0)

            # Mid-tropospheric cooling rate, converted from W/m^3 with the
            # column's own density and moist heat capacity.
            k5 = argmin(abs.(r.z .- 5.0e3))
            cool5 = -86400.0 * r.q_lw[k5] / (r.rho[k5] * r.cp[k5])
            @info "LW cooling at 5 km" K_per_day = cool5 z_m = r.z[k5]
            @test 1.0 <= cool5 <= 2.5

            # Shortwave warms, everywhere, in the clear sky.
            @test all(r.q_sw .>= 0.0)

            # Energy: the stored heating integrates to the column's net longwave
            # flux difference exactly.
            @test isapprox(sum(r.q_lw .* r.dz),
                           r.fluxes.lw_net[1, 1] - r.fluxes.lw_net[nlay + 1, 1];
                           rtol = 1e-12)
            @test isapprox(sum(r.q_sw .* r.dz),
                           r.fluxes.sw_net[1, 1] - r.fluxes.sw_net[nlay + 1, 1];
                           rtol = 1e-12)

            # Downward longwave at the surface exceeds the OLR (the greenhouse).
            @test r.fluxes.lw_dn[1, 1] > r.olr
            @test all(isfinite, r.fluxes.lw_up)
            @test all(isfinite, r.fluxes.sw_dn)
        end

        @testset "shortwave boundary condition" begin
            r = SR.radiation_offline_column(; method = :clearsky, solar = :fixed)
            expected = r.toa_flux * r.cos_zenith
            got = r.fluxes.sw_dn[end, 1]
            @info "TOA downward shortwave" got = got expected = expected rel = abs(got - expected) / expected
            # RRTMGP's spectral solar source is renormalized inside the lookup
            # tables, so this is close but not exact (S0 measured 0.4%).
            @test isapprox(got, expected; rtol = 0.02)

            # Some of it reaches the ground, and the surface albedo sends a
            # little back up.
            @test 0.0 < r.fluxes.sw_dn[1, 1] < got
            @test r.fluxes.sw_up[1, 1] > 0.0

            # Night / longwave-only: identically zero, not merely small.
            rn = SR.radiation_offline_column(; method = :clearsky, solar = :none)
            @test all(iszero, rn.fluxes.sw_dn)
            @test all(iszero, rn.fluxes.sw_up)
            @test all(iszero, rn.fluxes.sw_net)
            @test all(iszero, rn.q_sw)
            # Turning the sun off does not perturb the longwave.
            @test rn.olr ≈ r.olr rtol = 1e-12
        end

        @testset "all-sky cloud path" begin
            nlay, n_ext = 40, 10
            z_top = 25.0e3
            dzc = z_top / nlay
            z = [(k - 0.5) * dzc for k in 1:nlay]
            z_face, dz = Scythe.radiation_faces(z, 0.0, z_top)
            ext = SR.standard_extension(:tropical, n_ext, z_top)
            o3 = SR.model_ozone(:tropical, z)

            # Build the column exactly as radiation_offline_column does, so the
            # only difference between the two solves below is the cloud.
            base = SR.radiation_offline_column(; method = :allsky, nlay = nlay,
                                               n_ext = n_ext, solar = :fixed)
            # `base.fluxes` aliases the solver's buffers, which the second solve
            # below overwrites in place (this is the documented contract of
            # `rrtmgp_fluxes`). Copy the two numbers that have to survive it.
            olr_clear = base.olr
            sw_sfc_clear = base.fluxes.sw_dn[1, 1]
            @test 270.0 <= olr_clear <= 295.0

            T_lay = reshape(base.Tk, nlay, 1)
            p_lay = reshape(base.p, nlay, 1)
            T_lev = reshape(base.T_lev, nlay + 1, 1)
            p_lev = reshape(base.p_lev, nlay + 1, 1)
            vmr = reshape(base.vmr_h2o, nlay, 1)

            # A liquid slab between about 1 and 3 km.
            lwp = zeros(nlay, 1); iwp = zeros(nlay, 1)
            re_liq = zeros(nlay, 1); re_ice = zeros(nlay, 1); cf = zeros(nlay, 1)
            for k in 1:nlay
                if 1.0e3 <= z[k] <= 3.0e3
                    lwp[k, 1] = 20.0        # g/m^2 per layer
                    re_liq[k, 1] = 10.0     # micron
                    cf[k, 1] = 1.0
                end
            end
            @test sum(lwp) > 0.0

            SR.rrtmgp_solve_columns!(base.solver, T_lay, p_lay, T_lev, p_lev, vmr,
                                     o3, (lwp, iwp, re_liq, re_ice, cf), ext,
                                     [base.t_sfc], base.cos_zenith, base.toa_flux)
            f = SR.rrtmgp_fluxes(base.solver)
            olr_cloudy = f.lw_up[end, 1]
            @info "all-sky liquid slab" olr_clear = olr_clear olr_cloudy = olr_cloudy

            # A warm low cloud emits from a colder level than the surface, so
            # the OLR must drop; and it reflects, so the surface sees less sun.
            @test olr_cloudy < olr_clear
            @test f.sw_dn[1, 1] < sw_sfc_clear

            # Still exactly conservative with clouds in the column.
            q = fill(NaN, nlay)
            SR.radiation_divergence!(q, f.lw_net, dz, nlay, 1, nlay, 1)
            @test isapprox(sum(q .* dz), f.lw_net[1, 1] - f.lw_net[nlay + 1, 1];
                           rtol = 1e-12)

            # `o3` is accepted as a shared vector or as a per-column matrix; the
            # two must give the same answer when the matrix holds the same
            # profile in every column (the model-facing driver stages ozone into
            # its per-column input batch, so it passes the matrix form).
            o3m = repeat(reshape(o3, nlay, 1), 1, 1)
            SR.rrtmgp_solve_columns!(base.solver, T_lay, p_lay, T_lev, p_lev, vmr,
                                     o3m, (lwp, iwp, re_liq, re_ice, cf), ext,
                                     [base.t_sfc], base.cos_zenith, base.toa_flux)
            @test SR.rrtmgp_fluxes(base.solver).lw_up[end, 1] ≈ olr_cloudy rtol = 1e-14

            # A wrongly-shaped ozone field is a configuration error, not a
            # silently truncated profile.
            @test_throws ErrorException SR.rrtmgp_solve_columns!(
                base.solver, T_lay, p_lay, T_lev, p_lev, vmr, o3[1:(nlay - 1)],
                (lwp, iwp, re_liq, re_ice, cf), ext, [base.t_sfc],
                base.cos_zenith, base.toa_flux)
            @test_throws ErrorException SR.rrtmgp_solve_columns!(
                base.solver, T_lay, p_lay, T_lev, p_lev, vmr, zeros(nlay, 2),
                (lwp, iwp, re_liq, re_ice, cf), ext, [base.t_sfc],
                base.cos_zenith, base.toa_flux)
        end
    end
end
