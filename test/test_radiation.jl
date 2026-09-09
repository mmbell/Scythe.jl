using Test
using Scythe
using Springsteel

# Tests for the RRTMGP-independent half of the radiation driver
# (src/radiation_state.jl): the fixed column geometry (layer faces), the
# level (face) thermodynamic reconstruction, solar geometry, the z_max
# taper, and the option validation table.
#
# Everything here is pure arithmetic on plain arrays -- no RRTMGP, no
# artifacts, no ModelTile -- so these tests run in the base test suite with
# no network and no lookup tables.

@testset "Radiation state (RRTMGP-independent)" begin

    import Springsteel.Thermodynamics: Rd, Rv, gravity

    # ──────────────────────────────────────────────
    # 1. Layer faces on a real RiRk mish
    # ──────────────────────────────────────────────
    @testset "radiation_faces on the RiRk mish" begin
        # 100 cubic-B-spline cells over 0-25 km, exactly as the O01 quick
        # vertical is built (see src/reference_state.jl's mish probe).
        sp = SplineParameters(xmin = 0.0, xmax = 25000.0, num_cells = 100,
                              BCL = CubicBSpline.R0, BCR = CubicBSpline.R0)
        z = Spline1D(sp).mishPoints
        @test length(z) == 300

        z_face, dz = Scythe.radiation_faces(z, 0.0, 25000.0)
        @test length(z_face) == 301
        @test length(dz) == 300
        @test z_face[1] == 0.0
        @test z_face[end] == 25000.0

        # The whole point of the midpoint face rule on this mish: the layer
        # thicknesses tile the column EXACTLY, so the same dz serves both the
        # water paths and the flux divergence.
        @test sum(dz) == 25000.0
        @test all(dz .> 0.0)

        # Three Gauss nodes per cell at 0.1127/0.5/0.8873 dX means the
        # inter-cell midpoint lands on the cell boundary k*dX.
        dX = 25000.0 / 100
        for k in 1:99
            @test z_face[3k + 1] ≈ k * dX atol=1e-9
        end

        # Interior faces are plain midpoints; the ends are the domain bounds.
        @test z_face[2] ≈ 0.5 * (z[1] + z[2])
        @test z_face[300] ≈ 0.5 * (z[299] + z[300])

        # A non-increasing z is a programming error, not a runtime condition.
        @test_throws ErrorException Scythe.radiation_faces([100.0, 50.0, 200.0], 0.0, 300.0)
        @test_throws ErrorException Scythe.radiation_faces([100.0], 0.0, 300.0)
    end

    # ──────────────────────────────────────────────
    # 2. Level (face) reconstruction: isothermal hydrostatic column
    # ──────────────────────────────────────────────
    @testset "radiation_levels! on an isothermal hydrostatic column" begin
        sp = SplineParameters(xmin = 0.0, xmax = 25000.0, num_cells = 100,
                              BCL = CubicBSpline.R0, BCR = CubicBSpline.R0)
        z = Spline1D(sp).mishPoints
        z_face, dz = Scythe.radiation_faces(z, 0.0, 25000.0)
        nlay = length(z)

        T0 = 250.0
        p0 = 101325.0
        H = Rd * T0 / gravity                 # dry scale height (rho_v = 0)

        work = Scythe.RadiationWork(nlay)
        @test length(work.p) == nlay
        @test length(work.p_face) == nlay + 1
        for k in 1:nlay
            work.p[k] = p0 * exp(-z[k] / H)
            work.Tk[k] = T0
            work.rho_d[k] = work.p[k] / (Rd * T0)
            work.rho_v[k] = 0.0
        end

        Scythe.radiation_levels!(work, z, z_face)

        # ln p is linear in z for this column, so the interior linear-in-ln-p
        # interpolation is exact; the ends use the local moist R_m*T scale
        # height, which is exactly H here.
        for k in 1:(nlay + 1)
            p_exact = p0 * exp(-z_face[k] / H)
            @test work.p_face[k] ≈ p_exact rtol=1e-10
        end
        @test work.p_face[1] ≈ p0 rtol=1e-10          # extrapolated to the ground
        @test work.T_face ≈ fill(T0, nlay + 1)        # isothermal stays isothermal
        @test issorted(work.p_face, rev = true)       # monotone, positive
        @test all(work.p_face .> 0.0)
    end

    @testset "radiation_levels! moist end extrapolation" begin
        # A moist bottom layer must use R_m = Rd + q_v*Rv, not Rd: the check is
        # that the ground-face pressure matches the analytic moist hydrostatic
        # value, which is measurably different from the dry one.
        z = [50.0, 150.0, 250.0]
        z_face, dz = Scythe.radiation_faces(z, 0.0, 300.0)
        work = Scythe.RadiationWork(3)
        T0 = 300.0
        work.Tk .= T0
        work.rho_d .= [1.15, 1.14, 1.13]
        work.rho_v .= [0.02 * 1.15, 0.0, 0.0]     # q_v = 0.02 in the bottom layer
        work.p .= work.rho_d .* Rd .* T0

        Scythe.radiation_levels!(work, z, z_face)

        q_v = work.rho_v[1] / work.rho_d[1]
        R_m = Rd + q_v * Rv
        H_moist = R_m * T0 / gravity
        @test work.p_face[1] ≈ work.p[1] * exp((z[1] - 0.0) / H_moist) rtol=1e-12
        # Moist air has the LARGER scale height, so extrapolating downward from the same
        # layer pressure gains LESS than the dry extrapolation would.
        @test work.p_face[1] < work.p[1] * exp((z[1] - 0.0) / (Rd * T0 / gravity))
    end

    # ──────────────────────────────────────────────
    # 3. Solar geometry
    # ──────────────────────────────────────────────
    @testset "solar_geometry" begin
        S0 = 1360.8

        # March equinox (doy 80), local noon at lon 0, on the equator: the sun
        # is overhead to within the declination error of the cosine formula.
        cosz, toa = Scythe.solar_geometry(0.0, 0.0, 80.5, S0)
        @test cosz ≈ 1.0 atol=0.01
        @test toa ≈ S0 * (1.0 + 0.033 * cos(2π * 80.5 / 365.25)) rtol=1e-12
        @test toa > 0.0

        # Polar night: 80 N in late December never sees the sun.
        for hour in 0.0:0.05:0.95
            cz, _ = Scythe.solar_geometry(80.0, 0.0, 355.0 + hour, S0)
            @test cz == 0.0
        end

        # Symmetry about local noon. Taken at a FIXED doy (so the declination is fixed)
        # by moving the longitude either side of the sub-solar meridian: stepping the
        # time instead would also step the declination by ~0.004 deg/day, which shows up
        # at the 1e-7 level and is a property of the sun, not of the hour-angle formula.
        for dt in (0.05, 0.10, 0.20)
            a, _ = Scythe.solar_geometry(20.0, -15.0 * 24.0 * dt, 172.5, S0)
            b, _ = Scythe.solar_geometry(20.0,  15.0 * 24.0 * dt, 172.5, S0)
            @test a ≈ b atol=1e-12
            @test a < 1.0
        end
        # The same symmetry in time holds to the declination drift.
        for dt in (0.05, 0.10, 0.20)
            a, _ = Scythe.solar_geometry(20.0, 0.0, 172.5 - dt, S0)
            b, _ = Scythe.solar_geometry(20.0, 0.0, 172.5 + dt, S0)
            @test a ≈ b atol=1e-5
        end

        # Midnight is dark, noon is not, and cos_z is never negative.
        midnight, _ = Scythe.solar_geometry(20.0, 0.0, 172.0, S0)
        noon, _ = Scythe.solar_geometry(20.0, 0.0, 172.5, S0)
        @test midnight == 0.0
        @test noon > 0.9

        # Longitude shifts local noon: 90 E is noon six hours earlier in UTC.
        # (the two differ only through the 0.25-day declination drift, ~1e-6)
        east, _ = Scythe.solar_geometry(20.0, 90.0, 172.25, S0)
        @test east ≈ noon atol=1e-4
    end

    @testset "solar_state" begin
        opts = Dict{Symbol,Any}(:solar => :none)
        pp = Dict{Symbol,Float64}()
        @test Scythe.solar_state(opts, pp, 0.0) == (0.0, 0.0)

        opts[:solar] = :fixed
        cosz, toa = Scythe.solar_state(opts, pp, 1234.0)
        @test cosz == 0.2588
        @test toa == 551.58
        pp[:cos_zenith] = 0.5
        pp[:sw_toa_flux] = 700.0
        @test Scythe.solar_state(opts, pp, 0.0) == (0.5, 700.0)

        # :diurnal without a latitude is an error, not a silent equator run.
        opts[:solar] = :diurnal
        @test_throws ErrorException Scythe.solar_state(opts, Dict{Symbol,Float64}(), 0.0)

        # doy = start_doy + (start_hour*3600 + t)/86400
        pp2 = Dict{Symbol,Float64}(:latitude => 0.0, :start_doy => 80.0,
                                   :start_hour => 12.0)
        cz1, tf1 = Scythe.solar_state(opts, pp2, 0.0)
        cz2, tf2 = Scythe.solar_geometry(0.0, 0.0, 80.5, 1360.8)
        @test cz1 == cz2
        @test tf1 == tf2
        # Twelve hours later it is local midnight.
        cz3, _ = Scythe.solar_state(opts, pp2, 43200.0)
        @test cz3 == 0.0

        opts[:solar] = :bogus
        @test_throws ErrorException Scythe.solar_state(opts, pp2, 0.0)
    end

    # ──────────────────────────────────────────────
    # 4. z_max taper
    # ──────────────────────────────────────────────
    @testset "radiation_taper" begin
        z = collect(0.0:250.0:25000.0)
        @test Scythe.radiation_taper(z, Inf) == ones(length(z))

        w = Scythe.radiation_taper(z, 20000.0, 2000.0)
        @test length(w) == length(z)
        @test all(w[z .<= 18000.0] .== 1.0)
        @test all(w[z .>= 20000.0] .== 0.0)
        @test all(0.0 .<= w .<= 1.0)
        @test issorted(w, rev = true)                       # monotone non-increasing
        @test Scythe.radiation_taper([19000.0], 20000.0, 2000.0)[1] ≈ 0.5 atol=1e-12
        @test_throws ErrorException Scythe.radiation_taper(z, 20000.0, 0.0)
    end

    # ──────────────────────────────────────────────
    # 5. Option validation (D7 table)
    # ──────────────────────────────────────────────
    @testset "validate_radiation_options" begin
        eqs = "moist_compressible_XZ"
        base() = Dict{Symbol,Any}(:radiation => :rrtmgp)
        nopp() = Dict{Symbol,Float64}()

        # Radiation off: defaults, no complaints, nothing to validate.
        r = Scythe.validate_radiation_options(Dict{Symbol,Any}(), nopp(),
                                              "LinearAdvection1D", 0.3, 300)
        @test r.scheme == :none
        @test r.forcing == :full
        # N2: the sidecar is OPT-IN now (the radiation fields ride in the model's own
        # comprehensive <t>.nc). :radiation_output => true is what turns it back on.
        @test r.output == false

        # Resolved defaults with radiation on.
        r = Scythe.validate_radiation_options(base(), nopp(), eqs, 0.3, 300)
        @test r.scheme == :rrtmgp
        @test r.method == :allsky
        @test r.solar == :none
        @test r.interval_steps == 1000            # 300 s / 0.3 s
        @test r.stride == 1
        @test r.z_max == Inf
        @test r.rain_in_cloud == false
        @test r.extension == :tropical
        @test r.n_ext == 15
        @test r.level_interp == :hydrostatic
        @test r.sw_rescale == false               # default true only for :diurnal
        @test r.check_values == false

        # :diurnal turns the per-step SW rescale on by default.
        o = base(); o[:solar] = :diurnal
        r = Scythe.validate_radiation_options(o, Dict{Symbol,Float64}(:latitude => 20.0),
                                              eqs, 0.3, 300)
        @test r.sw_rescale == true
        @test r.solar == :diurnal

        # Unknown values.
        o = Dict{Symbol,Any}(:radiation => :rrtmg)
        @test_throws ErrorException Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300)
        o = base(); o[:radiation_forcing] = :anomoly
        @test_throws ErrorException Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300)
        o = base(); o[:radiation_method] = :fast
        @test_throws ErrorException Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300)
        o = base(); o[:radiation_extension] = :arctic
        @test_throws ErrorException Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300)

        # Unknown :radiation_* option key (a typo must not be silently ignored).
        o = base(); o[:radiation_intervals] = 300.0
        @test_throws ErrorException Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300)

        # Radiation only exists on the pressure-reference (moist_compressible) sets.
        @test_throws ErrorException Scythe.validate_radiation_options(
            base(), nopp(), "ShallowWaterRL", 0.3, 300)

        # :diurnal without a latitude.
        o = base(); o[:solar] = :diurnal
        @test_throws ErrorException Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300)

        # SST in Celsius.
        @test_throws ErrorException Scythe.validate_radiation_options(
            base(), Dict{Symbol,Float64}(:SST => 28.0), eqs, 0.3, 300)

        # Albedo / emissivity out of range.
        @test_throws ErrorException Scythe.validate_radiation_options(
            base(), Dict{Symbol,Float64}(:sfc_albedo => 1.5), eqs, 0.3, 300)
        @test_throws ErrorException Scythe.validate_radiation_options(
            base(), Dict{Symbol,Float64}(:sfc_emissivity => 0.0), eqs, 0.3, 300)

        # Interval <= 0.
        o = base(); o[:radiation_interval] = 0.0
        @test_throws ErrorException Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300)

        # Stride must divide kDim.
        o = base(); o[:radiation_layer_stride] = 7
        @test_throws ErrorException Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300)
        o = base(); o[:radiation_layer_stride] = 3
        @test Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300).stride == 3

        # Negative z_max is meaningless.
        o = base(); o[:radiation_z_max] = -1.0
        @test_throws ErrorException Scythe.validate_radiation_options(o, nopp(), eqs, 0.3, 300)

        # Warnings (not errors): a sub-10-step interval, and :gray with ice on.
        o = base(); o[:radiation_interval] = 1.0
        @test (@test_logs (:warn,) match_mode=:any Scythe.validate_radiation_options(
            o, nopp(), eqs, 0.3, 300)).interval_steps == 3
        o = base(); o[:radiation_method] = :gray; o[:ice_microphysics] = :ishmael
        @test (@test_logs (:warn,) match_mode=:any Scythe.validate_radiation_options(
            o, nopp(), eqs, 0.3, 300)).method == :gray
    end

    # ──────────────────────────────────────────────
    # 6. Containers
    # ──────────────────────────────────────────────
    @testset "containers" begin
        cld = Scythe.CloudOpticsColumn(12)
        for f in (cld.lwp, cld.iwp, cld.re_liq, cld.re_ice, cld.cf)
            @test length(f) == 12
            @test all(f .== 0.0)
        end

        ext = Scythe.EMPTY_EXTENSION
        @test ext.nlay == 0
        @test isempty(ext.z)
        @test isempty(ext.z_face)
        @test isempty(ext.vmr_o3)

        rad = Scythe.EMPTY_RADIATION
        @test rad.active == false
        @test rad.scheme == :none
        @test isempty(rad.q_lw)
        @test isempty(rad.q_sw)
        @test isempty(rad.q_lw_ref)
        @test rad.solver === nothing
        @test rad.extension.nlay == 0
        @test rad.n_clamp_tk == 0
        @test rad.sw_scale == 1.0
    end
end
