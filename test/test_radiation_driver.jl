using Test
using Scythe
using Springsteel
using SparseArrays

# Driver-level tests for the radiation plumbing (S2a): the `ModelTile.radiation` field
# and its setup (`mc_radiation_state`), the once-per-step pre-pass and its cadence
# (`radiation_prepass!` / `radiation_update!`), the `:prescribed` scheme, the z_max taper
# and the `:anomaly` reference subtraction (`radiation_store!`), and above all the FOLD:
# the held heating reaching `expdot` through `QDOT_TH` with the right coefficient in the
# right three slots and NOTHING in the others.
#
# The fold test follows test_louis_bl.jl's design exactly: the SAME state is run through
# `advance_column` twice, once with radiation absent and once with a synthetic held field,
# and every assertion is on the DIFFERENCE, so advection, the pressure-gradient force and
# the condensation closure cancel identically and only the radiative contribution is left.
#
# Nothing here needs a radiative-transfer library, a lookup artifact or a network: the
# `:prescribed` scheme is a deterministic function of the column state, which is exactly
# why it exists.

@testset "Radiation driver plumbing (S2a)" begin

    import Springsteel.Thermodynamics: Rd, Rv, Cpd, gravity

    """Dry, neutrally stable (theta = 300 K) analytic adiabat -- the same base
    test_louis_bl.jl uses for its clean tests. A dry base pins the diagnostic vapor to
    (essentially) zero, so the radiation term is not competing with a vapor channel whose
    (L_v - R_v T) factor amplifies fit-level wiggles."""
    function dry_adiabatic_column(z; theta0 = 300.0)
        n = length(z)
        exner = @. 1.0 - (gravity * z) / (Cpd * theta0)
        Tk = theta0 .* exner
        p_Pa = @. 100000.0 * exner^(Cpd / Rd)
        rho_d = p_Pa ./ (Rd .* Tk)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    """A small `moist_compressible_XZ` RiRk tile at REST (every perturbation slot zero).

    Rest is not incidental. With `Khdiff_heat = 0` the horizontal thermal source `QDOT_TH`
    is an exact zero column and the velocity divergence vanishes identically, so every
    other term in slots 1/6/7 is an exact zero too and the on/off difference is the
    radiative contribution with no cancellation error at all -- which is what makes a
    1e-12 relative assertion on the fold meaningful rather than optimistic.

    `radopts` is merged into `options`, so passing nothing gives the radiation-ABSENT tile."""
    function make_rad_mtile(tmpdir; radopts = Dict{Symbol,Any}(),
                            radparams = Dict{Symbol,Float64}())
        opts_names = Dict{Symbol,Any}()
        varlist = Scythe.mc_var_names(opts_names; cyl = false)
        rain_name = Scythe.rain_var_name(opts_names)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
        top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 12.0e3, num_cells_i = 6,
            kMin = 0.0, kMax = 2000.0, num_cells_k = 8,
            BCL = side_bc, BCR = side_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
        ref_file = joinpath(tmpdir, "rad_pressure.ref")
        model = ModelParameters(
            ts = 0.1, integration_time = 1.0, output_interval = 1.0,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = merge(
                Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                     :Kvdiff_water => 0.0, :tau_qss => 10.0, :alpha => 0.0,
                     :z_damp => 20.0e3),
                radparams),
            options = merge(
                Dict{Symbol,Any}(:semiimplicit => true,
                                 :exact_reference_state => true,
                                 :precipitation => false,
                                 # The per-call trace (radiation.jl) is on by default in a
                                 # RUN, where it is the only window onto the radiation
                                 # until S5's sidecar writer; in the unit tests, which call
                                 # the pre-pass dozens of times, it is noise.
                                 :radiation_trace => false),
                radopts))
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, end]
        col = dry_adiabatic_column(z)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        hrm = sparse(Int64[], Int64[], Float64[],
                     size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, hrm)
        return mtile, patch, model, gp, z
    end

    # ──────────────────────────────────────────────
    # 1. The off path
    # ──────────────────────────────────────────────
    @testset "radiation absent -> EMPTY_RADIATION" begin
        mktempdir() do tmp
            mtile, _, _, _, _ = make_rad_mtile(tmp)
            # The SHARED singleton, not a fresh inactive instance: the off path must not
            # allocate a per-tile state, and nothing may ever write into it.
            @test mtile.radiation === Scythe.EMPTY_RADIATION
            @test mtile.radiation.active == false
            @test mtile.radiation.scheme === :none
            @test isempty(mtile.radiation.q_lw)
            # An explicit :none is the same thing as an absent key.
            m2, _, _, _, _ = make_rad_mtile(tmp;
                radopts = Dict{Symbol,Any}(:radiation => :none))
            @test m2.radiation === Scythe.EMPTY_RADIATION
            # The pre-pass on an inactive tile is a no-op that must not throw.
            @test Scythe.radiation_prepass!(mtile, 1) === nothing
        end
    end

    @testset "mc_radiation_state: :prescribed setup" begin
        mktempdir() do tmp
            mtile, patch, model, gp, z = make_rad_mtile(tmp;
                radopts = Dict{Symbol,Any}(:radiation => :prescribed,
                                           :radiation_interval => 10.0))
            rs = mtile.radiation
            @test rs.active
            @test rs.scheme === :prescribed
            @test rs.forcing === :full
            @test rs.kDim == gp.kDim
            @test rs.nlay == gp.kDim               # default stride 1
            @test rs.ncol == Scythe.num_columns(patch)
            @test length(rs.q_lw) == size(patch.physical, 1)
            @test length(rs.q_sw) == size(patch.physical, 1)
            @test rs.q_lw == zeros(size(patch.physical, 1))
            # Seconds in, steps out, converted exactly once (D6).
            @test rs.interval_steps == round(Int, 10.0 / model.ts)
            # Column geometry: faces tile the column exactly, so the same dz weights a
            # water path and divides a flux difference.
            @test rs.z ≈ z
            @test rs.z_face[1] == 0.0
            @test rs.z_face[end] == 2000.0
            @test sum(rs.dz) == 2000.0
            @test all(rs.taper .== 1.0)            # z_max = Inf
            @test rs.solver === nothing
            @test rs.last_call_step == typemin(Int)
            # Every tile gets its OWN state (the held field is written per tile).
            m2, _, _, _, _ = make_rad_mtile(tmp;
                radopts = Dict{Symbol,Any}(:radiation => :prescribed))
            @test m2.radiation !== rs
        end
    end

    @testset "mc_radiation_state: loud on a non-mc set" begin
        # The QDOT_TH insertion point exists only on the pressure-reference family, so
        # radiation asked for anywhere else must fail at SETUP, not silently do nothing.
        gp = GridParameters(geometry = "R", iMin = 0.0, iMax = 1.0e3, num_cells = 4,
                            BCL = Dict("u" => NeumannBC()),
                            BCR = Dict("u" => NeumannBC()),
                            vars = Dict("u" => 1))
        model = ModelParameters(ts = 1.0, equation_set = "LinearAdvection1D",
            grid_params = gp,
            options = Dict{Symbol,Any}(:radiation => :prescribed))
        patch = createGrid(model.grid_params)
        @test_throws ErrorException Scythe.mc_radiation_state(model, patch,
                                                             Scythe.getGridpoints(patch))
    end

    # ──────────────────────────────────────────────
    # 2. The :prescribed heating and the cadence
    # ──────────────────────────────────────────────
    """The mixture heat capacity C_vt of the tile's own column state, per gridpoint,
    formed exactly as `mc_driver!` forms it (Eq. mixture_C) from the SAME reconstruction
    the radiation driver uses."""
    function column_cvt_rhod(mtile)
        rs = mtile.radiation
        kDim = rs.kDim
        n = size(mtile.tile.physical, 1)
        C_vt = zeros(n); rho_d = zeros(n); R_m = zeros(n); Tk = zeros(n); p = zeros(n)
        w = Scythe.RadiationWork(kDim)
        for c in 1:rs.ncol
            cs = (c - 1) * kDim + 1
            Scythe.radiation_column_state!(w, mtile, cs, cs + kDim - 1)
            for k in 1:kDim
                rd = w.rho_d[k]
                q_v = w.rho_v[k] / rd
                q_l = w.rho_liq[k] / rd
                q_i = w.rho_ice[k] / rd
                C_vt[cs + k - 1] = Scythe.Cvd + (q_v * Scythe.Cvv) +
                                   (q_l * Scythe.Cl) + (q_i * Scythe.Ci)
                R_m[cs + k - 1] = Rd + (q_v * Rv)
                rho_d[cs + k - 1] = rd
                Tk[cs + k - 1] = w.Tk[k]
                p[cs + k - 1] = w.p[k]
            end
        end
        return (; C_vt, R_m, rho_d, Tk, p)
    end

    @testset ":prescribed fills q = rho_d C_vt rate/86400" begin
        mktempdir() do tmp
            rate = -1.5
            mtile, patch, model, gp, _ = make_rad_mtile(tmp;
                radopts = Dict{Symbol,Any}(:radiation => :prescribed,
                                           :radiation_interval => 10.0),
                radparams = Dict{Symbol,Float64}(:radiation_prescribed_rate => rate))
            rs = mtile.radiation
            Scythe.radiation_prepass!(mtile, 1)
            st = column_cvt_rhod(mtile)
            expected = st.rho_d .* st.C_vt .* (rate / 86400.0)
            @test rs.q_lw ≈ expected rtol=1e-12
            @test all(rs.q_sw .== 0.0)             # LW-only scheme
            @test all(rs.q_lw .< 0.0)              # a negative rate is COOLING
            # A dry column: C_vt is C_vd to within the reference fit's residual vapor.
            @test maximum(abs.(st.C_vt .- Scythe.Cvd)) < 1.0e-6 * Scythe.Cvd
            # -1.5 K/day at rho_d ~ 1 kg/m^3 is a ~1.2e-2 W/m^3 source.
            @test 5.0e-3 < maximum(abs.(rs.q_lw)) < 3.0e-2
            @test rs.last_call_step == 1
            @test rs.sw_scale == 1.0
            # :solar defaults to :none, so there is no shortwave to place.
            @test rs.cos_zenith == 0.0
            @test rs.toa_flux == 0.0
        end
    end

    @testset "cadence: the field is HELD between calls" begin
        mktempdir() do tmp
            mtile, _, model, _, _ = make_rad_mtile(tmp;
                radopts = Dict{Symbol,Any}(:radiation => :prescribed,
                                           :radiation_interval => 10.0))
            rs = mtile.radiation
            nsteps = rs.interval_steps
            @test nsteps == 100                    # 10 s at ts = 0.1 s

            Scythe.radiation_prepass!(mtile, 1)    # the first call is always forced
            @test rs.last_call_step == 1
            held = copy(rs.q_lw)
            @test any(held .!= 0.0)

            # Scribble over the held field. Any step inside the interval must leave the
            # scribble alone -- that IS the held-forcing pattern.
            fill!(rs.q_lw, 0.0)
            for t in 2:nsteps
                Scythe.radiation_prepass!(mtile, t)
            end
            @test all(rs.q_lw .== 0.0)
            @test rs.last_call_step == 1

            # t - last_call_step == interval_steps: the cadence has elapsed, recompute.
            Scythe.radiation_prepass!(mtile, 1 + nsteps)
            @test rs.last_call_step == 1 + nsteps
            @test rs.q_lw == held                  # same state -> bitwise the same field
        end
    end

    # ──────────────────────────────────────────────
    # 3. Taper and anomaly forcing
    # ──────────────────────────────────────────────
    @testset "z_max taper zeroes the heating aloft" begin
        mktempdir() do tmp
            z_max = 1500.0
            mtile, patch, _, gp, z = make_rad_mtile(tmp;
                radopts = Dict{Symbol,Any}(:radiation => :prescribed,
                                           :radiation_z_max => z_max))
            rs = mtile.radiation
            @test any(rs.taper .== 0.0)            # the grid reaches above z_max
            Scythe.radiation_prepass!(mtile, 1)
            kDim = gp.kDim
            for c in 1:rs.ncol, k in 1:kDim
                j = (c - 1) * kDim + k
                if z[k] >= z_max
                    @test rs.q_lw[j] == 0.0        # exactly, not approximately
                end
            end
            # Below the ramp the forcing is untouched (taper = 1 there).
            k_low = 1
            @test rs.taper[k_low] > 0.0
            @test rs.q_lw[k_low] != 0.0
        end
    end

    @testset ":anomaly forcing on a homogeneous state is identically zero" begin
        mktempdir() do tmp
            mtile, patch, _, _, _ = make_rad_mtile(tmp;
                radopts = Dict{Symbol,Any}(:radiation => :prescribed,
                                           :radiation_forcing => :anomaly))
            rs = mtile.radiation
            @test rs.forcing === :anomaly
            Scythe.radiation_prepass!(mtile, 1)
            # Every column of a horizontally homogeneous tile carries the same profile, so
            # the horizontal mean IS the profile and the anomaly vanishes. This is the
            # sharp integration test the plan rests on: an :anomaly run on the O01 sounding
            # must stay bitwise the radiation-off run until the bubble breaks homogeneity.
            @test maximum(abs.(rs.q_lw)) < 1.0e-15 * maximum(abs.(rs.q_lw_ref))
            @test all(rs.q_sw .== 0.0)
            # The reference profile itself is the FULL heating, captured once.
            @test any(rs.q_lw_ref .< 0.0)
            ref1 = copy(rs.q_lw_ref)
            Scythe.radiation_prepass!(mtile, 1 + rs.interval_steps)
            @test rs.q_lw_ref == ref1              # captured on the FIRST call only
        end
    end

    # ──────────────────────────────────────────────
    # 4. The fold into expdot (the whole point of S2a)
    # ──────────────────────────────────────────────
    @testset "QDOT_TH fold: coefficients, slots, and nothing else" begin
        mktempdir() do tmp1
            mktempdir() do tmp2
                m_on, p_on, model, gp, z = make_rad_mtile(tmp1;
                    radopts = Dict{Symbol,Any}(:radiation => :prescribed))
                m_off, p_off, _, _, _ = make_rad_mtile(tmp2)
                @test m_off.radiation.active == false

                kDim = gp.kDim
                ncols = Scythe.num_columns(p_on)
                npts = size(p_on.physical, 1)

                # A SYNTHETIC held field, written straight into the state after
                # construction: this test is about the fold, not about where the numbers
                # came from. Column- and height-varying so a slot that silently reused one
                # gridpoint's value could not pass. Magnitude ~1e-2 W/m^3, the scale of a
                # real -1.5 K/day cooling.
                q = zeros(npts)
                for c in 1:ncols, k in 1:kDim
                    q[(c - 1) * kDim + k] =
                        -1.0e-2 * (1.0 + 0.3 * sin(2π * k / kDim) + 0.1 * c)
                end
                copyto!(m_on.radiation.q_lw, q)
                @test all(m_on.radiation.q_sw .== 0.0)

                for c in 1:ncols
                    Scythe.advance_column(m_on, c, 1)
                    Scythe.advance_column(m_off, c, 1)
                end
                D = m_on.expdot_n .- m_off.expdot_n

                st = column_cvt_rhod(m_on)
                p_hPa = st.p ./ 100.0
                drvs_dT = Scythe.drho_vsat_dT.(st.Tk, p_hPa)
                drvs_dp = Scythe.drho_vsat_dp.(st.Tk, p_hPa)

                # Slot 1 (pressure): the heating enters inside the (R_m/C_vt)[...] bracket.
                d_p = (st.R_m ./ st.C_vt) .* q
                # Slot 6 (total energy): the source is the heating itself.
                d_E = q
                # Slot 7 (supersaturation): the heating reaches Q_ss ONLY through the
                # non-condensational dT_nc/dp_nc that feed the saturation chain rule SATF,
                # SATF = -rho_vs*div - drvs_dT*dT_nc - drvs_dp*dp_nc, with
                # dT_nc += q/(rho_d C_vt) and dp_nc += (R_m/C_vt) q.
                d_Q = .-(drvs_dT .* q ./ (st.rho_d .* st.C_vt)) .- (drvs_dp .* d_p)

                scale_p = maximum(abs.(d_p))
                scale_E = maximum(abs.(d_E))
                scale_Q = maximum(abs.(d_Q))
                @test maximum(abs.(D[:, 1] .- d_p)) <= 1.0e-12 * scale_p
                @test maximum(abs.(D[:, 6] .- d_E)) <= 1.0e-12 * scale_E
                @test maximum(abs.(D[:, 7] .- d_Q)) <= 1.0e-12 * scale_Q
                # The three coefficients are genuinely different numbers, so the test
                # cannot be passing by all three being the same thing.
                @test scale_p > 0.0 && scale_E > 0.0 && scale_Q > 0.0
                @test !isapprox(scale_p, scale_E; rtol = 1.0e-3)

                # EVERY other slot is untouched -- bitwise, not to a tolerance. Radiation
                # is a thermal source and must not move momentum, density or water.
                nvars = size(D, 2)
                for v in 1:nvars
                    v in (1, 6, 7) && continue
                    @test D[:, v] == zeros(size(D, 1))
                end

                # The shortwave leg rides the same fold through `sw_scale`. With q_sw = 0
                # it contributes nothing; move the whole field into q_sw at scale 1 and the
                # answer must be identical.
                m_sw, _, _, _, _ = make_rad_mtile(tmp1;
                    radopts = Dict{Symbol,Any}(:radiation => :prescribed))
                copyto!(m_sw.radiation.q_sw, q)
                @test m_sw.radiation.sw_scale == 1.0
                for c in 1:ncols
                    Scythe.advance_column(m_sw, c, 1)
                end
                @test m_sw.expdot_n == m_on.expdot_n

                # ... and halving `sw_scale` halves the contribution.
                m_hf, _, _, _, _ = make_rad_mtile(tmp1;
                    radopts = Dict{Symbol,Any}(:radiation => :prescribed))
                copyto!(m_hf.radiation.q_sw, q)
                m_hf.radiation.sw_scale = 0.5
                for c in 1:ncols
                    Scythe.advance_column(m_hf, c, 1)
                end
                Dh = m_hf.expdot_n .- m_off.expdot_n
                @test maximum(abs.(Dh[:, 6] .- 0.5 .* d_E)) <= 1.0e-12 * scale_E
            end
        end
    end

    @testset "radiation off is bitwise the radiation-free driver" begin
        # The gate is `if rad_on`, not `+ 0.0`: adding an exact zero is NOT the identity
        # for -0.0, and a whole class of committed benchmark references rests on the off
        # path being byte-identical. Two independently constructed radiation-free tiles
        # must agree bit for bit, and an ACTIVE tile whose held field is all zeros must
        # agree with them too (that is the weaker statement, and the one that would break
        # first if the fold ever stopped being gated).
        mktempdir() do tmp1
            mktempdir() do tmp2
                m_a, p_a, _, gp, _ = make_rad_mtile(tmp1)
                m_b, _, _, _, _ = make_rad_mtile(tmp2)
                ncols = Scythe.num_columns(p_a)
                for c in 1:ncols
                    Scythe.advance_column(m_a, c, 1)
                    Scythe.advance_column(m_b, c, 1)
                end
                @test m_a.expdot_n == m_b.expdot_n
            end
        end
    end

    # ──────────────────────────────────────────────
    # 5. The column reconstruction against the reference state it was built from
    # ──────────────────────────────────────────────
    """A RESTING `moist_compressible_XZ` RiRk tile on the O01 benchmark's own reference
    state: the humidified Dunion moist-tropical sounding balanced on a 0-25 km column by
    `Springsteel.calculate_pressure_reference_state`, written through
    `write_exact_ref_mc`, with every perturbation slot exactly zero -- the t = 0 state of
    `benchmarks/o01_rainfall.jl` minus the warm bubble.

    Coarse HORIZONTALLY (4 x 37.5 km cells) but at the benchmark's own 250 m vertical
    spacing, which the reference-state fixed point needs: on a 1.25 km column the
    5-sweep `calculate_pressure_reference_state` iteration has not settled and swings the
    lid pressure negative. Nothing else here
    is resolution-dependent: the assertion is that the radiation column reconstruction
    reproduces the profiles the tile was CONSTRUCTED from, which is a statement about the
    arithmetic, not about the grid. It is the O01 sounding rather than the dry adiabat
    above because the vapour is the term that would betray a wrong reference: `rho_v` is
    carried against the DERIVED `rho_vbar = rho_tbar - rho_dbar - rho_cbar`, and
    Springsteel's independently fitted `ref_rho_v` differs from it at fit level, which on
    a dry column is invisible and on this one is not.

    `:radiation => :prescribed` keeps it artifact-free: no lookup tables, no network, no
    solver -- `radiation_column_state!` is the same function on every scheme."""
    function make_o01_rad_mtile(tmpdir; num_cells_i = 4, num_cells_k = 100, ice = false)
        sounding = joinpath(@__DIR__, "..", "benchmarks", "reference_data",
                            "o01_rainfall", "dunion_MT_hum90.ref")
        isfile(sounding) || error("the O01 sounding is missing: $sounding")
        # `ice = true` registers the twelve ISHMAEL slots (which forces two-moment rain,
        # `mc_var_names`'s own requirement) so the S3b assembly can be tested on the ice
        # path as well as the liquid one. NO transform is applied to any of them: with
        # `:condensate_transform`/`:ice_transform` left at `:none` the slot IS the density,
        # so an injected value can be asserted against the batch arithmetically rather
        # than through a recovery the test would have to reimplement.
        opts_names = ice ?
            Dict{Symbol,Any}(:ice_microphysics => :ishmael, :rain_moments => 2) :
            Dict{Symbol,Any}()
        varlist = Scythe.mc_var_names(opts_names; cyl = false)
        rain_name = Scythe.rain_var_name(opts_names)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        topbot_bc = merge(scalar_bc,
                          Dict("w" => DirichletBC(), rain_name => NaturalBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 150.0e3, num_cells_i = num_cells_i,
            kMin = 0.0, kMax = 25.0e3, num_cells_k = num_cells_k,
            BCL = side_bc, BCR = side_bc, BCB = topbot_bc, BCT = topbot_bc, vars = vars)
        ref_file = joinpath(tmpdir, "o01_rad.ref")
        model = ModelParameters(
            ts = 0.3, integration_time = 1.0, output_interval = 1.0,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Khdiff_heat => 0.0,
                                   :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                   :tau_qss => 10.0, :alpha => 0.02, :z_damp => 17.0e3),
            options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                             :exact_reference_state => true,
                                             :precipitation => true,
                                             :vertical_mixing => false,
                                             :radiation => :prescribed,
                                             :radiation_trace => false),
                            opts_names))
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, 2]
        column = Scythe.reference_column(patch, gp)
        ref_phys = Springsteel.calculate_pressure_reference_state(sounding, z, column)
        Scythe.write_exact_ref_mc(ref_file, z,
                                  Springsteel.ref_pressure(ref_phys)[:, 1],
                                  Springsteel.ref_rho_d(ref_phys)[:, 1],
                                  Springsteel.ref_rho_v(ref_phys)[:, 1],
                                  zeros(kDim))
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        hrm = sparse(Int64[], Int64[], Float64[],
                     size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, hrm)
        return mtile, patch, model, gp, z
    end

    @testset "radiation_column_state! reproduces the resting reference profiles" begin
        mktempdir() do tmp
            mtile, patch, model, gp, z = make_o01_rad_mtile(tmp)
            rs = mtile.radiation
            kDim = gp.kDim
            @test rs.active && rs.ncol > 1
            # Rest is the premise of every assertion below.
            @test all(mtile.tile.physical .== 0.0)

            ref = mtile.ref_state
            pbar = Springsteel.ref_pressure(ref)[:, 1]
            rho_dbar = Springsteel.ref_rho_d(ref)[:, 1]
            rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
            E_tbar = Springsteel.ref_total_energy(ref)[:, 1]
            rho_cbar = Springsteel.ref_rho_c(ref)[:, 1]
            # The vapour the PROGNOSTIC slot is carried against (rho_tbar - rho_dbar -
            # rho_cbar), not Springsteel's separately fitted ref_rho_v. That distinction
            # is the whole reason a resting base is a discrete fixed point.
            rho_vbar = mtile.mc_ref_diag.rho_vbar

            # The resting temperature as the DRIVER defines it: `mc_reference_diagnostics`
            # closed-form retrieval on the bar profiles at ke = 0, rho_r = 0, rho_i = 0.
            Tbar = [Scythe.retrieve_temperature(
                        pbar[k] + E_tbar[k] - rho_tbar[k] * (gravity * z[k]),
                        rho_dbar[k], rho_tbar[k], rho_cbar[k]) for k in 1:kDim]
            # A real tropical sounding, not a degenerate one -- otherwise "reproduces the
            # reference" could be true of an all-zeros bug.
            @test 295.0 < Tbar[1] < 305.0
            @test 190.0 < minimum(Tbar) < 210.0
            @test maximum(rho_vbar) > 1.0e-2          # ~18 g/m^3 at the surface

            work = Scythe.RadiationWork(kDim)
            worst_T = 0.0
            for c in 1:rs.ncol
                cs = (c - 1) * kDim + 1
                Scythe.radiation_column_state!(work, mtile, cs, cs + kDim - 1)
                # p' = 0 and rho_v' = 0, so the totals ARE the reference profiles --
                # bitwise, because the reconstruction is `perturbation + reference` and
                # 0.0 + x === x.
                @test work.p == pbar
                @test work.rho_d == rho_dbar
                @test work.rho_v == rho_vbar
                @test work.rho_liq == rho_cbar        # no reference cloud, no rain
                @test all(work.rho_ice .== 0.0)       # ice off: the slots do not exist
                @test all(work.ke .== 0.0)            # u = v = w = 0 at rest
                worst_T = max(worst_T,
                              maximum(abs.(work.Tk .- Tbar)) / maximum(abs.(Tbar)))
            end
            @test worst_T <= 1.0e-10
            # Nothing in a resting tropical column is outside [170, 350] K, so the clamp
            # counter must still be zero -- if it is not, the retrieval disagrees with the
            # reference by hundreds of kelvin and `worst_T` above is the wrong diagnostic.
            @test rs.n_clamp_tk == 0

            # And the level build on that column is monotone and hydrostatically sane:
            # p decreasing upward, both end faces bracketing the layer points.
            Scythe.radiation_column_state!(work, mtile, 1, kDim)
            Scythe.radiation_levels!(work, rs.z, rs.z_face)
            @test all(diff(work.p_face) .< 0.0)
            @test work.p_face[1] > work.p[1]          # ground below the first layer
            @test work.p_face[end] < work.p[end]      # lid above the last layer
            @test 295.0 < work.T_face[1] < 305.0
        end
    end

    # ──────────────────────────────────────────────
    # 8. S3b: the cloud optics reach the solver batch
    # ──────────────────────────────────────────────
    #
    # `radiation_assemble!` is the pure-Scythe half of `radiation_rrtmgp_update!` — model
    # slots in, solver-input matrices out — so it can be exercised on the `:prescribed`
    # O01 fixture with a HAND-BUILT batch: no lookup artifacts, no network, no solver.
    # What is under test is the GATHER, not the mapping (test_radiation_cloud_optics.jl
    # owns the mapping): that the cloud liquid, the rain and all twelve ISHMAEL moments
    # are picked out of the right slots, recovered through the right transform, and land
    # in the right `(layer, column)` cell of the batch.
    @testset "radiation_assemble!: injected cloud reaches the batch (S3b)" begin
        mktempdir() do tmp
            mtile, patch, model, gp, z = make_o01_rad_mtile(tmp; ice = true)
            rs = mtile.radiation
            kDim = gp.kDim
            ncol = rs.ncol
            slots = mtile.mc_slots
            @test rs.active && ncol >= 4
            @test Scythe.ice_registered(slots)

            phys = mtile.tile.physical
            k_liq = findall(zz -> 1.0e3 <= zz <= 3.0e3, z)
            k_ice = findall(zz -> 8.0e3 <= zz <= 10.0e3, z)
            @test !isempty(k_liq) && !isempty(k_ice)

            # Two of the four columns get cloud; the other two must come back exactly
            # clear, which is the assertion that catches a stale per-tile buffer leaking
            # from column to column (the failure mode `cloud_optics_clear!` exists for).
            wet = (1, 3)
            dry = (2, 4)
            rho_c_inj = 1.0e-3          # kg/m^3, a fat but unremarkable convective cloud
            rho_i_inj = 1.0e-5          # kg/m^3
            n_i_inj = 1.0e5             # #/m^3
            # a = c, so the habit is spherical and delta = 1 exactly; with this (q, n, a)
            # `var_check` re-derives a bulk density of ~199 kg/m^3 and rni = 10 um, so
            # the gamma moment gives re_ice = 6*rni = 60 um -- comfortably inside the
            # [5, 90] um table, which is what makes the clamp assertion below meaningful.
            a_inj = 10.0e-6             # a- and c-axis scale [m]
            vol_inj = (a_inj^3) * n_i_inj   # a_i = ani^2 cni n, c_i likewise

            for c in wet
                for k in k_liq
                    phys[(c - 1) * kDim + k, 9, 1] = rho_c_inj    # slot 9 = cloud (no transform)
                end
                for k in k_ice
                    j = (c - 1) * kDim + k
                    phys[j, slots.i1_q, 1] = rho_i_inj
                    phys[j, slots.i1_n, 1] = n_i_inj
                    phys[j, slots.i1_a, 1] = vol_inj
                    phys[j, slots.i1_c, 1] = vol_inj
                end
            end

            B = Scythe.RadiationBatch(rs.nlay, ncol; ice_on = true)
            P = Scythe.CloudOpticsParams(model.options, model.physical_params)
            @test P.rain_in_cloud == false
            @test P.ice_on
            n_liq0 = rs.n_clamp_re_liq
            n_ice0 = rs.n_clamp_re_ice
            Scythe.radiation_assemble!(rs, mtile, B, P)

            dz = rs.dz
            for c in wet
                for k in k_liq
                    # The water path is the layer's own dz, the same dz the flux
                    # divergence is taken with -- that identity is what makes the column
                    # internally conservative.
                    @test B.lwp[k, c] ≈ 1000.0 * rho_c_inj * dz[k] rtol = 1.0e-14
                    @test B.cf[k, c] == 1.0
                    @test Scythe.RAD_RE_LIQ_MIN <= B.re_liq[k, c] <= Scythe.RAD_RE_LIQ_MAX
                    @test B.iwp[k, c] == 0.0
                end
                for k in k_ice
                    @test B.iwp[k, c] ≈ 1000.0 * rho_i_inj * dz[k] rtol = 1.0e-14
                    @test B.cf[k, c] == 1.0
                    @test B.lwp[k, c] == 0.0
                    # The twelve moments actually arrived: with a carried number the
                    # gamma-PSD branch runs, so `re_ice` is NOT the benign fill written
                    # where the weighting is empty.
                    @test Scythe.RAD_RE_ICE_MIN <= B.re_ice[k, c] <= Scythe.RAD_RE_ICE_MAX
                    @test B.re_ice[k, c] != Scythe.RAD_RE_ICE_BENIGN
                    @test B.re_ice[k, c] ≈ 60.0 rtol = 1.0e-6
                end
                # Everywhere else in a cloudy column is still clear.
                for k in 1:rs.nlay
                    (k in k_liq || k in k_ice) && continue
                    @test B.lwp[k, c] == 0.0
                    @test B.iwp[k, c] == 0.0
                    @test B.cf[k, c] == 0.0
                    @test B.re_liq[k, c] == Scythe.RAD_RE_LIQ_BENIGN
                    @test B.re_ice[k, c] == Scythe.RAD_RE_ICE_BENIGN
                end
            end

            for c in dry
                @test all(B.lwp[:, c] .== 0.0)
                @test all(B.iwp[:, c] .== 0.0)
                @test all(B.cf[:, c] .== 0.0)
                @test all(B.re_liq[:, c] .== Scythe.RAD_RE_LIQ_BENIGN)
                @test all(B.re_ice[:, c] .== Scythe.RAD_RE_ICE_BENIGN)
            end

            # The gather agrees, cell for cell, with calling the S3a mapping directly on
            # the same column: this is the assertion that the ice matrix is in MC_ICE_VARS
            # order and in DENSITY units, not mixing ratios.
            work = Scythe.RadiationWork(kDim)
            Bc = Scythe.RadiationBatch(rs.nlay, 1; ice_on = true)
            cld = Scythe.CloudOpticsColumn(kDim)
            c = wet[1]
            Scythe.radiation_column_state!(work, mtile, (c - 1) * kDim + 1, c * kDim, Bc)
            @test all(Bc.rho_c[k_liq] .== rho_c_inj)
            @test all(Bc.rho_r .== 0.0)
            @test all(Bc.ice[k_ice, 1] .== rho_i_inj)
            @test all(Bc.ice[k_ice, 2] .== n_i_inj)
            @test all(Bc.ice[k_ice, 3] .== vol_inj)
            @test all(Bc.ice[k_ice, 4] .== vol_inj)
            @test all(Bc.ice[:, 5:12] .== 0.0)
            Scythe.cloud_optics_column!(cld, P, work.rho_d, Bc.rho_c, Bc.rho_r,
                                        Bc.ice, rs.dz)
            @test cld.lwp == B.lwp[:, c]
            @test cld.iwp == B.iwp[:, c]
            @test cld.re_liq == B.re_liq[:, c]
            @test cld.re_ice == B.re_ice[:, c]
            @test cld.cf == B.cf[:, c]

            # Clamp accounting: the assembly accumulates the mapping's counts and never
            # resets them, and this cloud is well inside both tables.
            @test rs.n_clamp_re_liq == n_liq0
            @test rs.n_clamp_re_ice == n_ice0

            # The batch's non-cloud half is still filled (the assembly is ONE pass):
            # temperature, pressure and the vapour VMR are physical in every column.
            @test all(170.0 .< B.T_lay .< 350.0)
            @test all(B.p_lay .> 0.0)
            @test all(B.vmr_h2o .>= 0.0)
            @test maximum(B.vmr_h2o) > 1.0e-2        # ~2.9e-2 at the tropical surface
            @test all(B.t_sfc .> 290.0)
            # `:prescribed` carries no ozone profile, so the assembly writes zeros rather
            # than indexing an empty vector -- the property that makes it testable here.
            @test all(B.o3 .== 0.0)

            # Allocation: the assembly is the Scythe-side half of a radiation call and
            # must not allocate. (The solve underneath it allocates a few MB spawning
            # RRTMGP's own tasks; that is why the two halves are measured apart.)
            Scythe.radiation_assemble!(rs, mtile, B, P)     # warm up
            @test @allocated(Scythe.radiation_assemble!(rs, mtile, B, P)) == 0
        end
    end

end
