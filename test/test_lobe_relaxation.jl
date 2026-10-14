# test/test_lobe_relaxation.jl -- Stage L of reference/RHO_T_RETIREMENT_DESIGN.md: the LOBE
# RELAXATION of the transformed water-MASS slots (`options[:lobe_relaxation]`,
# `physical_params[:tau_lobe]`). Where a cloud, rain or ice-mass control variable has been rung
# below ν = 0 by the spline refit AND that species' channel is closed at the point (recovered
# density 0, every process term of the slot exactly 0), ν is relaxed toward 0 as
# ν ← ν·exp(−Δt/τ_lobe). Nothing else moves: the recovered density is 0 before and after, so
# ρ_t, ρ_d, the vapor, E_t, p and Q_ss — and every number and habit moment — are untouched.
#
# Every assertion compares two tiles advanced from the SAME state with the device on and off,
# on the SAME step (later steps legitimately diverge: the relaxed ν enters the next refit), so
# advection, the acoustic solve and every microphysical source cancel identically and what is
# left is the device itself. The tile is a periodic XZ slice with 500 m vertical cells, the
# RCE's vertical resolution, and every seed is horizontally uniform, as the pre-convective
# lobe layer is (§2.1).

using Test
using Scythe
using Springsteel
using SparseArrays
using Springsteel: PeriodicBC, NeumannBC, DirichletBC

@testset "lobe relaxation (Stage L)" begin

    # ── fixture ──────────────────────────────────────────────────────────────────────
    SPAN = 8.0e3                          # periodic span [m]: 4 cells of 2 km
    NCELL = 4
    ZTOP = 10.0e3                         # 20 cells of 500 m (the RCE's dz)
    NCELL_K = 20
    MU = 1.0e-7                           # cloud / rain widths (the shipped defaults)
    MU_NR = 1.0

    """Reference column: lapse-rate 6.5 K/km, RH `rh(z)`, cloud-free."""
    function lr_column(z, rh)
        Tk = @. 300.0 - 0.0065 * z
        p_Pa = @. 100000.0 * exp(-z / 8000.0)
        rho_v = rh.(z) .* Scythe.rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Scythe.Rv .* Tk .* rho_v)) ./ (Scythe.Rd .* Tk)
        return (; Tk, p_Pa, rho_d, rho_v, rho_c = zeros(length(z)))
    end

    """A periodic XZ RiRk tile, every water slot under `:bhyp` (or `tr`). `lobe` is the
    `options[:lobe_relaxation]` value, or `nothing` to leave the key absent; `tau` likewise for
    `physical_params[:tau_lobe]`. `seed!(patch, pts, V)` writes the perturbation state; the
    tile is then refit (spectral + grid transform), so the slots hold what the model's own
    refit makes of the seed — the lobes included."""
    function lr_tile(tmpdir, tag; seed! = (p, x, V) -> nothing, lobe = false, tau = nothing,
                     attr = false, ice = false, precip = ice, tr = :bhyp, ts = 0.5,
                     rh = z -> 0.7, ncell_k = NCELL_K, ztop = ZTOP)
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :precipitation => precip, :vertical_mixing => false,
                                :condensate_transform => tr, :rain_transform => tr)
        lobe === nothing || (opts[:lobe_relaxation] = lobe)
        attr && (opts[:water_lobe_attribution] = true)
        if ice
            opts[:rain_moments] = 2
            opts[:rain_number_transform] = tr
            opts[:ice_microphysics] = :ishmael
            opts[:ice_transform] = tr
        end
        varnames = Scythe.mc_var_names(opts)
        vars = Dict(v => i for (i, v) in enumerate(varnames))
        side_bc = Dict{String,Any}(v => PeriodicBC() for v in varnames)
        wall_bc = merge(Dict{String,Any}(v => NeumannBC() for v in varnames),
                        Dict{String,Any}("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = SPAN, num_cells_i = NCELL,
            kMin = 0.0, kMax = ztop, num_cells_k = ncell_k,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
        ref_file = joinpath(tmpdir, "lr_$(tag).ref")
        pp = Dict{Symbol,Float64}(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                                  :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                  :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3,
                                  :f => 0.0, :N_r => 1.0e-3)
        tau === nothing || (pp[:tau_lobe] = tau)
        model = ModelParameters(ts = ts, integration_time = 10.0, output_interval = 10.0,
            equation_set = "moist_compressible_XZ", ref_state_file = ref_file,
            grid_params = gp, physical_params = pp, options = opts)
        gp = model.grid_params
        patch = createGrid(gp)
        pts = Scythe.getGridpoints(patch)
        z = pts[1:gp.kDim, end]
        col = lr_column(z, rh)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        seed!(patch, pts, gp.vars)
        spectralTransform!(patch)
        gridTransform!(patch)
        hrm = sparse(Int64[], Int64[], Float64[], size(patch.spectral, 1),
                     size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, hrm)
        return mtile, patch, model
    end

    ncols(patch, model) = div(size(patch.physical, 1), model.grid_params.kDim)
    advance_all!(m, p, mo; t = 1) =
        (for c in 1:ncols(p, mo); Scythe.advance_column(m, c, t); end)

    """One step of two identically seeded tiles, device `on` vs `off`: (on tile, off tile,
    on patch, model, the state-n slots)."""
    function one_step(tmp, tag; on = true, off = false, kw...)
        m1, p1, mo1 = lr_tile(tmp, "$(tag)_on"; lobe = on, kw...)
        m0, p0, mo0 = lr_tile(tmp, "$(tag)_off"; lobe = off, kw...)
        @test p1.physical == p0.physical                    # the same state n
        nu_n = copy(p1.physical[:, :, 1])
        advance_all!(m1, p1, mo1)
        advance_all!(m0, p0, mo0)
        return m1, m0, p1, mo1, nu_n
    end

    """The kinked boundary-layer sheet of §2.1 / the 1-D toy: the positive part of a parabola
    from `zb` to `zt` [m], amplitude `amp` [kg/m^3], horizontally uniform. Its top is sharper
    than one 500 m cell, so the refit rings it negative in the air above."""
    kink(z; zb = 200.0, zt = 1150.0, amp = 1.0e-4) =
        max(0.0, amp * (z - zb) * (zt - z) / ((zt - zb) / 2)^2)

    function seed_sheet!(patch, pts, V)
        rc_i = Scythe.mc_slot(V, "rho_c")
        for i in 1:size(patch.physical, 1)
            rc = kink(pts[i, end])
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rc, 0.0, :bhyp, MU)
            patch.physical[i, V["rho_t"], 1] = rc      # the partition starts closed
        end
    end

    a_of(ts, tau) = -expm1(-ts / tau)

    # ── 1. inert ─────────────────────────────────────────────────────────────────────
    @testset "bitwise inert: switch off, every slot >= 0, rest, :none" begin
        mktempdir() do tmp
            # (a) The switch off is the absent key's behaviour of the pre-stage code: the key
            # set to `false` and to the shipped default resolve identically to its absence.
            for v in (false, Scythe.LOBE_RELAXATION_DEFAULT)
                mA, pA, moA = lr_tile(tmp, "abs_$(v)"; seed! = seed_sheet!, lobe = nothing)
                mB, pB, moB = lr_tile(tmp, "set_$(v)"; seed! = seed_sheet!, lobe = v)
                advance_all!(mA, pA, moA); advance_all!(mB, pB, moB)
                v == Scythe.LOBE_RELAXATION_DEFAULT &&
                    (@test mA.var_np1 == mB.var_np1 && mA.expdot_n == mB.expdot_n)
            end
            # (b) The sheet, on vs off: the tendencies are identical (the device is not a
            # tendency), and the state differs ONLY at points whose state-n mass slot was < 0.
            m1, m0, p1, mo, nu_n = one_step(tmp, "sheet"; seed! = seed_sheet!)
            @test m1.expdot_n == m0.expdot_n
            V = mo.grid_params.vars
            mass = [Scythe.mc_slot(V, "rho_c"), Scythe.mc_slot(V, "rho_r")]
            anyneg = falses(size(nu_n, 1))
            for s in mass
                anyneg .|= nu_n[:, s] .< 0.0
            end
            @test any(anyneg)                       # the sheet's refit does ring negative
            @test m1.var_np1[.!anyneg, :] == m0.var_np1[.!anyneg, :]
            # (c) Every slot >= 0 everywhere (cloud-free, no lobes): bitwise.
            m1, m0, _, _, nu_n = one_step(tmp, "clean")
            @test all(nu_n .== 0.0)
            @test m1.var_np1 == m0.var_np1 && m1.expdot_n == m0.expdot_n
            # (d) Under `:none` the slots are densities: nothing to relax, bitwise.
            m1, m0, _, _, _ = one_step(tmp, "none"; seed! = seed_sheet!, tr = :none)
            @test m1.var_np1 == m0.var_np1
        end
    end

    # ── 2. the gate ──────────────────────────────────────────────────────────────────
    @testset "closed channel: nu <- nu*exp(-ts/tau), never across zero" begin
        mktempdir() do tmp
            ts = 0.5
            m1, m0, p1, mo, nu_n = one_step(tmp, "closed"; seed! = seed_sheet!, ts = ts)
            V = mo.grid_params.vars
            c = Scythe.mc_slot(V, "rho_c")
            z = Scythe.getGridpoints(p1)[:, end]
            a = a_of(ts, 10.0)                       # tau_lobe defaults to 10 s
            # Above the sheet (RH 70 %, no cloud, S < 0): the cloud channel is closed at every
            # lobe point, and at rest the slot has no other tendency there, so the step is
            # exactly ν·(1 − a) on top of the off arm's ν* = ν^n.
            lobe = (nu_n[:, c] .< 0.0) .& (z .> 1500.0)
            @test count(lobe) > 0
            @test m0.var_np1[lobe, c] == nu_n[lobe, c]
            @test m1.var_np1[lobe, c] == nu_n[lobe, c] .+ (-(a .* nu_n[lobe, c]))
            @test all(m1.var_np1[lobe, c] .<= 0.0)
            @test isapprox(m1.var_np1[lobe, c] ./ nu_n[lobe, c],
                           fill(exp(-ts / 10.0), count(lobe)); rtol = 1.0e-14)
            # Every changed point is a lobe point, and moved toward zero, not past it.
            ch = m1.var_np1[:, c] .!= m0.var_np1[:, c]
            @test all(nu_n[ch, c] .< 0.0)
            @test all(m1.var_np1[ch, c] .> m0.var_np1[ch, c])
            @test all(m1.var_np1[ch, c] .<= 0.0)
            # tau -> 0: a = 1 lands exactly ON zero, never above it.
            m1, m0, _, _, nu_n = one_step(tmp, "tau0"; seed! = seed_sheet!, ts = ts,
                                          tau = 1.0e-9)
            ch = m1.var_np1[:, c] .!= m0.var_np1[:, c]
            @test any(ch) && all(m1.var_np1[ch, c] .== 0.0)
            # tau is a rate: a second tau gives exp(-ts/tau) again.
            m1, _, _, _, nu_n = one_step(tmp, "tau3"; seed! = seed_sheet!, ts = ts, tau = 3.0)
            @test isapprox(m1.var_np1[lobe, c] ./ nu_n[lobe, c],
                           fill(exp(-ts / 3.0), count(lobe)); rtol = 1.0e-14)
        end
    end

    @testset "live channel: a lobe point that is condensing is left alone" begin
        # The same sheet, with the air above it SUPERSATURATED (vapor and Q_ss raised to
        # RH 101.5 % above 1.5 km): S > 1e-4 opens nucleation (`qss_condensation_rates`,
        # invtau_c > 0) at the lobe points, so the step-mean condensation `qc_bar` is
        # nonzero there and the device must not touch them.
        mktempdir() do tmp
            function seed_super!(patch, pts, V)
                seed_sheet!(patch, pts, V)
                rv_i = Scythe.mc_slot(V, "rho_v")
                for i in 1:size(patch.physical, 1)
                    zz = pts[i, end]
                    zz > 1500.0 || continue
                    Tk = 300.0 - 0.0065 * zz
                    pp = 100000.0 * exp(-zz / 8000.0)
                    rvs = Scythe.rho_v_sat(Tk, pp / 100.0)
                    drv = (1.015 - 0.7) * rvs
                    patch.physical[i, rv_i, 1] += drv
                    patch.physical[i, V["rho_t"], 1] += drv
                    patch.physical[i, V["Q_ss"], 1] += drv
                end
            end
            m1, m0, p1, mo, nu_n = one_step(tmp, "live"; seed! = seed_super!)
            V = mo.grid_params.vars
            c = Scythe.mc_slot(V, "rho_c")
            z = Scythe.getGridpoints(p1)[:, end]
            lobe = (nu_n[:, c] .< 0.0) .& (z .> 2000.0)
            @test count(lobe) > 0
            # the channel really is open there: the off arm condenses onto the lobe point
            @test all(m0.var_np1[lobe, c] .> nu_n[lobe, c])
            @test m1.var_np1[lobe, c] == m0.var_np1[lobe, c]
        end
    end

    # A horizontally uniform column whose rain or ice-1 MASS slot is negative everywhere
    # (an offset under a wide dip, so the refit cannot ring above zero): nothing in the column
    # is positive, so the sedimentation flux is identically zero and its fitted divergence an
    # exact 0.0 — the only way the exact-zero gate on `F_z` can close for a sedimenting slot.
    neg_profile(z; z0 = 3000.0) = -5.0e-8 * (1.0 + exp(-((z - z0) / 2000.0)^2))

    @testset "rain: closed column relaxes; a lobe under falling rain does not" begin
        mktempdir() do tmp
            ts = 0.5
            a = a_of(ts, 10.0)
            function seed_rneg!(patch, pts, V)
                r = Scythe.mc_slot(V, "rho_r")
                for i in 1:size(patch.physical, 1)
                    patch.physical[i, r, 1] = neg_profile(pts[i, end])
                end
            end
            m1, m0, p1, mo, nu_n = one_step(tmp, "rneg"; seed! = seed_rneg!, precip = true)
            r = Scythe.mc_slot(mo.grid_params.vars, "rho_r")
            @test all(nu_n[:, r] .< 0.0)
            @test m0.var_np1[:, r] == nu_n[:, r]
            @test m1.var_np1[:, r] == nu_n[:, r] .+ (-(a .* nu_n[:, r]))
            # A kinked rain shaft: the refit rings it negative below/above, and the rain
            # falling through the column makes the fitted flux divergence nonzero at those
            # lobe points — the channel is open, the device leaves them alone.
            function seed_rshaft!(patch, pts, V)
                r = Scythe.mc_slot(V, "rho_r")
                for i in 1:size(patch.physical, 1)
                    rr = kink(pts[i, end]; zb = 2000.0, zt = 2900.0, amp = 3.0e-4)
                    patch.physical[i, r, 1] = Scythe.rain_slot(rr, :bhyp, MU)
                    patch.physical[i, V["rho_t"], 1] = rr
                end
            end
            m1, m0, p1, mo, nu_n = one_step(tmp, "rshaft"; seed! = seed_rshaft!,
                                            precip = true)
            lobe = nu_n[:, r] .< 0.0
            @test count(lobe) > 0
            @test m1.var_np1[:, r] == m0.var_np1[:, r]
        end
    end

    # ── 3. water neutrality ──────────────────────────────────────────────────────────
    @testset "water-neutral: only the lobe's nu moves; nothing recovered moves" begin
        mktempdir() do tmp
            m1, m0, p1, mo, nu_n = one_step(tmp, "neutral"; seed! = seed_sheet!)
            V = mo.grid_params.vars
            c = Scythe.mc_slot(V, "rho_c")
            @test m1.var_np1[:, c] != m0.var_np1[:, c]          # the device did act
            for s in 1:size(m1.var_np1, 2)
                s == c && continue
                @test m1.var_np1[:, s] == m0.var_np1[:, s]
            end
            # ρ_t, ρ_d, the vapor, E_t, p, Q_ss by name (the loop above covers them; this is
            # the statement the note makes, spelled out)
            for nm in ("rho_t", "rho_d", "rho_v", "E_t", "p", "Q_ss", "u", "w")
                s = Scythe.mc_slot(V, nm)
                @test m1.var_np1[:, s] == m0.var_np1[:, s]
            end
            # The recovered cloud is bitwise the same: 0 at every point the device touched.
            rc1 = Scythe.recover_total.(m1.var_np1[:, c], :bhyp, MU)
            rc0 = Scythe.recover_total.(m0.var_np1[:, c], :bhyp, MU)
            @test rc1 == rc0
        end
    end

    # ── 4. the held kinked sheet: the pool the refit builds, with and without ─────────
    @testset "held kinked sheet: the lobe pool grows > 10x slower with the device" begin
        # The model-level analogue of the 1-D toy (`preconv/toy1d_b.jl`): the §2.1 sheet, in
        # SATURATED air (RH 100 % below its 1150 m top, 70 % above), is HELD — each step,
        # after the model's own step, the cloudy levels are relaxed toward the kinked target
        # on 10 s (the toy's hold, standing in for the LW cooling and MYNN that hold the real
        # sheet) — and the model does the rest: its refit, its ETD evaporation of the positive
        # rings in the subsaturated air above, and (on) the device. The pool is
        # Σ 2 min(ν_c, 0), column-integrated, after each refit. The first ten minutes are
        # the imposition transient (the toy's "imposed once" bounded pool); the RATE is
        # measured over the second ten, where the off arm grows secularly.
        mktempdir() do tmp
            ts = 0.5
            nsteps = 2400                            # 20 minutes
            rh_sheet(z) = z < 1150.0 ? 1.0 : 0.7
            function pool_run(on)
                m, p, mo = lr_tile(tmp, "held_$(on)"; seed! = seed_sheet!, lobe = on, ts = ts,
                                   rh = rh_sheet)
                V = mo.grid_params.vars
                c = Scythe.mc_slot(V, "rho_c")
                z = Scythe.getGridpoints(p)[:, end]
                T = Scythe.condensate_slot.(kink.(z), 0.0, :bhyp, MU)
                cloudy = T .> 0.0
                ah = a_of(ts, 10.0)
                kDim = mo.grid_params.kDim
                dzc = ZTOP / NCELL_K
                wz = [(5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)[mod1(k, 3)] * dzc for k in 1:kDim]
                w = repeat(wz, ncols(p, mo))
                pool() = sum(w .* 2.0 .* min.(p.physical[:, c, 1], 0.0)) / ncols(p, mo)
                cloud() = sum(w .* Scythe.recover_total.(p.physical[:, c, 1], :bhyp, MU)) /
                          ncols(p, mo)
                pools = Float64[pool()]
                for n in 1:nsteps
                    advance_all!(m, p, mo; t = n)
                    @views m.var_np1[cloudy, c] .+= (T[cloudy] .- m.var_np1[cloudy, c]) .* ah
                    Scythe.calcTendency(m)
                    gridTransform!(p)
                    n % 300 == 0 && push!(pools, pool())
                end
                @test all(isfinite, p.physical[:, :, 1])
                return pools, cloud()
            end
            off, cloud_off = pool_run(false)
            on, cloud_on = pool_run(true)
            # kg m^-2 per hour over minutes 10-20 (pools are every 150 s)
            rate(v) = (v[end] - v[5]) / (600.0 / 3600.0)
            r_off = rate(off); r_on = rate(on)
            @info "held kinked sheet: column lobe pool every 150 s [kg/m^2]; rate over 10-20 min [kg/m^2/h]" off on r_off r_on cloud_off cloud_on
            @test r_off < -0.1                       # the refit really builds a pool, secularly
            @test abs(r_on) < 0.1 * abs(r_off)       # > 90 % reduction of the growth rate
            @test on[end] > 0.1 * off[end]           # and the standing pool is a fraction
            @test cloud_on > 0.5 * cloud_off         # the held cloud itself is still there
        end
    end

    # ── 5. ice: the mass relaxes, the moments do not ─────────────────────────────────
    @testset "ice mass lobe relaxes; number and habit moments are untouched" begin
        mktempdir() do tmp
            ts = 0.5
            a = a_of(ts, 10.0)
            function seed_ineg!(patch, pts, V)
                ii = Scythe.mc_ice_slot_indices(V)
                for i in 1:size(patch.physical, 1)
                    zz = pts[i, end]
                    patch.physical[i, ii[1], 1] = neg_profile(zz; z0 = 7000.0) * 1.0e-3
                    # a carried population, as the moments' own transport leaves them
                    n = 1.0e3 * exp(-((zz - 7000.0) / 1500.0)^2)
                    patch.physical[i, ii[2], 1] = Scythe.total_slot(n, :bhyp,
                                                      Scythe.MC_ICE_MU_DEFAULTS[2])
                    patch.physical[i, ii[3], 1] = Scythe.total_slot(1.0e-13 * n, :bhyp,
                                                      Scythe.MC_ICE_MU_DEFAULTS[3])
                    patch.physical[i, ii[4], 1] = Scythe.total_slot(2.0e-13 * n, :bhyp,
                                                      Scythe.MC_ICE_MU_DEFAULTS[4])
                end
            end
            m1, m0, p1, mo, nu_n = one_step(tmp, "ineg"; seed! = seed_ineg!, ice = true)
            V = mo.grid_params.vars
            ii = Scythe.mc_ice_slot_indices(V)
            q1 = ii[1]
            @test all(nu_n[:, q1] .< 0.0)
            @test m1.var_np1[:, q1] == m0.var_np1[:, q1] .+ (-(a .* nu_n[:, q1]))
            @test all(m1.var_np1[:, q1] .<= 0.0)
            @test all(Scythe.recover_total.(m1.var_np1[:, q1], :bhyp,
                                            Scythe.MC_ICE_MU_DEFAULTS[1]) .== 0.0)
            for s in 1:size(m1.var_np1, 2)
                s == q1 && continue
                @test m1.var_np1[:, s] == m0.var_np1[:, s]
            end
        end
    end

    # ── 6. refusal, config, allocations ──────────────────────────────────────────────
    @testset "refused with water_lobe_attribution; tau_lobe must be positive" begin
        mktempdir() do tmp
            m, p, mo = lr_tile(tmp, "refuse"; lobe = true, attr = true)
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
            m, p, mo = lr_tile(tmp, "attr_off"; lobe = false, attr = true)
            Scythe.advance_column(m, 1, 1)                     # attribution alone runs
            @test all(isfinite, m.var_np1)
            m, p, mo = lr_tile(tmp, "tau_neg"; lobe = true, tau = 0.0)
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
        end
    end

    @testset "config round trip" begin
        mktempdir() do tmp
            owb = joinpath(pkgdir(Scythe), "simulations", "ocean_warm_bubble",
                           "scythe_config.toml")
            cfg = load_config(owb; set = ["output.dir=$(tmp)",
                                          "extra.options.lobe_relaxation=false",
                                          "extra.physical_params.tau_lobe=20"])
            @test cfg.extra.options[:lobe_relaxation] === false
            @test cfg.extra.physical_params[:tau_lobe] === 20.0
            m = build_model(cfg)
            @test m.options[:lobe_relaxation] === false
            @test m.physical_params[:tau_lobe] == 20.0
            path = write_config(cfg, joinpath(tmp, "rt.toml"))
            back = load_config(path)
            @test back.extra.options[:lobe_relaxation] === false
            @test back.extra.physical_params[:tau_lobe] === 20.0
            @test haskey(Scythe.KNOWN_OPTION_KEYS, :lobe_relaxation)
            @test haskey(Scythe.KNOWN_PHYSICAL_KEYS, :tau_lobe)
        end
    end

    @testset "allocation-free with the device acting" begin
        mktempdir() do tmp
            for ice in (false, true)
                seed = ice ? ((p, x, V) -> begin
                                  seed_sheet!(p, x, V)
                                  q1 = Scythe.mc_ice_slot_indices(V)[1]
                                  for i in 1:size(p.physical, 1)
                                      p.physical[i, q1, 1] = neg_profile(x[i, end]; z0 = 7000.0)
                                  end
                              end) : seed_sheet!
                m, p, mo = lr_tile(tmp, "alloc_$(ice)"; seed! = seed, lobe = true, ice = ice)
                kDim = mo.grid_params.kDim
                Scythe.moist_compressible_XZ(m, 1, kDim, 2)          # compile
                @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 2)) == 0
            end
        end
    end
end
