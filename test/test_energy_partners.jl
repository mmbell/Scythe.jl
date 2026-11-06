# test/test_energy_partners.jl -- energy-consistent water-partition bookkeeping under the EOS
# temperature. With `T = p/(ρ_d R_d + ρ_v R_v)` the prognostic `E_t` no longer defines T, and
# every operator that moves a transformed condensate's RECOVERED density by a different amount
# than its partners assume opens the residual `R = E_t − E_implied(p, ρ_d, ρ_v, ρ_liq, ρ_ice,
# T_eos)` (the closed per-operator budget of 2026-10-16). The cures, each a switch:
#
#   1. `options[:exchange_realization]` — the ETD phase change books its heat and Q_ss partners
#      on the mass the slots' recovered densities actually receive (under the derived water
#      anchor, the only one since Stage 4 of the rho_t retirement, the vapor slot holds the
#      nu-space closure of the exchange and is not re-written: rho_w = sum_k alpha_k nu_k is
#      invariant and the realization is energy only);
#   2. `options[:refit_partner]` — the spline refit's change in the recovered water is booked
#      energy-consistently (under the derived anchor: the affine defect, energy only).
# Section 3b: the refit partner under nesting (`refit_reference_after_exchange!`).
#
# RETIRED at Stage 4 with slot 3 (the prognostic rho_t), and their tests with them: the MYNN
# clip-gap partner (`options[:mynn_exchange_realization]`, `_mynn_clip_gap`), the vapor
# reconciliation nudge's pressure partner (`options[:vapor_nudge_partner]`,
# `vapor_nudge_partner_rate`) and the MYNN total-water closure (`options[:mynn_water_closure]`).
# The derived-anchor forms of the realization (E4), the refit partner (E2) and the MYNN legs
# (E7) are tested in test_water_anchor.jl.
#
# Every assertion is on ONE step of tiles advanced from the same state, so the property tested
# is the operator's own. The tile is the lobe-relaxation fixture: a periodic XZ slice with
# 500 m vertical cells (the RCE's dz), horizontally uniform, whose kinked boundary-layer cloud
# sheet the refit rings negative in the air above.

using Test
using Scythe
using Springsteel
using SparseArrays
using Random
using Springsteel: PeriodicBC, NeumannBC, DirichletBC

@testset "energy-consistent water partners" begin

    SPAN = 8.0e3; NCELL = 4
    ZTOP = 10.0e3; NCELL_K = 20
    MU = 1.0e-7

    function lr_column(z, rh)
        Tk = @. 300.0 - 0.0065 * z
        p_Pa = @. 100000.0 * exp(-z / 8000.0)
        rho_v = rh.(z) .* Scythe.rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Scythe.Rv .* Tk .* rho_v)) ./ (Scythe.Rd .* Tk)
        return (; Tk, p_Pa, rho_d, rho_v, rho_c = zeros(length(z)))
    end

    """A periodic XZ RiRk tile, cloud and rain under `tr`. `opts_extra` is merged into the
    options; `seed!(patch, pts, V)` writes the perturbation state, which is then refit."""
    function ep_tile(tmpdir, tag; seed! = (p, x, V) -> nothing, opts_extra = Dict{Symbol,Any}(),
                     precip = false, tr = :bhyp, ts = 0.5, rh = z -> 0.7, ncell = NCELL,
                     bcl = PeriodicBC(), bcr = PeriodicBC())
        # The lobe relaxation (Stage L) is held OFF: since Stage 4 it defaults ON (the paired
        # form, which moves the vapor, p and Q_ss at lobe points inside the same ETD stage);
        # these tests isolate the partners, as they did when Stage L defaulted off, and the
        # paired Stage L has its own tests (test_lobe_relaxation.jl, test_water_anchor.jl).
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :precipitation => precip, :vertical_mixing => false,
                                :condensate_transform => tr, :rain_transform => tr,
                                :lobe_relaxation => false)
        merge!(opts, opts_extra)
        varnames = Scythe.mc_var_names(opts)
        vars = Dict(v => i for (i, v) in enumerate(varnames))
        side_l = Dict{String,Any}(v => bcl for v in varnames)
        side_r = Dict{String,Any}(v => bcr for v in varnames)
        wall_bc = merge(Dict{String,Any}(v => NeumannBC() for v in varnames),
                        Dict{String,Any}("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = SPAN * ncell / NCELL, num_cells_i = ncell,
            kMin = 0.0, kMax = ZTOP, num_cells_k = NCELL_K,
            BCL = side_l, BCR = side_r, BCB = wall_bc, BCT = wall_bc, vars = vars)
        ref_file = joinpath(tmpdir, "ep_$(tag).ref")
        pp = Dict{Symbol,Float64}(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                                  :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                  :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3,
                                  :f => 0.0, :N_r => 1.0e-3)
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
    scratch(m) = m.mc_scratch[Threads.threadid()]

    kink(z; zb = 200.0, zt = 1150.0, amp = 1.0e-4) =
        max(0.0, amp * (z - zb) * (zt - z) / ((zt - zb) / 2)^2)

    function seed_sheet!(patch, pts, V; tr = :bhyp)
        rc_i = Scythe.mc_slot(V, "rho_c")
        for i in 1:size(patch.physical, 1)
            rc = kink(pts[i, end])
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rc, 0.0, tr, MU)
        end
    end

    """The sheet with the air above 1.5 km SUPERSATURATED (RH 101.5 %): nucleation opens at the
    lobe points, so the step-mean condensation lands on slots whose recovered density is 0."""
    function seed_super!(patch, pts, V; tr = :bhyp)
        seed_sheet!(patch, pts, V; tr)
        rv_i = Scythe.mc_slot(V, "rho_v")
        for i in 1:size(patch.physical, 1)
            zz = pts[i, end]
            zz > 1500.0 || continue
            Tk = 300.0 - 0.0065 * zz
            pp = 100000.0 * exp(-zz / 8000.0)
            drv = (1.015 - 0.7) * Scythe.rho_v_sat(Tk, pp / 100.0)
            patch.physical[i, rv_i, 1] += drv
            patch.physical[i, V["Q_ss"], 1] += drv
        end
    end

    """A healthy cloud: `amp` at every level (ν ≫ μ, J ≈ ½), in air of RH `rh`."""
    function seed_uniform!(patch, pts, V; amp = 1.0e-3)
        rc_i = Scythe.mc_slot(V, "rho_c")
        for i in 1:size(patch.physical, 1)
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(amp, 0.0, :bhyp, MU)
        end
    end

    # ── the residual R = E_t − E_implied(T_eos), pointwise, of a slot matrix ──────────
    """`R` and the recovered liquid at every gridpoint of the slot matrix `X` (the layout of
    `var_np1`: perturbation slots, slot 9 against `ν̄_c ≡ 0`, slot 8 a total). The mechanical
    term reads the DERIVED total `ρ_t = ρ_d + W`, `W = Σ_k α_k ν_k` over the water-mass slots
    (the energy budget's definition, `_eb_point_RX`; there is no ρ_t slot since Stage 4). Also
    returns `W` at every point."""
    function residual(m, X; tr = :bhyp)
        mo = m.model; V = mo.grid_params.vars; kDim = mo.grid_params.kDim
        # the vapor as the model reads it: `recover_rho_v` against the tile's own `nu_vbar`
        # (≡ rho_vbar, bit for bit, under the default `vapor_transform = :none`)
        vtr = Scythe.vapor_transform_mode(mo.options)
        vmu = Scythe.vapor_mu(mo.physical_params)
        nuvbar = Scythe.mc_nu_vbar(m)
        rs = m.ref_state
        pbar = Scythe.ref_pressure(rs)[:, 1]; rdbar = Scythe.ref_rho_d(rs)[:, 1]
        Ebar = Scythe.ref_total_energy(rs)[:, 1]
        rvbar = m.mc_ref_diag.rho_vbar
        z = m.tilepoints[:, end]
        rv_i = Scythe.mc_slot(V, "rho_v")
        ws = Scythe.water_anchor_set(mo.options, mo.physical_params, m.mc_slots)
        av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]
        n = size(X, 1)
        R = zeros(n); rl = zeros(n); rv = zeros(n); W = zeros(n); Wsc = zeros(n)
        for g in 1:n
            k = mod1(g, kDim)
            p = X[g, 1] + pbar[k]; rd = X[g, 2] + rdbar[k]
            W[g] = av * (X[g, rv_i] + nuvbar[k]) + ac * X[g, 9] + ar * X[g, 8]
            Wsc[g] = av * abs(X[g, rv_i] + nuvbar[k]) + ac * abs(X[g, 9]) + ar * abs(X[g, 8])
            rt = rd + W[g]
            u = X[g, 4]; w = X[g, 5]; Et = X[g, 6] + Ebar[k]
            rv[g] = vtr === :none ? X[g, rv_i] + rvbar[k] :
                                    Scythe.recover_rho_v(X[g, rv_i], nuvbar[k], vtr, vmu)
            rl[g] = Scythe.recover_rho_c(X[g, 9], 0.0, tr, MU) +
                    Scythe.recover_rho_r(X[g, 8], tr, MU)
            geo = 0.5 * (u * u + w * w) + Scythe.gravity * z[g]
            T = Scythe.eos_temperature(p, rd, rv[g])
            R[g] = Et - Scythe.implied_total_energy(T, rd, rv[g], rl[g], 0.0, rt, geo)
        end
        return R, rl, rv, W, Wsc
    end
    # rho_w invariance through a stage, to the rounding of its slot additions: |ΔW| within a
    # few ulps of the summands Σ α|ν| on either side
    w_invariant(dW, sc0, sc1) = all(abs.(dW) .<= 8.0 * eps(Float64) .* (sc0 .+ sc1))

    """Re-apply the LAST column's relaxation adjustment of the step just taken onto the state
    `X0` (a copy, written into `m.var_np1` for the call): the ETD stage in isolation, on the
    scratch the step left behind. Returns the post-stage slot matrix."""
    function etd_stage(m, X0, kDim, nc)
        cs = (nc - 1) * kDim + 1
        m.var_np1 .= X0
        Scythe.relaxation_adjustment_qss!(m, cs, cs + kDim - 1, 1)
        return copy(m.var_np1), cs:(cs + kDim - 1)
    end

    # ── 1. the ETD exchange on the recovered density ─────────────────────────────────
    @testset "phase_change_partner is the ETD bracket and holds E_implied" begin
        Tk = 285.0; rd = 1.1; rv = 0.012; rl = 1.0e-4; rt = rd + rv + rl
        p = (rd * Scythe.Rd + rv * Scythe.Rv) * Tk
        qv = rv / rd; ql = rl / rd
        C_vt = Scythe.Cvd + qv * Scythe.Cvv + ql * Scythe.Cl
        R_m = Scythe.Rd + qv * Scythe.Rv
        C_pt = C_vt + R_m
        Lv = Scythe.L_v(Tk); Ls = Scythe.L_s(Tk)
        Qs = Scythe.Q_s_energy(Tk, p, rd, qv, ql)
        for (dl, di) in ((1.0e-7, 0.0), (-3.0e-8, 0.0), (0.0, 2.0e-8), (4.0e-8, -1.0e-8))
            dp, dv, dq = Scythe.phase_change_partner(dl, di, Tk, R_m, C_vt, C_pt, Lv, Ls, Qs, Qs)
            @test dv == -(dl + di)
            # the ETD's own etd_d1 with ts*rate -> mass
            @test dp ≈ (R_m / C_vt) * ((Lv - Scythe.Rv * C_pt * Tk / R_m) * dl +
                                       (Ls - Scythe.Rv * C_pt * Tk / R_m) * di) rtol = 1e-15
            @test dq ≈ -(1.0 + Qs) * (dl + di) rtol = 1e-14
            # E_implied is held to second order in the exchanged mass
            E0 = Scythe.implied_total_energy(Tk, rd, rv, rl, 0.0, rt, 0.0)
            T1 = Scythe.eos_temperature(p + dp, rd, rv + dv)
            E1 = Scythe.implied_total_energy(T1, rd, rv + dv, rl + dl, di, rt, 0.0)
            lin = abs(Lv * (dl + di))
            @test abs(E1 - E0) <= 1.0e-4 * lin                 # O(δ/ρ) of the exchanged energy
            # ...and the bare exchange (vapor moved, no heat) is the full latent energy
            E2 = Scythe.implied_total_energy(Scythe.eos_temperature(p, rd, rv + dv), rd,
                                             rv + dv, rl + dl, di, rt, 0.0)
            @test abs(E2 - E0) > 0.5 * lin
        end
    end

    # ── 1b. the one partner primitive (plan E1) and the affine defect (plan E2) ──────
    """A random thermodynamic point: `(Tk, rd, rv, rl, ri)` with the mixture coefficients and
    saturation derivatives the driver's scratch holds (`S.R_m`, `S.C_vt`, `S.C_pt`, `S.Lv`,
    `S.drvs_dT`, `S.drvs_dp`) and the two psychrometric factors."""
    function rcp_point(rng)
        Tk = 230.0 + 75.0 * rand(rng)
        rd = 0.3 + 0.9 * rand(rng)
        rv = 0.02 * rand(rng) * rd
        rl = 2.0e-3 * rand(rng); ri = 2.0e-3 * rand(rng)
        qv = rv / rd; ql = rl / rd; qi = ri / rd
        C_vt = Scythe.Cvd + qv * Scythe.Cvv + ql * Scythe.Cl + qi * Scythe.Ci
        R_m = Scythe.Rd + qv * Scythe.Rv
        C_pt = C_vt + R_m
        p = (rd * Scythe.Rd + rv * Scythe.Rv) * Tk
        Lv = Scythe.L_v(Tk); Ls = Scythe.L_s(Tk)
        dT = Scythe.drho_vsat_dT(Tk, p / 100.0); dP = Scythe.drho_vsat_dp(Tk, p / 100.0)
        Qs = Scythe.Q_s_energy(Tk, p, rd, qv, ql, qi)
        Qsi = Scythe.Q_s_energy_ice(Tk, p, rd, qv, ql, qi)
        return (; Tk, rd, rv, rl, ri, p, C_vt, R_m, C_pt, Lv, Ls, dT, dP, Qs, Qsi)
    end

    @testset "recovered_change_partner holds E_implied for any recovered change" begin
        rng = Random.MersenneTwister(20261019)
        for _ in 1:200
            x = rcp_point(rng)
            # both condensates change the same way (a condensate-condensate swap at fixed vapor
            # can cancel to nothing at p held, which would not test the partner) and the vapor
            # by anything up to their sum, either sign
            sg = rand(rng, (-1.0, 1.0))
            dl = sg * 1.0e-6 * rand(rng); di = sg * 1.0e-6 * rand(rng)
            dv = (2.0 * rand(rng) - 1.0) * (abs(dl) + abs(di))
            dp, dq = Scythe.recovered_change_partner(dv, dl, di, x.Tk, x.rd, x.R_m, x.C_vt,
                                                     x.Lv, x.Ls, x.dT, x.dP)
            rt = x.rd + x.rv + x.rl + x.ri
            E0 = Scythe.implied_total_energy(x.Tk, x.rd, x.rv, x.rl, x.ri, rt, 0.0)
            T1 = Scythe.eos_temperature(x.p + dp, x.rd, x.rv + dv)
            E1 = Scythe.implied_total_energy(T1, x.rd, x.rv + dv, x.rl + dl, x.ri + di, rt, 0.0)
            # the energy the three moves carry, coefficient by coefficient
            lin = abs(Scythe.Cvv * x.Tk * dv) + abs((Scythe.Cpv * x.Tk - x.Lv) * dl) +
                  abs((Scythe.Cpv * x.Tk - x.Ls) * di)
            @test abs(E1 - E0) <= 1.0e-4 * lin
            # unpartnered (p held), the move opens E_implied by most of that energy
            T2 = Scythe.eos_temperature(x.p, x.rd, x.rv + dv)
            E2 = Scythe.implied_total_energy(T2, x.rd, x.rv + dv, x.rl + dl, x.ri + di, rt, 0.0)
            @test abs(E2 - E0) > 0.5 * lin
            # dq is the vapor's change less the saturation chain rule of the partner's dT, dp
            dTk = -(Scythe.Cvv * x.Tk * dv + (Scythe.Cpv * x.Tk - x.Lv) * dl +
                    (Scythe.Cpv * x.Tk - x.Ls) * di) / (x.rd * x.C_vt)
            @test dq ≈ dv - (x.dT * dTk + x.dP * dp) rtol = 1.0e-12
            # and dp, dT obey the equation of state: dp = ρ_d R_m dT + R_v T dv
            @test dp ≈ x.rd * x.R_m * dTk + Scythe.Rv * x.Tk * dv rtol = 1.0e-10
        end
    end

    # The vapor-alone pressure partner, `T(R_v − (R_m/C_vt) C_vv)·δ` (the expression the retired
    # vapor nudge's `vapor_nudge_partner_rate` was, with a rate for the mass).
    vapor_alone_dp(d, Tk, R_m, C_vt) = Tk * (Scythe.Rv - ((R_m / C_vt) * Scythe.Cvv)) * d

    @testset "recovered_change_partner: the phase change and the vapor alone are special cases" begin
        rng = Random.MersenneTwister(1015)
        for _ in 1:200
            x = rcp_point(rng)
            dl, di = 1.0e-6 .* (2.0 .* rand(rng, 2) .- 1.0)
            for (l, i) in ((dl, di), (dl, 0.0), (0.0, di))
                dp0, dv0, dq0 = Scythe.phase_change_partner(l, i, x.Tk, x.R_m, x.C_vt, x.C_pt,
                                                           x.Lv, x.Ls, x.Qs, x.Qsi)
                dp, dq = Scythe.recovered_change_partner(dv0, l, i, x.Tk, x.rd, x.R_m, x.C_vt,
                                                         x.Lv, x.Ls, x.dT, x.dP)
                # rtol on the terms' scale: liquid condensing while ice sublimes can cancel
                b = Scythe.Rv * x.C_pt * x.Tk / x.R_m
                sp = (x.R_m / x.C_vt) * (abs((x.Lv - b) * l) + abs((x.Ls - b) * i))
                sq = abs((1.0 + x.Qs) * l) + abs((1.0 + x.Qsi) * i)
                @test isapprox(dp, dp0; rtol = 1.0e-13, atol = 1.0e-13 * sp)
                @test isapprox(dq, dq0; rtol = 1.0e-13, atol = 1.0e-13 * sq)
            end
            ts = 0.5
            vrec = 1.0e-7 * (2.0 * rand(rng) - 1.0)
            dp, dq = Scythe.recovered_change_partner(vrec * ts, 0.0, 0.0, x.Tk, x.rd, x.R_m,
                                                     x.C_vt, x.Lv, x.Ls, x.dT, x.dP)
            @test dp ≈ vapor_alone_dp(vrec, x.Tk, x.R_m, x.C_vt) * ts rtol = 1.0e-14
            # the vapor-only partner is that expression's bits
            @test dp === vapor_alone_dp(vrec * ts, x.Tk, x.R_m, x.C_vt)
        end
        # nothing moves: exact zeros, and a −0.0 vapor change is not flipped by the zero terms
        x = rcp_point(Random.MersenneTwister(7))
        dp, dq = Scythe.recovered_change_partner(0.0, 0.0, 0.0, x.Tk, x.rd, x.R_m, x.C_vt,
                                                 x.Lv, x.Ls, x.dT, x.dP)
        @test dp == 0.0 && dq == 0.0
        dp, dq = Scythe.recovered_change_partner(-0.0, 0.0, 0.0, x.Tk, x.rd, x.R_m, x.C_vt,
                                                 x.Lv, x.Ls, x.dT, x.dP)
        @test dp === vapor_alone_dp(-0.0, x.Tk, x.R_m, x.C_vt)
        @test dq === -0.0
    end

    @testset "affine_defect and its change across a refit" begin
        mu = MU
        for nu in (0.0, 1.0e-9, 1.0e-7, 3.0e-6, 1.0e-3)
            d = Scythe.affine_defect(nu, :bhyp, mu)
            # 2ν = ρ + μ s with s ∈ [0, 1): the defect is −μ s ∈ (−μ, 0]
            @test -mu < d <= 0.0
            @test d ≈ Scythe.recover_total(nu, :bhyp, mu) - 2.0 * nu rtol = 1.0e-14 atol = 1.0e-22
            @test Scythe.affine_defect(nu, :none, mu) === 0.0
        end
        @test Scythe.affine_defect(0.0, :bhyp, mu) === 0.0
        for nu in (-1.0e-9, -4.0e-7, -2.0e-5)
            @test Scythe.affine_defect(nu, :bhyp, mu) === -2.0 * nu
            @test Scythe.affine_defect(nu, :none, mu) === 0.0
        end
        # deep above the knee the defect saturates at −μ
        @test Scythe.affine_defect(1.0e-2, :bhyp, mu) ≈ -mu rtol = 1.0e-4   # −μ + μ²/(2ν)
        # across a refit: exactly 0.0 where the bits did not change
        for nu in (-3.0e-6, -0.0, 0.0, 2.0e-8, 5.0e-4)
            @test Scythe.affine_defect_change(nu, nu, :bhyp, mu) === 0.0
            @test Scythe.affine_defect_change(nu, nu, :none, mu) === 0.0
        end
        # a deep clipped lobe on both sides: the pool's growth, −2δν
        a, b = -3.0e-6, -2.5e-6
        @test Scythe.affine_defect_change(a, b, :bhyp, mu) === -2.0 * (a - b)
        @test Scythe.affine_defect_change(a, b, :none, mu) === 0.0
        # affine on both sides: O(μ s) and equal to the difference of the two defects
        a, b = 4.0e-4, 3.9e-4
        dd = Scythe.affine_defect_change(a, b, :bhyp, mu)
        @test abs(dd) < 1.0e-3 * mu
        @test dd ≈ Scythe.affine_defect(a, :bhyp, mu) - Scythe.affine_defect(b, :bhyp, mu) rtol = 1.0e-4
        # it is the recovered change less the anchor's: Δ = δρ − 2δν
        a, b = 3.0e-8, -1.0e-8
        @test Scythe.affine_defect_change(a, b, :bhyp, mu) ===
              Scythe.refit_recovered_change(a, b, :bhyp, mu) - 2.0 * (a - b)
    end

    @testset "the primitive and the defect allocate nothing" begin
        x = rcp_point(Random.MersenneTwister(3))
        Scythe.recovered_change_partner(1.0e-7, -2.0e-8, 3.0e-9, x.Tk, x.rd, x.R_m, x.C_vt,
                                        x.Lv, x.Ls, x.dT, x.dP)
        @test (@allocations Scythe.recovered_change_partner(1.0e-7, -2.0e-8, 3.0e-9, x.Tk,
                   x.rd, x.R_m, x.C_vt, x.Lv, x.Ls, x.dT, x.dP)) == 0
        Scythe.affine_defect(2.0e-7, :bhyp, MU); Scythe.affine_defect_change(2.0e-7, 1.0e-7, :bhyp, MU)
        @test (@allocations Scythe.affine_defect(2.0e-7, :bhyp, MU)) == 0
        @test (@allocations Scythe.affine_defect_change(2.0e-7, 1.0e-7, :bhyp, MU)) == 0
    end

    @testset "a clipped lobe: the heat moves by exactly the recovered change" begin
        # Under the derived water anchor (the only one since Stage 4) the ETD's vapor increment
        # is the nu-space CLOSURE of the condensate increments (`etd_dv`, rho_w invariant), and
        # the realization re-books only p and Q_ss on what the recovered densities received;
        # the vapor slot is not re-written. (The `:slot` form, which moved the vapor by exactly
        # the recovered cloud change, went with slot 3.)
        mktempdir() do tmp
            res = Dict{Bool,Any}()
            for on in (true, false)
                m, p, mo = ep_tile(tmp, "lobe_$(on)"; seed! = seed_super!,
                                   opts_extra = Dict{Symbol,Any}(:exchange_realization => on))
                advance_all!(m, p, mo)
                kDim = mo.grid_params.kDim; nc = ncols(p, mo)
                X0 = copy(m.var_np1)
                X1, col = etd_stage(m, X0, kDim, nc)
                R0, rl0, rv0, W0, s0 = residual(m, X0)
                R1, rl1, rv1, W1, s1 = residual(m, X1)
                S = scratch(m)
                rv_i = Scythe.mc_slot(mo.grid_params.vars, "rho_v")
                res[on] = (; dR = (R1 .- R0)[col], drl = (rl1 .- rl0)[col],
                           drv = (X1 .- X0)[col, rv_i], dW = (W1 .- W0)[col],
                           Wsc0 = s0[col], Wsc1 = s1[col],
                           x0v = X0[col, rv_i], x1v = X1[col, rv_i],
                           dv = S.etd_dv[1:kDim], req = S.Qdot_bar[1:kDim] .* mo.ts,
                           z = m.tilepoints[col, end], nu = X0[col, 9],
                           Lv = copy(S.Lv), d9 = copy(S.etd_d9))
            end
            on = res[true]; off = res[false]
            # the requested exchange and its closure are the same in both arms (one step, one
            # state): the realization writes no water
            @test on.dv == off.dv && on.req == off.req
            @test on.x1v == on.x0v .+ on.dv && off.x1v == off.x0v .+ off.dv
            # lobe points: the slot is below zero and stays there -- condensation that never
            # appears in the recovered density
            lobe = (on.nu .+ on.d9 .< 0.0) .& (on.d9 .> 0.0)
            @test count(lobe) > 0
            @test all(on.drl[lobe] .== 0.0)
            # rho_w is invariant through the ETD stage at every point, both arms (the closure)
            act = on.dv .!= 0.0
            @test count(act) > count(lobe)
            @test w_invariant(on.dW, on.Wsc0, on.Wsc1) && w_invariant(off.dW, off.Wsc0, off.Wsc1)
            # OFF (the requested booking): the heat of the full request with the recovered
            # cloud not moving -> R jumps by ~L*request at the lobe points
            @test all(abs.(off.dR[lobe]) .> 0.5 .* off.Lv[lobe] .* abs.(off.req[lobe]))
            # ON: the exchange is booked on the recovered change, and the ETD leaves R unchanged
            # to second order in the exchanged mass at every active point (the heat capacity the
            # exchange itself changes, at the frozen ETD coefficients)
            @test all(abs.(on.dR[act]) .<= 1.0e-3 .* on.Lv[act] .* abs.(on.req[act]))
            @test maximum(abs.(on.dR[act])) < 1.0e-3 * maximum(abs.(off.dR[act]))
        end
    end

    @testset "a healthy cloud: the realized and requested exchange agree to O(J linearization)" begin
        mktempdir() do tmp
            res = Dict{Bool,Any}()
            for on in (true, false)
                m, p, mo = ep_tile(tmp, "ok_$(on)"; seed! = seed_uniform!, rh = z -> 0.9,
                                   opts_extra = Dict{Symbol,Any}(:exchange_realization => on))
                advance_all!(m, p, mo)
                kDim = mo.grid_params.kDim; nc = ncols(p, mo)
                X0 = copy(m.var_np1)
                X1, col = etd_stage(m, X0, kDim, nc)
                _, rl0, rv0 = residual(m, X0); _, rl1, rv1 = residual(m, X1)
                res[on] = (; drv = (rv1 .- rv0)[col], drl = (rl1 .- rl0)[col],
                           dp = (X1 .- X0)[col, 1], req = copy(scratch(m).etd_dv[1:kDim]))
            end
            on = res[true]; off = res[false]
            act = on.req .!= 0.0
            @test count(act) == length(act)                 # it evaporates everywhere
            # the derived realization writes no water: the vapor and cloud are bitwise the
            # requested arm's (Stage 4; the `:slot` form re-booked the vapor here)
            @test on.drv == off.drv && on.drl == off.drl
            # the two heat bookings differ, by the Jacobian's linearization error and no more
            @test on.dp != off.dp
            @test maximum(abs.(on.dp[act] .- off.dp[act]) ./ abs.(off.dp[act])) < 1.0e-6
        end
    end

    @testset "bitwise: :none transforms, a dry column, and the switch off is the old booking" begin
        mktempdir() do tmp
            for (tag, kw) in (("none", (; seed! = (p, x, V) -> seed_super!(p, x, V; tr = :none),
                                        tr = :none)),
                              ("dry", (; seed! = (p, x, V) -> nothing)))
                m1, p1, mo1 = ep_tile(tmp, "$(tag)_on"; kw...,
                                      opts_extra = Dict{Symbol,Any}(:exchange_realization => true))
                m0, p0, mo0 = ep_tile(tmp, "$(tag)_off"; kw...,
                                      opts_extra = Dict{Symbol,Any}(:exchange_realization => false))
                advance_all!(m1, p1, mo1); advance_all!(m0, p0, mo0)
                @test m1.var_np1 == m0.var_np1
            end
            # the default is ON: the absent key is the `true` arm
            mA, pA, moA = ep_tile(tmp, "abs"; seed! = seed_super!)
            mB, pB, moB = ep_tile(tmp, "set"; seed! = seed_super!,
                                  opts_extra = Dict{Symbol,Any}(:exchange_realization => true))
            advance_all!(mA, pA, moA); advance_all!(mB, pB, moB)
            @test mA.var_np1 == mB.var_np1
            @test Scythe.EXCHANGE_REALIZATION_DEFAULT === true
            # OFF: the ETD stage applies exactly the requested increments to p and the vapor
            m0, p0, mo0 = ep_tile(tmp, "off2"; seed! = seed_super!,
                                  opts_extra = Dict{Symbol,Any}(:exchange_realization => false))
            advance_all!(m0, p0, mo0)
            kDim = mo0.grid_params.kDim; nc = ncols(p0, mo0)
            X0 = copy(m0.var_np1)
            X1, col = etd_stage(m0, X0, kDim, nc)
            S = scratch(m0)
            rv_i = Scythe.mc_slot(mo0.grid_params.vars, "rho_v")
            @test X1[col, rv_i] == X0[col, rv_i] .+ S.etd_dv[1:kDim]
            @test X1[col, 1] == X0[col, 1] .+ S.etd_d1[1:kDim]
        end
    end

    @testset "allocation-free with the exchange realization acting" begin
        mktempdir() do tmp
            m, p, mo = ep_tile(tmp, "alloc"; seed! = seed_super!)
            kDim = mo.grid_params.kDim
            Scythe.moist_compressible_XZ(m, 1, kDim, 2)
            @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 2)) == 0
        end
    end

    # ── 2. (retired) the MYNN mass legs ──────────────────────────────────────────────
    # "_mynn_clip_gap" and "MYNN cloud leg into a lobe" were deleted at Stage 4 of the rho_t
    # retirement: the clip-gap partner (`options[:mynn_exchange_realization]`) was a `:slot`-
    # anchor device and went with slot 3; the derived anchor's MYNN legs are nu legs realized on
    # their own recovered rates (test_water_anchor.jl, "3c E7").

    # ── 3. the refit partner ─────────────────────────────────────────────────────────
    """Step 1, the model's own refit (`calcTendency` + `gridTransform!`), step 2: returns the
    pre-refit slots (var_np1 after step 1), the post-refit slots, and the tile."""
    function refit_two_steps(tmp, tag; on = true, extra = Dict{Symbol,Any}(), kw...)
        m, p, mo = ep_tile(tmp, tag;
                           opts_extra = merge(Dict{Symbol,Any}(:refit_partner => on), extra),
                           kw...)
        advance_all!(m, p, mo; t = 1)
        xpre = copy(m.var_np1)
        Scythe.calcTendency(m)
        gridTransform!(p)
        xpost = copy(p.physical[:, :, 1])
        advance_all!(m, p, mo; t = 2)
        return m, p, mo, xpre, xpost
    end

    # "refit partner: the refit's recovered change is booked as condensation" was deleted at
    # Stage 4 of the rho_t retirement: it pinned the `:slot`-anchor form (the vapor partner
    # exactly minus the recovered cloud change, folded into etd_dv by `_refit_partner_fold!`),
    # which went with slot 3. The derived form books the AFFINE DEFECT, energy only, no nu
    # written (`_refit_partner_defect!`; test_water_anchor.jl, "3c E2").

    @testset "refit partner: inert at t = 1, under :none, and tiny on a smooth cloud" begin
        mktempdir() do tmp
            # t = 1: var_np1 holds no fitted state; the arms are bitwise one step in
            st = Dict{Bool,Matrix{Float64}}()
            for on in (true, false)
                m, p, mo = ep_tile(tmp, "t1_$(on)"; seed! = seed_sheet!,
                                   opts_extra = Dict{Symbol,Any}(:refit_partner => on))
                advance_all!(m, p, mo; t = 1)
                st[on] = copy(m.var_np1)
            end
            @test st[true] == st[false]
            # :none: no transform, no partner, bitwise over a refit
            m1, _, _, _, _ = refit_two_steps(tmp, "none_on"; tr = :none,
                seed! = (p, x, V) -> seed_sheet!(p, x, V; tr = :none))
            m0, _, _, _, _ = refit_two_steps(tmp, "none_off"; tr = :none, on = false,
                seed! = (p, x, V) -> seed_sheet!(p, x, V; tr = :none))
            @test m1.var_np1 == m0.var_np1
            # a smooth, well-resolved cloud: the refit barely moves its recovered density
            smooth!(patch, pts, V) = begin
                c = Scythe.mc_slot(V, "rho_c")
                for i in 1:size(patch.physical, 1)
                    rcv = 2.0e-4 * exp(-((pts[i, end] - 4000.0) / 2500.0)^2)
                    patch.physical[i, c, 1] = Scythe.condensate_slot(rcv, 0.0, :bhyp, MU)
                end
            end
            m1, p1, mo, xpre, xpost = refit_two_steps(tmp, "sm_on"; seed! = smooth!,
                                                      rh = z -> 1.0)
            m0, _, _, _, _ = refit_two_steps(tmp, "sm_off"; seed! = smooth!, rh = z -> 1.0,
                                             on = false)
            kDim = mo.grid_params.kDim; nc = ncols(p1, mo)
            col = ((nc - 1) * kDim + 1):(nc * kDim)
            S1 = scratch(m1); S0 = scratch(m0)
            # the derived partner writes no water (Stage 4): the ETD's vapor increment is the
            # partner-off arm's, bitwise
            @test S1.etd_dv[1:kDim] == S0.etd_dv[1:kDim]
            # the refit's filter still moves a resolved cloud a little every step, and the
            # partner books the affine defect of that change: energy only, and a small fraction
            # (<= 1e-3) of the latent heat lambda_l * drho the full recovered change would book
            dps = S1.etd_d1[1:kDim] .- S0.etd_d1[1:kDim]
            full = Scythe.refit_recovered_change.(xpost[col, 9], xpre[col, 9], :bhyp, MU)
            lam = (S1.R_m[1:kDim] ./ S1.C_vt[1:kDim]) .*
                  (S1.Lv[1:kDim] .- Scythe.Cpv .* S1.Tk[1:kDim])
            @test maximum(abs.(full)) > 0.0
            @test 0.0 < maximum(abs.(dps)) <= 1.0e-3 * maximum(abs.(lam .* full))
        end
    end

    @testset "refit partner: allocation-free acting, config round trip" begin
        mktempdir() do tmp
            m, p, mo = ep_tile(tmp, "rf_alloc"; seed! = seed_sheet!)
            kDim = mo.grid_params.kDim
            advance_all!(m, p, mo; t = 1)
            Scythe.calcTendency(m); gridTransform!(p)
            Scythe.moist_compressible_XZ(m, 1, kDim, 2)
            Scythe.moist_compressible_XZ(m, 1, kDim, 2)
            @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 2)) == 0
            @test Scythe.REFIT_PARTNER_DEFAULT === true
            @test haskey(Scythe.KNOWN_OPTION_KEYS, :refit_partner)
            @test haskey(Scythe.KNOWN_OPTION_KEYS, :exchange_realization)
        end
    end

    # ── 3b. the refit partner under nesting ──────────────────────────────────────────
    # The exchange a nest patch receives between steps sits between the partner's two
    # states. `refit_reference_after_exchange!` (src/nesting.jl) is called by
    # `advance_nested_timestep` right after the collar injection; these tests make the same
    # call in the same place of the sequence (refit, exchange, call, step) on one tile.
    advance_last!(m, p, mo, clast; t) =
        (for c in 1:ncols(p, mo); c == clast || Scythe.advance_column(m, c, t); end;
         Scythe.advance_column(m, clast, t))
    no_inj = Tuple{Vector{Int},Array{Float64,3}}[]

    @testset "refit partner under nesting: a parent's injected collar rows book nothing" begin
        mktempdir() do tmp
            arms = Dict{String,Any}()
            for (tag, on, fix) in (("on", true, true), ("off", false, true),
                                   ("nofix", true, false))
                m, p, mo = ep_tile(tmp, "inj_$(tag)"; seed! = seed_sheet!,
                                   opts_extra = Dict{Symbol,Any}(:refit_partner => on))
                kDim = mo.grid_params.kDim; nc = ncols(p, mo)
                col = ((nc - 1) * kDim + 1):(nc * kDim)
                advance_all!(m, p, mo; t = 1)
                xpre = copy(m.var_np1)
                Scythe.calcTendency(m); gridTransform!(p)
                xpost = copy(p.physical[:, :, 1])
                # the collar: the upper half of the last column takes the child's state, a
                # cloud of its own (ν raised by 3e-7; the derived ρ_t moves with it)
                rows = collect(col[(div(kDim, 2) + 1):end])
                vals = copy(p.physical[rows, :, :])
                vals[:, 9, 1] .+= 3.0e-7
                p.physical[rows, :, :] .= vals
                inj = [(rows, vals)]
                fix && Scythe.refit_reference_after_exchange!(m, inj, 2)
                fix && on && @test m.var_np1[rows, 9] == vals[:, 9, 1]
                fix && !on && @test m.var_np1 == xpre          # partner off: nothing written
                advance_all!(m, p, mo; t = 2)
                # the derived partner is ENERGY ONLY (Stage 4): its booking is the p increment
                arms[tag] = (; d1 = copy(scratch(m).etd_d1[1:kDim]),
                             dv = copy(scratch(m).etd_dv[1:kDim]), xpre, xpost, kDim, col)
            end
            kDim = arms["on"].kDim; col = arms["on"].col
            on = arms["on"].d1 .- arms["off"].d1
            nofix = arms["nofix"].d1 .- arms["off"].d1
            cz = (div(kDim, 2) + 1):kDim; own = 1:div(kDim, 2)
            # no arm writes water (the derived partner books energy only)
            @test arms["on"].dv == arms["off"].dv == arms["nofix"].dv
            # the injected rows: the exchange is not booked, to the bit
            @test all(on[cz] .== 0.0)
            # ...where the unfixed partner books the child-parent cloud difference (the affine
            # defect of the 3e-7 nu step, which crosses the knee mu = 1e-7)
            @test maximum(abs.(nofix[cz])) > 0.0
            # the rest of the column: the refit's own booking, untouched by the exchange
            rc(x) = Scythe.recover_rho_c.(x[col, 9], 0.0, :bhyp, MU)
            drho = rc(arms["on"].xpost) .- rc(arms["on"].xpre)
            @test maximum(abs.(drho[own])) > 1.0e-7
            @test maximum(abs.(on[own])) > 0.0
            @test on[own] == nofix[own]
        end
    end

    @testset "refit partner under nesting: a child's R3X coupling zone books nothing" begin
        mktempdir() do tmp
            NC = 8
            arms = Dict{String,Any}()
            for (tag, on, fix, clast) in (("on1", true, true, 1), ("off1", false, true, 1),
                                          ("nofix1", true, false, 1),
                                          ("onN", true, true, 3 * NC),
                                          ("offN", false, true, 3 * NC),
                                          ("nofixN", true, false, 3 * NC))
                # A CHILD patch: `build_nest`'s FixedBC() (R3X, the parent's trio) on its
                # left, the parent's donated ahat zero -- a parent with no cloud there
                m, p, mo = ep_tile(tmp, "r3x_$(tag)"; seed! = seed_sheet!, ncell = NC,
                                   bcl = Springsteel.FixedBC(), bcr = NeumannBC(),
                                   opts_extra = Dict{Symbol,Any}(:refit_partner => on))
                kDim = mo.grid_params.kDim
                advance_all!(m, p, mo; t = 1)
                # the child's own cloud in its first two cells grows past the parent's:
                # the R3X refit pins it back, the exchange inside the refit
                x = m.tilepoints[:, 1]
                dx = (mo.grid_params.iMax - mo.grid_params.iMin) / NC
                bump = x .< 2.0 * dx
                m.var_np1[bump, 9] .+= 5.0e-7
                xpre = copy(m.var_np1)
                Scythe.calcTendency(m); gridTransform!(p)
                xpost = copy(p.physical[:, :, 1])
                fix && Scythe.refit_reference_after_exchange!(m, no_inj, 2)
                advance_last!(m, p, mo, clast; t = 2)
                col = ((clast - 1) * kDim + 1):(clast * kDim)
                # the derived partner is ENERGY ONLY (Stage 4): its booking is the p increment
                arms[tag] = (; dv = copy(scratch(m).etd_d1[1:kDim]), xpre, xpost, col,
                             zone = copy(x .< Scythe.NEST_EXCHANGE_ZONE_CELLS * dx))
            end
            d(a, b) = arms[a].dv .- arms[b].dv
            rc(a) = Scythe.recover_rho_c.(arms[a].xpost[arms[a].col, 9], 0.0, :bhyp, MU) .-
                    Scythe.recover_rho_c.(arms[a].xpre[arms[a].col, 9], 0.0, :bhyp, MU)
            zone = arms["on1"].zone
            @test all(zone[arms["on1"].col]) && !any(zone[arms["onN"].col])
            @test count(zone) == 3 * 3 * size(arms["on1"].xpre, 1) ÷ (3 * NC)
            # in the coupling zone: nothing booked, to the bit -- where the unfixed partner
            # books the pinned-back cloud's affine defect (the `:slot` form's "the full
            # recovered change as evaporation" amplitude went with slot 3; the derived form's
            # amplitude is test_water_anchor.jl's "3c E2")
            @test all(d("on1", "off1") .== 0.0)
            @test maximum(abs.(rc("on1"))) > 1.0e-7
            @test maximum(abs.(d("nofix1", "off1"))) > 0.0
            # far from it (the right wall): the refit's own booking, as on a single patch
            @test maximum(abs.(rc("onN"))) > 1.0e-7
            @test maximum(abs.(d("onN", "offN"))) > 0.0
            @test d("onN", "offN") == d("nofixN", "offN")
        end
    end

    @testset "refit partner under nesting: inert where it must be, a resting child exact" begin
        mktempdir() do tmp
            inj = [(collect(1:5), zeros(5, 1, 1))]
            # t = 1 and a periodic single patch: nothing is written
            m, p, mo = ep_tile(tmp, "inert"; seed! = seed_sheet!)
            advance_all!(m, p, mo; t = 1)
            x0 = copy(m.var_np1)
            Scythe.refit_reference_after_exchange!(m, no_inj, 2)
            @test m.var_np1 == x0
            p.physical[1:5, 9, 1] .+= 1.0
            Scythe.refit_reference_after_exchange!(m, inj, 1)
            @test m.var_np1 == x0
            # :none transforms: no transformed slot, nothing to re-take
            mN, pN, moN = ep_tile(tmp, "inert_none"; tr = :none,
                                  seed! = (p, x, V) -> seed_sheet!(p, x, V; tr = :none))
            advance_all!(mN, pN, moN; t = 1)
            xN = copy(mN.var_np1)
            pN.physical[1:5, 9, 1] .+= 1.0
            Scythe.refit_reference_after_exchange!(mN, inj, 2)
            @test mN.var_np1 == xN
            # a RESTING child (R3X left, ahat zero): a discrete fixed point with the
            # re-take in the loop, and bitwise the partner-off run
            st = Dict{Bool,Matrix{Float64}}()
            for on in (true, false)
                mr, pr, mor = ep_tile(tmp, "rest_$(on)"; ncell = 8,
                                      bcl = Springsteel.FixedBC(), bcr = NeumannBC(),
                                      opts_extra = Dict{Symbol,Any}(:refit_partner => on))
                for t in 1:3
                    Scythe.refit_reference_after_exchange!(mr, no_inj, t)
                    advance_all!(mr, pr, mor; t)
                    Scythe.calcTendency(mr); gridTransform!(pr)
                end
                st[on] = copy(mr.var_np1)
            end
            @test st[true] == st[false]
            @test all(st[true][:, 8] .== 0.0) && all(st[true][:, 9] .== 0.0)
        end
    end

    # ── 4. (retired) the vapor nudge's pressure partner ──────────────────────────────
    # "vapor nudge partner: R unchanged by the nudge, bitwise when off" was deleted at Stage 4
    # of the rho_t retirement: the vapor reconciliation nudge (`rho_v_reconcile`, VREC) and its
    # partner (`options[:vapor_nudge_partner]`) went with slot 3, whose residual res_rho_t they
    # relaxed the vapor toward.

    # ── 4b. the partners under the VAPOR CONTROL VARIABLE (options[:vapor_transform]) ──
    # Stage 1 of the rho_t retirement: the vapor slot holds ν_v' = bhyp(ρ_v) − bhyp(ρ̄_v). R
    # (read with the RECOVERED vapor) must be held across the ETD exchange to the tolerances the
    # density-vapor tests above hold it to. (The refit partner's and the E6 nudge's vapor-CV
    # tests pinned the `:slot` forms and went with slot 3 at Stage 4.)
    VT = Dict{Symbol,Any}(:vapor_transform => :bhyp)
    VMU = 1.0e-9
    """`seed_super!` with the vapor slot written as the control-variable deviation."""
    function seed_super_v!(patch, pts, V)
        seed_sheet!(patch, pts, V)
        rv_i = Scythe.mc_slot(V, "rho_v")
        for i in 1:size(patch.physical, 1)
            zz = pts[i, end]
            zz > 1500.0 || continue
            Tk = 300.0 - 0.0065 * zz
            pp = 100000.0 * exp(-zz / 8000.0)
            rvb = 0.7 * Scythe.rho_v_sat(Tk, pp / 100.0)
            drv = (1.015 - 0.7) * Scythe.rho_v_sat(Tk, pp / 100.0)
            patch.physical[i, rv_i, 1] += Scythe.bhyp(rvb + drv, VMU) - Scythe.bhyp(rvb, VMU)
            patch.physical[i, V["Q_ss"], 1] += drv
        end
    end

    @testset "vapor control variable: the ETD exchange on a clipped lobe holds R" begin
        mktempdir() do tmp
            m, p, mo = ep_tile(tmp, "v_lobe"; seed! = seed_super_v!, opts_extra = VT)
            advance_all!(m, p, mo)
            kDim = mo.grid_params.kDim; nc = ncols(p, mo)
            X0 = copy(m.var_np1)
            X1, col = etd_stage(m, X0, kDim, nc)
            R0, rl0, rv0, W0, s0 = residual(m, X0)
            R1, rl1, rv1, W1, s1 = residual(m, X1)
            S = scratch(m)
            # the requested exchange in DENSITY: the step-mean condensation (the derived
            # anchor's etd_dv is its nu-space closure, Stage 4)
            req = S.Qdot_bar[1:kDim] .* mo.ts
            dR = (R1 .- R0)[col]; drl = (rl1 .- rl0)[col]; dW = (W1 .- W0)[col]
            Lv = S.Lv[1:kDim]
            lobe = (X0[col, 9] .+ S.etd_d9[1:kDim] .< 0.0) .& (S.etd_d9[1:kDim] .> 0.0)
            @test count(lobe) > 0
            @test all(drl[lobe] .== 0.0)
            act = S.etd_dv[1:kDim] .!= 0.0
            @test count(act) > count(lobe)
            # rho_w = sum alpha nu is invariant through the ETD stage (the vapor is the closure;
            # the `:slot` form's "no recovered vapor moves at a lobe point" went with slot 3)
            @test w_invariant(dW, s0[col], s1[col])
            # R: second order in the exchanged mass, as with the density vapor
            @test all(abs.(dR[act]) .<= 1.0e-3 .* Lv[act] .* abs.(req[act]))
        end
    end

    # ── 5. (retired) the MYNN water closure ──────────────────────────────────────────
    # "MYNN water closure: one step leaves the partition and R closed" was deleted at Stage 4:
    # `options[:mynn_water_closure]` closed the MYNN vapor leg against slot 3's total-water leg,
    # and both went with the slot (test_water_anchor.jl, "3c E7", is the derived MYNN test).
end
