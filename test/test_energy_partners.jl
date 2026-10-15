# test/test_energy_partners.jl -- energy-consistent water-partition bookkeeping under the EOS
# temperature. With `T = p/(ρ_d R_d + ρ_v R_v)` the prognostic `E_t` no longer defines T, and
# every operator that moves a transformed condensate's RECOVERED density by a different amount
# than its partners assume opens the residual `R = E_t − E_implied(p, ρ_d, ρ_v, ρ_liq, ρ_ice,
# T_eos)` (the closed per-operator budget of 2026-10-16). The cures, each a switch:
#
#   1. `options[:exchange_realization]` — the ETD phase change books its vapor, heat and Q_ss
#      partners on the mass the slot's recovered density actually receives;
#   2. `options[:mynn_exchange_realization]` — the MYNN eddy legs of the transformed mass slots
#      book the mass a clipped lobe cannot hold as a phase change;
#   3. `options[:refit_partner]` — the spline refit's change in recovered condensate is booked
#      as condensation (the author's decision, 2026-10-16);
#
# Every assertion is on ONE step of tiles advanced from the same state, so the property tested
# is the operator's own. The tile is the lobe-relaxation fixture: a periodic XZ slice with
# 500 m vertical cells (the RCE's dz), horizontally uniform, whose kinked boundary-layer cloud
# sheet the refit rings negative in the air above.

using Test
using Scythe
using Springsteel
using SparseArrays
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
                     precip = false, tr = :bhyp, ts = 0.5, rh = z -> 0.7)
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :precipitation => precip, :vertical_mixing => false,
                                :condensate_transform => tr, :rain_transform => tr)
        merge!(opts, opts_extra)
        varnames = Scythe.mc_var_names(opts)
        vars = Dict(v => i for (i, v) in enumerate(varnames))
        side_bc = Dict{String,Any}(v => PeriodicBC() for v in varnames)
        wall_bc = merge(Dict{String,Any}(v => NeumannBC() for v in varnames),
                        Dict{String,Any}("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = SPAN, num_cells_i = NCELL,
            kMin = 0.0, kMax = ZTOP, num_cells_k = NCELL_K,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
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
            patch.physical[i, V["rho_t"], 1] = rc
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
            patch.physical[i, V["rho_t"], 1] += drv
            patch.physical[i, V["Q_ss"], 1] += drv
        end
    end

    """A healthy cloud: `amp` at every level (ν ≫ μ, J ≈ ½), in air of RH `rh`."""
    function seed_uniform!(patch, pts, V; amp = 1.0e-3)
        rc_i = Scythe.mc_slot(V, "rho_c")
        for i in 1:size(patch.physical, 1)
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(amp, 0.0, :bhyp, MU)
            patch.physical[i, V["rho_t"], 1] = amp
        end
    end

    # ── the residual R = E_t − E_implied(T_eos), pointwise, of a slot matrix ──────────
    """`R` and the recovered liquid at every gridpoint of the slot matrix `X` (the layout of
    `var_np1`: perturbation slots, slot 9 against `ν̄_c ≡ 0`, slot 8 a total)."""
    function residual(m, X; tr = :bhyp)
        mo = m.model; V = mo.grid_params.vars; kDim = mo.grid_params.kDim
        rs = m.ref_state
        pbar = Scythe.ref_pressure(rs)[:, 1]; rdbar = Scythe.ref_rho_d(rs)[:, 1]
        rtbar = Scythe.ref_rho_t(rs)[:, 1]; Ebar = Scythe.ref_total_energy(rs)[:, 1]
        rvbar = m.mc_ref_diag.rho_vbar
        z = m.tilepoints[:, end]
        rv_i = Scythe.mc_slot(V, "rho_v")
        n = size(X, 1)
        R = zeros(n); rl = zeros(n); rv = zeros(n)
        for g in 1:n
            k = mod1(g, kDim)
            p = X[g, 1] + pbar[k]; rd = X[g, 2] + rdbar[k]; rt = X[g, 3] + rtbar[k]
            u = X[g, 4]; w = X[g, 5]; Et = X[g, 6] + Ebar[k]; rv[g] = X[g, rv_i] + rvbar[k]
            rl[g] = Scythe.recover_rho_c(X[g, 9], 0.0, tr, MU) +
                    Scythe.recover_rho_r(X[g, 8], tr, MU)
            geo = 0.5 * (u * u + w * w) + Scythe.gravity * z[g]
            T = Scythe.eos_temperature(p, rd, rv[g])
            R[g] = Et - Scythe.implied_total_energy(T, rd, rv[g], rl[g], 0.0, rt, geo)
        end
        return R, rl, rv
    end

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

    @testset "a clipped lobe: vapor and heat move by exactly the recovered change" begin
        mktempdir() do tmp
            res = Dict{Bool,Any}()
            for on in (true, false)
                m, p, mo = ep_tile(tmp, "lobe_$(on)"; seed! = seed_super!,
                                   opts_extra = Dict{Symbol,Any}(:exchange_realization => on))
                advance_all!(m, p, mo)
                kDim = mo.grid_params.kDim; nc = ncols(p, mo)
                X0 = copy(m.var_np1)
                X1, col = etd_stage(m, X0, kDim, nc)
                R0, rl0, rv0 = residual(m, X0)
                R1, rl1, rv1 = residual(m, X1)
                S = scratch(m)
                rv_i = Scythe.mc_slot(mo.grid_params.vars, "rho_v")
                res[on] = (; dR = (R1 .- R0)[col], drl = (rl1 .- rl0)[col],
                           drv = (X1 .- X0)[col, rv_i], req = S.etd_dv[1:kDim],
                           z = m.tilepoints[col, end], nu = X0[col, 9],
                           Lv = copy(S.Lv), d9 = copy(S.etd_d9))
            end
            on = res[true]; off = res[false]
            # the requested exchange is the same in both arms (one step, one state)
            @test on.req == off.req
            # lobe points: the slot is below zero and stays there -- condensation that never
            # appears in the recovered density
            lobe = (on.nu .+ on.d9 .< 0.0) .& (on.d9 .> 0.0)
            @test count(lobe) > 0
            @test all(on.drl[lobe] .== 0.0)
            # OLD booking: vapor and heat for the full request -> R jumps by ~L*request
            @test off.drv[lobe] ≈ off.req[lobe] rtol = 1.0e-9
            @test all(abs.(off.dR[lobe]) .> 0.5 .* off.Lv[lobe] .* abs.(off.req[lobe]))
            # NEW: the exchange is closed on the recovered mass -- no vapor moves at a lobe
            # point, and the ETD leaves R unchanged to rounding of the exchanged energy
            @test all(abs.(on.drv[lobe]) .<= 1.0e-12 .* abs.(on.req[lobe]))
            act = on.req .!= 0.0
            @test count(act) > count(lobe)
            # water: vapor + recovered liquid unchanged by the ETD at every active point
            @test all(abs.(on.drv[act] .+ on.drl[act]) .<= 1.0e-9 .* abs.(on.req[act]) .+ 1.0e-18)
            # R: what is left is second order in the exchanged mass (the heat capacity the
            # exchange itself changes, at the frozen ETD coefficients) -- measured 2e-4 of
            # L*|request| where the step evaporates a thin sheet outright, 3000x below the
            # off arm's opening there
            @test all(abs.(on.dR[act]) .<= 1.0e-3 .* on.Lv[act] .* abs.(on.req[act]))
            @test maximum(abs.(on.dR[act])) < 1.0e-3 * maximum(abs.(off.dR[act]))
            # the off arm's water is open by exactly the clipped request
            @test maximum(abs.(off.drv[lobe] .+ off.drl[lobe])) > 0.0
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
            # the two bookings differ, by the Jacobian's linearization error and no more
            @test on.drv != off.drv
            @test maximum(abs.(on.drv[act] .- off.drv[act]) ./ abs.(off.drv[act])) < 1.0e-6
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

    # ── 2. the MYNN mass legs on the recovered density ───────────────────────────────
    @testset "_mynn_clip_gap: zero off the lobe, the unrealized rate in it" begin
        mu = MU; ts = 0.5
        # healthy at both ends: an exact zero, whatever the rate
        @test Scythe._mynn_clip_gap(1.0e-5, 2.0e-6, 4.0e-6, ts, :bhyp, mu) === 0.0
        @test Scythe._mynn_clip_gap(0.0, 0.0, 0.0, ts, :bhyp, mu) === 0.0
        # inside the lobe and staying there: nothing is realized, the whole rate is the gap
        @test Scythe._mynn_clip_gap(-1.0e-6, 2.0e-7, 4.0e-7, ts, :bhyp, mu) == -4.0e-7
        # crossing zero upward: only the part above zero is realized
        g = Scythe._mynn_clip_gap(-1.0e-7, 1.0e-6, 2.0e-6, ts, :bhyp, mu)
        @test g ≈ (Scythe.ahyp(-1.0e-7 + 0.5e-6, mu) / ts) - 2.0e-6 rtol = 1e-14
        @test -2.0e-6 < g < 0.0
        # draining a thin slot through zero: the recovered loss is what was there
        g = Scythe._mynn_clip_gap(1.0e-7, -1.0e-6, -2.0e-6, ts, :bhyp, mu)
        @test g ≈ (-Scythe.ahyp(1.0e-7, mu) / ts) + 2.0e-6 rtol = 1e-14
    end

    @testset "MYNN cloud leg into a lobe: the exchange closes on the recovered density" begin
        isdefined(Main, :MYNNTestFixtures) || include("mynn_test_fixtures.jl")
        F = Main.MYNNTestFixtures
        mktempdir() do tmp
            # A boundary-layer cloud (positive below ~700 m) under a negative-ν lobe layer
            # (900-1600 m) inside the mixed layer: K_h mixes cloud up into the lobe, where the
            # recovered density is 0 and stays 0.
            function seed!(patch, mo, kDim, z, col; lobe = true)
                V = mo.grid_params.vars
                c = Scythe.mc_slot(V, "rho_c")
                u = F.bl_jet(z, 12.0, 500.0)
                rho_t = col.rho_d .+ col.rho_v
                for j in 1:size(patch.physical, 1)
                    k = mod1(j, kDim); zz = z[k]
                    patch.physical[j, V["u"], 1] = u[k]
                    haskey(V, "rho_e") &&
                        (patch.physical[j, V["rho_e"], 1] = rho_t[k] * 0.4 * exp(-zz / 1500.0))
                    bump = 2.0e-3 * exp(-((zz - 800.0) / 600.0)^2)
                    rc = lobe ? 4.0e-4 * exp(-((zz - 300.0) / 300.0)^2) : 0.0
                    patch.physical[j, V["rho_v"], 1] = rho_t[k] * bump
                    patch.physical[j, V["rho_t"], 1] = rho_t[k] * bump + rc
                    nu = Scythe.condensate_slot(rc, 0.0, :bhyp, MU)
                    lobe && (nu -= 2.0e-6 * exp(-((zz - 1250.0) / 300.0)^2))
                    patch.physical[j, c, 1] = nu
                end
                spectralTransform!(patch)
                gridTransform!(patch)
            end
            arms = Dict{Any,Any}()
            for (tag, mynn, on, lobe) in (("on", true, true, true), ("off", true, false, true),
                                          ("ctl", false, true, true),
                                          ("nl_on", true, true, false),
                                          ("nl_off", true, false, false))
                m, p, mo, gp, z, col = F.make_mynn_tile(tmp; mynn = mynn, init = :zero,
                                                        num_cells_k = 25, ctrans = :bhyp,
                                                        rtrans = :bhyp)
                mo.options[:mynn_exchange_realization] = on
                seed!(p, mo, gp.kDim, z, col; lobe)
                F.step_all!(m, p, gp.kDim)
                arms[tag] = (; m, p, mo, gp, x = copy(p.physical[:, :, 1]))
            end
            mo = arms["on"].mo; V = mo.grid_params.vars; kDim = arms["on"].gp.kDim
            rv_i = V["rho_v"]; c = Scythe.mc_slot(V, "rho_c")
            nf = size(arms["ctl"].m.expdot_n, 2)
            Don = arms["on"].m.expdot_n[:, 1:nf] .- arms["ctl"].m.expdot_n
            Doff = arms["off"].m.expdot_n[:, 1:nf] .- arms["ctl"].m.expdot_n
            x = arms["on"].x
            @test x == arms["off"].x
            lobe = x[:, c] .< 0.0
            @test count(lobe) > 0
            # the cloud leg itself is untouched; so are rho_t and E_t (the flux form)
            for v in (3, 6, 9)
                @test Don[:, v] == Doff[:, v]
            end
            # at the lobe points the slot moves and the recovered density cannot
            ts = mo.ts
            ch = lobe .& (Don[:, 9] .!= 0.0) .& ((x[:, 9] .+ ts .* Don[:, 9]) .< 0.0)
            @test count(ch) > 0
            # OFF: the mixed-in cloud vanishes; ON: it is vapor, mass for mass
            J = arms["on"].m.mc_scratch[Threads.threadid()].Jc     # (any column: uniform in x)
            dens = Don[ch, 9] ./ J[mod1.(findall(ch), kDim)]
            @test Don[ch, rv_i] .- Doff[ch, rv_i] ≈ dens rtol = 1.0e-10
            # p and Q_ss carry the evaporation's partners, of the right sign
            @test all(sign.(Don[ch, 1] .- Doff[ch, 1]) .== -sign.(dens))
            # bitwise where the clip never acts: the same boundary layer with no condensate
            # (on 1 km cells even a smooth cloud rings below zero somewhere, so the clean
            # comparison is the cloud-free one; the masked one follows)
            @test arms["nl_on"].m.expdot_n == arms["nl_off"].m.expdot_n
            # and away from the lobe points in the lobed state
            far = .!(lobe .| ((x[:, 9] .+ ts .* Don[:, 9]) .< 0.0))
            @test Don[far, :] == Doff[far, :]
            # allocation-free with the partner acting
            m = arms["on"].m
            Scythe.moist_compressible_XZ(m, 1, kDim, 2)
            @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 2)) == 0
        end
    end

    # ── 3. the refit partner ─────────────────────────────────────────────────────────
    """Step 1, the model's own refit (`calcTendency` + `gridTransform!`), step 2: returns the
    pre-refit slots (var_np1 after step 1), the post-refit slots, and the tile."""
    function refit_two_steps(tmp, tag; on = true, kw...)
        m, p, mo = ep_tile(tmp, tag; opts_extra = Dict{Symbol,Any}(:refit_partner => on), kw...)
        advance_all!(m, p, mo; t = 1)
        xpre = copy(m.var_np1)
        Scythe.calcTendency(m)
        gridTransform!(p)
        xpost = copy(p.physical[:, :, 1])
        advance_all!(m, p, mo; t = 2)
        return m, p, mo, xpre, xpost
    end

    @testset "refit partner: the refit's recovered change is booked as condensation" begin
        mktempdir() do tmp
            m1, p1, mo, xpre, xpost = refit_two_steps(tmp, "rf_on"; seed! = seed_sheet!)
            m0, p0, _, xpre0, xpost0 = refit_two_steps(tmp, "rf_off"; seed! = seed_sheet!,
                                                       on = false)
            @test xpre == xpre0 && xpost == xpost0          # step 1 and the refit: identical
            V = mo.grid_params.vars; kDim = mo.grid_params.kDim; nc = ncols(p1, mo)
            rv_i = Scythe.mc_slot(V, "rho_v")
            col = ((nc - 1) * kDim + 1):(nc * kDim)          # the scratch is the last column's
            S1 = scratch(m1); S0 = scratch(m0)
            dp = S1.etd_d1[1:kDim] .- S0.etd_d1[1:kDim]
            dvv = S1.etd_dv[1:kDim] .- S0.etd_dv[1:kDim]
            # the refit's change in recovered cloud, computed here independently
            rc(x) = Scythe.recover_rho_c.(x[col, 9], 0.0, :bhyp, MU)
            drho = rc(xpost) .- rc(xpre)
            @test maximum(abs.(drho)) > 1.0e-7               # the sheet's refit does move it
            # water: the vapor partner is exactly minus the recovered change
            @test dvv ≈ -drho rtol = 1.0e-12 atol = 1.0e-22
            # energy: on the pre-refit state, replacing the cloud slot by its refit value opens
            # R by ~L*drho; the partner closes it to second order in drho
            Rpre, _, _ = residual(m1, xpre)
            xc = copy(xpre); xc[col, 9] .= xpost[col, 9]
            Rc, _, _ = residual(m1, xc)
            xp = copy(xc); xp[col, 1] .+= dp; xp[col, rv_i] .+= dvv
            Rp, _, _ = residual(m1, xp)
            open_ = abs.(Rc[col] .- Rpre[col]); closed = abs.(Rp[col] .- Rpre[col])
            big = abs.(drho) .> 1.0e-3 * maximum(abs.(drho))
            @test all(open_[big] .> 1.0e6 .* abs.(drho[big]))      # ~L per kg, unpartnered
            @test all(closed[big] .<= 1.0e-3 .* open_[big])
            @test sum(closed) <= 1.0e-3 * sum(open_)
            # the Q_ss give-back rides with it: (1 + Q_s) per kg, of the vapor's sign
            S = scratch(m1)
            dq = S1.etd_qnp1[1:kDim] .- S0.etd_qnp1[1:kDim]
            lam0 = S.etd_lam[1:kDim] .== 0.0
            @test all((sign.(dq[big]) .== sign.(dvv[big])) .| lam0[big])
            # nothing but p, the vapor and Q_ss moved in the ETD scratch
            @test S1.etd_d9[1:kDim] == S0.etd_d9[1:kDim]
            @test S1.etd_d8[1:kDim] == S0.etd_d8[1:kDim]
        end
    end

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
                    patch.physical[i, V["rho_t"], 1] = rcv
                end
            end
            m1, p1, mo, xpre, xpost = refit_two_steps(tmp, "sm_on"; seed! = smooth!,
                                                      rh = z -> 1.0)
            m0, _, _, _, _ = refit_two_steps(tmp, "sm_off"; seed! = smooth!, rh = z -> 1.0,
                                             on = false)
            kDim = mo.grid_params.kDim
            dvs = scratch(m1).etd_dv[1:kDim] .- scratch(m0).etd_dv[1:kDim]
            # the refit's filter still moves a resolved cloud a little every step (the partner
            # books that too: the decision is per recovered kilogram, not per lobe), but by a
            # small fraction of the cloud: measured ~1e-3 of the 2e-4 kg/m3 peak
            @test 0.0 < maximum(abs.(dvs)) < 1.0e-2 * 2.0e-4
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
            @test haskey(Scythe.KNOWN_OPTION_KEYS, :mynn_exchange_realization)
        end
    end
end
