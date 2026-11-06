# test/test_energy_budget.jl -- the per-operator budget of R = E_t - E_implied and of the
# partition gap X = rho_t - rho_d - rho_v - rho_liq - rho_ice (`options[:energy_budget]`,
# src/energy_budget.jl), and the energy census rows (MC_REFIT_DR, MC_REFIT_BOOKED, MC_LOBE_*,
# MC_XGAP_*) that are measured on every run.
#
# What has to hold:
#   1. CLOSURE: the stages are finite differences of the column integral over the increment
#      each one wrote, in the order the column step applies them, so their sum IS the actual
#      change between two snapshots, to rounding (the instrument's own test), and the
#      sub-rows sum to their stage;
#   2. the budget is a DIAGNOSTIC: off, absent or on, the state is bitwise the code before the
#      budget existed (a golden capture at 5f3d8fe);
#   3. the census rows are exact zeros at rest, and the partner/lobe/partition rows are exact
#      zeros on a dry column;
#   4. the isolated operators carry what they should: the surface exchange and the momentum
#      diffusion are energy-consistent (their rows are second order), the refit partner's row
#      is an exact zero with the partner off, Stage L's rows are zeros with the device off;
#   5. the hooks allocate nothing;
#   6. the key is registered and accepted from a TOML [extra.options] block, the CSV is
#      written with its header, and exact_si is refused.
#
# The fixture is test_eos_temperature.jl's (the Stage L one): a periodic XZ slice with 500 m
# vertical cells, every water slot under :bhyp, over a hydrostatic reference, with the kinked
# cloud sheet the refit rings below zero.

using Test
using Scythe
using Springsteel
using SparseArrays
using Springsteel: PeriodicBC, NeumannBC, DirichletBC

isdefined(Main, :VaporTransformFixtures) || include("vapor_transform_fixtures.jl")

@testset "energy budget (options[:energy_budget]) and the energy census" begin
    EB_F = Main.VaporTransformFixtures

    import Springsteel.Thermodynamics: rho_v_sat, L_v, Rd, Rv, Cpv, gravity

    EB_SPAN = 8.0e3
    EB_NCELL = 4
    EB_ZTOP = 10.0e3
    EB_NCELL_K = 20
    EB_MU = 1.0e-7

    "Hydrostatic reference column (test_eos_temperature.jl's `eos_column`)."
    function eb_column(z, rh; q_l = 0.0)
        Tf(zz) = 300.0 - 0.0065 * zz
        function rho_t_of(zz, p)
            T = Tf(zz); rv = rh(zz) * rho_v_sat(T, p / 100.0)
            rd = (p - Rv * T * rv) / (Rd * T)
            return rd * (1.0 + q_l) + rv
        end
        f(zz, p) = -gravity * rho_t_of(zz, p)
        p_Pa = similar(z); zc = 0.0; pc = 100000.0
        for (k, zk) in enumerate(z)
            n = max(1, ceil(Int, (zk - zc) / 10.0)); h = (zk - zc) / n
            for _ in 1:n
                k1 = f(zc, pc); k2 = f(zc + h / 2, pc + h * k1 / 2)
                k3 = f(zc + h / 2, pc + h * k2 / 2); k4 = f(zc + h, pc + h * k3)
                pc += h * (k1 + 2k2 + 2k3 + k4) / 6; zc += h
            end
            p_Pa[k] = pc
        end
        Tk = Tf.(z)
        rho_v = rh.(z) .* rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Rv .* Tk .* rho_v)) ./ (Rd .* Tk)
        return (; Tk, p_Pa, rho_d, rho_v, rho_c = q_l .* rho_d)
    end

    "The periodic XZ tile (test_eos_temperature.jl's `eos_tile`)."
    function eb_tile(tmpdir, tag; seed! = (p, x, V, col) -> nothing, ice = false, tr = :bhyp,
                     ts = 0.5, rh = z -> 0.7, extra = Dict{Symbol,Any}(),
                     pp_extra = Dict{Symbol,Float64}())
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :consistent_qss_reference => true,
                                :hydrostatic_reference => true,
                                :precipitation => ice, :vertical_mixing => false,
                                :condensate_transform => tr, :rain_transform => tr)
        if ice
            opts[:rain_moments] = 2
            opts[:rain_number_transform] = tr
            opts[:ice_microphysics] = :ishmael
            opts[:ice_transform] = tr
        end
        merge!(opts, extra)
        varnames = Scythe.mc_var_names(opts)
        vars = Dict(v => i for (i, v) in enumerate(varnames))
        side_bc = Dict{String,Any}(v => PeriodicBC() for v in varnames)
        wall_bc = merge(Dict{String,Any}(v => NeumannBC() for v in varnames),
                        Dict{String,Any}("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = EB_SPAN, num_cells_i = EB_NCELL,
            kMin = 0.0, kMax = EB_ZTOP, num_cells_k = EB_NCELL_K,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
        ref_file = joinpath(tmpdir, "eb_$(tag).ref")
        pp = merge(Dict{Symbol,Float64}(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                                        :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                        :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3,
                                        :f => 0.0, :N_r => 1.0e-3), pp_extra)
        model = ModelParameters(ts = ts, integration_time = 10.0, output_interval = 10.0,
            equation_set = "moist_compressible_XZ", ref_state_file = ref_file,
            output_dir = joinpath(tmpdir, "out_$(tag)") * "/",
            grid_params = gp, physical_params = pp, options = opts)
        gp = model.grid_params
        patch = createGrid(gp)
        pts = Scythe.getGridpoints(patch)
        z = pts[1:gp.kDim, end]
        col = eb_column(z, rh)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        seed!(patch, pts, gp.vars, col)
        spectralTransform!(patch)
        gridTransform!(patch)
        hrm = sparse(Int64[], Int64[], Float64[], size(patch.spectral, 1),
                     size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, hrm)
        return mtile, patch, model
    end

    eb_ncols(patch, model) = div(size(patch.physical, 1), model.grid_params.kDim)

    "Steps `t0:t0+nsteps-1`: every column, then the model's refit (calcTendency + transform)."
    function eb_run!(m, p, mo, nsteps; t0 = 1)
        for t in t0:(t0 + nsteps - 1)
            for c in 1:eb_ncols(p, mo)
                Scythe.advance_column(m, c, t)
            end
            Scythe.calcTendency(m)
            gridTransform!(p)
        end
    end

    eb_kink(z; zb = 200.0, zt = 1150.0, amp = 1.0e-3) =
        max(0.0, amp * (z - zb) * (zt - z) / ((zt - zb) / 2)^2)

    "The kinked cloud sheet at fixed T (test_eos_temperature.jl's), `wamp` a smooth w."
    function eb_seed_sheet!(patch, pts, V, col; amp = 1.0e-3, wamp = 0.0)
        rc_i = Scythe.mc_slot(V, "rho_c")
        kDim = length(col.Tk)
        for i in 1:size(patch.physical, 1)
            z = pts[i, end]; k = mod1(i, kDim)
            rc = eb_kink(z; amp)
            T = col.Tk[k]
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rc, 0.0, :bhyp, EB_MU)
            # (no rho_t write: the slot was deleted at Stage 4 of the rho_t retirement; the
            # derived total rho_d + sum_k alpha_k nu_k carries the cloud's mass through rc_i)
            patch.physical[i, V["E_t"], 1] = rc * ((Cpv * T) - L_v(T) + (gravity * z))
            patch.physical[i, V["w"], 1] = wamp * sin(2pi * pts[i, 1] / EB_SPAN) *
                                           sin(pi * z / EB_ZTOP)
        end
    end
    sheet!(; wamp = 0.5) = (p, x, V, c) -> eb_seed_sheet!(p, x, V, c; wamp = wamp)

    "Order-sensitive exact checksum of a Float64 array (test_eos_temperature.jl's)."
    eb_checksum(A) = sum(UInt128(reinterpret(UInt64, x)) * UInt128(j)
                         for (j, x) in enumerate(vec(A)))

    ON = Dict{Symbol,Any}(:energy_budget => true)
    rowd(rows) = Dict(r.stage => r for r in rows)
    stage(rows, s) = rowd(rows)[s]

    # The same total as the `total` row, but per column from the raw snapshot: the closure is
    # asserted on the snapshot's own arrays as well as on the assembled rows.
    function closure_check(snap)
        tR = vec(sum(snap.dR[collect(Scythe.EB_TOP), :]; dims = 1))
        tX = vec(sum(snap.dX[collect(Scythe.EB_TOP), :]; dims = 1))
        okR = all(abs.(snap.actualR .- tR) .<= 1.0e-10 .* max.(abs.(snap.actualR), 1.0e-300) .+
                  1.0e-12 * maximum(abs.(snap.dR)))
        okX = all(abs.(snap.actualX .- tX) .<= 1.0e-10 .* max.(abs.(snap.actualX), 1.0e-300) .+
                  1.0e-12 * maximum(abs.(snap.dX)))
        return okR, okX
    end

    # ── 1. closure ───────────────────────────────────────────────────────────────────
    @testset "closure: the stages sum to the actual change, to rounding" begin
        mktempdir() do tmp
            for ice in (false, true)
                m, p, mo = eb_tile(tmp, "close_$(ice)"; ice = ice, extra = ON,
                                   seed! = sheet!())
                @test m.ebudget.active
                @test length(Scythe.EB_STAGES) == Scythe.EB_NSTAGE
                # Two intervals: 6 steps (startup Euler/AB2 then AB3) and 3 more, so the
                # snapshot's reset is exercised and the second interval starts at a mark.
                for (n, t0) in ((6, 1), (3, 7))
                    eb_run!(m, p, mo, n; t0 = t0)
                    snap = Scythe.energy_budget_snapshot!(m)
                    @test snap !== nothing
                    @test all(snap.nsteps .== n)
                    okR, okX = closure_check(snap)
                    @test okR
                    @test okX
                    rows = Scythe.energy_budget_rows([snap])
                    @test length(rows) == Scythe.EB_NSTAGE + 3
                    act = stage(rows, "actual"); clo = stage(rows, "closure")
                    tot = stage(rows, "total")
                    @test abs(act.dR_dt_Wm2) > 1.0              # the column really moved
                    @test abs(clo.dR_dt_Wm2) <= 1.0e-10 * abs(act.dR_dt_Wm2)
                    @test abs(clo.dX_dt_kgm2h) <= 1.0e-10 * abs(act.dX_dt_kgm2h)
                    @test tot.dR_dt_Wm2 ≈ act.dR_dt_Wm2 rtol = 1.0e-10
                    # The sub-rows split their stage: the explicit operators + nonlin, and the
                    # refit partner + the phase change.
                    xs = ["explicit:dyn", "explicit:rad", "explicit:hmom", "explicit:sponge",
                          "explicit:bl", "explicit:sfc", "explicit:hwater", "explicit:wexch",
                          "explicit:nonlin"]
                    # (`explicit:vrec`, the vapor nudge's row, went with the nudge at Stage 4;
                    # `explicit:wexch`, the derived anchor's exchange/sedimentation closures,
                    # is a sub-row of `explicit` too and is live now that the derived anchor is
                    # the only one)
                    se = sum(stage(rows, s).dR_dt_Wm2 for s in xs)
                    @test se ≈ stage(rows, "explicit").dR_dt_Wm2 rtol = 1.0e-9
                    sp = stage(rows, "etd:refit_partner").dR_dt_Wm2 +
                         stage(rows, "etd:phase").dR_dt_Wm2
                    @test sp ≈ stage(rows, "etd").dR_dt_Wm2 rtol = 1.0e-9
                    # The refit and its partner are the large pair on this sheet, and of
                    # opposite sign: the partner books what the refit opened.
                    rf = stage(rows, "refit").dR_dt_Wm2
                    pa = stage(rows, "etd:refit_partner").dR_dt_Wm2
                    @test abs(rf) > 100.0 && sign(pa) == -sign(rf)
                    # Off-device rows are exact zeros on this fixture. (`lobe_relax` is not off
                    # any more: Stage L defaults ON under the derived anchor, the only anchor
                    # since Stage 4, so its row is live here; it is held to an exact zero with
                    # the device off in the "Stage L" and "dry column" testsets below.)
                    for s in ("explicit:rad", "explicit:sponge", "explicit:bl", "explicit:sfc",
                              "explicit:hwater", "hsi_hist", "vdiff", "clamp")
                        @test stage(rows, s).dR_dt_Wm2 == 0.0
                    end
                end
                # Nothing left over: a snapshot with no step in between is empty.
                @test Scythe.energy_budget_snapshot!(m) === nothing
            end
        end
    end

    # ── 2. a diagnostic: bitwise off, absent and on ──────────────────────────────────
    @testset "the budget never moves the state (golden capture, re-seeded at bf577b2)" begin
        # Four full steps of the sheet with a w perturbation under the DEFAULT options (every
        # partner on) and the derived water anchor: every bit of the advanced state and of the
        # refit patch, off, absent and on. First captured at 5f3d8fe, before the budget and the
        # census existed (on the prognostic rho_t slot; in git history); RE-SEEDED at bf577b2
        # under the derived anchor for Stage 4 of the rho_t retirement and summed in the
        # Stage-4 layout (`s4_checksum`, slot 3 left out).
        golden = Dict("warm" => (UInt128(157606634960239587946259219),
                                 UInt128(4167038103528354491640813737)),
                      "ice" => (UInt128(451825243419528058947592109),
                                UInt128(19486616508510399785925558772)))
        got = Dict{String,Any}()
        # (the derived anchor is the only one since Stage 4: no key, the default exercised)
        WA = Dict{Symbol,Any}()
        mktempdir() do tmp
            for (tag, ice) in (("warm", false), ("ice", true))
                for (arm, extra) in (("absent", WA),
                                     ("off", merge(WA, Dict{Symbol,Any}(:energy_budget => false))),
                                     ("on", merge(WA, ON)))
                    m, p, mo = eb_tile(tmp, "gold_$(tag)_$(arm)"; ice = ice, extra = extra,
                                       seed! = sheet!())
                    @test m.ebudget.active == (arm == "on")
                    eb_run!(m, p, mo, 4)
                    V = mo.grid_params.vars
                    got[tag] = (EB_F.s4_checksum(m.var_np1, V), EB_F.s4_checksum(p.physical, V))
                    @test got[tag][1] == golden[tag][1]
                    @test got[tag][2] == golden[tag][2]
                end
            end
        end
        get(ENV, "SCYTHE_PRINT_GOLDENS", "") == "1" &&
            foreach(k -> println("GOLDEN energy_budget $(k) => $(got[k])"), sort(collect(keys(got))))
    end

    # ── 3. the census ─────────────────────────────────────────────────────────────────
    census_rows = (:MC_REFIT_DR, :MC_REFIT_BOOKED, :MC_LOBE_C, :MC_LOBE_R, :MC_LOBE_I1,
                   :MC_LOBE_I2, :MC_LOBE_I3, :MC_XGAP_MAX, :MC_XGAP_INT, :MC_XGAP_POOL)
    crow(m, s) = sum(view(m.mc_water_stats, getfield(Scythe, s), :))

    @testset "census rows: declared, and exact zeros at rest" begin
        @test length(Scythe.MC_WATER_STATS) == Scythe.MC_STATS_LAST
        @test Scythe.MC_REFIT_DR == Scythe.MC_CEVAP_LAST + 1
        mktempdir() do tmp
            for (tag, ice) in (("warm", false), ("ice", true))
                m, p, mo = eb_tile(tmp, "rest_$(tag)"; ice = ice, extra = ON)
                eb_run!(m, p, mo, 5)
                @test maximum(abs.(m.var_np1)) == 0.0            # rest is a fixed point
                # Before the report: the per-step rows have accumulated over 4 refits.
                for s in census_rows
                    @test crow(m, s) === 0.0
                end
                # The report fills the partition rows from the output state: zeros too.
                out_int = round(Int, mo.output_interval / mo.ts)
                for t in 2:(out_int + 1)
                    Scythe.energy_census_trace(m, t)
                end
                for s in census_rows
                    @test crow(m, s) === 0.0
                end
                # ...and every budget row is an exact zero.
                snap = Scythe.energy_budget_snapshot!(m)
                @test all(snap.dR .== 0.0) && all(snap.dX .== 0.0)
                @test all(snap.actualR .== 0.0) && all(snap.actualX .== 0.0)
            end
        end
    end

    @testset "census rows on a dry column: the partner, Stage L and X rows are exact zeros" begin
        mktempdir() do tmp
            for relax in (false, true)
                m, p, mo = eb_tile(tmp, "dry_$(relax)"; rh = z -> 0.0,
                    extra = Dict{Symbol,Any}(:energy_budget => true, :lobe_relaxation => relax),
                    seed! = (p, x, V, c) -> eb_seed_sheet!(p, x, V, c; amp = 0.0, wamp = 0.5))
                eb_run!(m, p, mo, 5)
                @test maximum(abs.(m.var_np1[:, mo.grid_params.vars["w"]])) > 0.1   # it moves
                for s in (:MC_REFIT_BOOKED, :MC_LOBE_C, :MC_LOBE_R, :MC_LOBE_I1,
                          :MC_LOBE_I2, :MC_LOBE_I3)
                    @test crow(m, s) === 0.0
                end
                # The refit of a moving dry state is NOT free: p and E_t are refit too, and
                # the census measures that (it is the dry part of the refit's commutator).
                @test isfinite(crow(m, :MC_REFIT_DR))
                out_int = round(Int, mo.output_interval / mo.ts)
                for t in 2:(out_int + 1)
                    Scythe.energy_census_trace(m, t)
                end
                for s in (:MC_XGAP_MAX, :MC_XGAP_INT, :MC_XGAP_POOL)
                    @test crow(m, s) === 0.0
                end
                snap = Scythe.energy_budget_snapshot!(m)
                # X moves only at rounding: rho_t' and rho_d' are not bitwise equal on a dry
                # column (the explicit and the slaved SI updates round them differently),
                # which the budget sees at 1e-16 of the dry-air column (~1e4 kg/m^2).
                @test maximum(abs.(snap.dX)) < 1.0e-12
                @test all(snap.dR[Scythe.EB_ETD_PARTNER, :] .== 0.0)
                @test all(snap.dR[Scythe.EB_LOBE, :] .== 0.0)
                okR, _ = closure_check(snap)
                @test okR
            end
        end
    end

    @testset "census on the sheet: the refit residual against what the partner booked" begin
        mktempdir() do tmp
            arms = Dict{Bool,Any}()
            for on in (true, false)
                m, p, mo = eb_tile(tmp, "rdr_$(on)";
                    extra = Dict{Symbol,Any}(:refit_partner => on), seed! = sheet!())
                @test !m.ebudget.active          # the census needs no budget
                eb_run!(m, p, mo, 4)
                arms[on] = (; dr = crow(m, :MC_REFIT_DR), bk = crow(m, :MC_REFIT_BOOKED))
            end
            # The refit opens R by +O(L ρ_c) on the sheet; the partner books energy of the
            # same sign and size into p (and none with the partner off).
            @test arms[true].dr > 0.0
            @test arms[true].bk > 0.5 * arms[true].dr
            @test arms[false].bk === 0.0
            @test arms[false].dr > 0.0
        end
    end

    @testset "Stage L: the relaxed-nu census and the budget row" begin
        mktempdir() do tmp
            for relax in (false, true)
                m, p, mo = eb_tile(tmp, "stl_$(relax)";
                    extra = Dict{Symbol,Any}(:energy_budget => true,
                                             :lobe_relaxation => relax),
                    seed! = sheet!())
                eb_run!(m, p, mo, 5)
                lc = crow(m, :MC_LOBE_C)
                if relax
                    # the relaxation lifts the negative lobe toward 0: a positive increment
                    @test lc > 0.0
                else
                    @test lc === 0.0
                    snap = Scythe.energy_budget_snapshot!(m)
                    @test all(snap.dR[Scythe.EB_LOBE, :] .== 0.0)
                end
                @test crow(m, :MC_LOBE_R) >= 0.0
            end
        end
    end

    # ── 4. what the isolated operators carry ─────────────────────────────────────────
    @testset "refit partner off: its sub-row is an exact zero, the budget still closes" begin
        mktempdir() do tmp
            m, p, mo = eb_tile(tmp, "nopart";
                extra = Dict{Symbol,Any}(:energy_budget => true, :refit_partner => false),
                seed! = sheet!())
            eb_run!(m, p, mo, 5)
            snap = Scythe.energy_budget_snapshot!(m)
            @test all(snap.dR[Scythe.EB_ETD_PARTNER, :] .== 0.0)
            okR, okX = closure_check(snap)
            @test okR && okX
        end
    end

    @testset "horizontal momentum diffusion and FRIC_KE: an energy-consistent pair" begin
        mktempdir() do tmp
            m, p, mo = eb_tile(tmp, "hmom"; extra = ON, seed! = sheet!(wamp = 2.0),
                               pp_extra = Dict{Symbol,Float64}(:Khdiff => 500.0))
            # ONE (Euler) step: E_t receives rho_t w dw exactly as the resolved KE falls, so
            # R moves by -rho_t dw^2/2 only, second order against the FRIC_KE booked. (Under
            # AB3 the E_t history carries (rho_t w)^{n-1} f^{n-1} against the state-n KE, an
            # O(dt) commutator of the multistep, ~5e-3 of FRIC_KE on this fixture.)
            eb_run!(m, p, mo, 1)
            snap = Scythe.energy_budget_snapshot!(m)
            eb = m.ebudget; kDim = mo.grid_params.kDim
            hm = snap.dR[5, :]
            kef = [sum(eb.wz[k] * abs(eb.tn[(c - 1) * kDim + k, eb.jEt, Scythe.EB_OP_HMOM])
                       for k in 1:kDim) * mo.ts for c in 1:eb_ncols(p, mo)]
            @test all(kef .> 0.0)
            @test maximum(abs.(hm) ./ kef) < 1.0e-3
            eb_run!(m, p, mo, 5; t0 = 2)
            snap = Scythe.energy_budget_snapshot!(m)
            @test any(!=(0.0), snap.dR[5, :])
            okR, okX = closure_check(snap)
            @test okR && okX
        end
    end

    @testset "MYNN + surface fluxes + sponge: the operators are split and the budget closes" begin
        isdefined(Main, :MYNNTestFixtures) || include("mynn_test_fixtures.jl")
        F = Main.MYNNTestFixtures
        mktempdir() do tmp
            m0, p, mo, gp, z, col = F.make_mynn_tile(tmp; mynn = true, init = :zero,
                                                     num_cells_k = 25, fidelity = nothing)
            mo.options[:energy_budget] = true
            mo.physical_params[:alpha] = 0.05
            mo.physical_params[:z_damp] = 1.0e3
            hrm = sparse(Int64[], Int64[], Float64[], size(p.spectral, 1), size(p.spectral, 2))
            m = createModelTile(p, p, mo, hrm)
            @test m.ebudget.active
            F.set_state!(p, mo, gp.kDim, z, col)
            for t in 1:5
                F.step_all!(m, p, gp.kDim, t)
                Scythe.calcTendency(m); gridTransform!(p)
            end
            snap = Scythe.energy_budget_snapshot!(m)
            okR, okX = closure_check(snap)
            @test okR && okX
            rows = Scythe.energy_budget_rows([snap])
            bl = stage(rows, "explicit:bl").dR_dt_Wm2
            sf = stage(rows, "explicit:sfc").dR_dt_Wm2
            @test bl != 0.0 && sf != 0.0
            @test stage(rows, "explicit:sponge").dR_dt_Wm2 != 0.0
            # The surface exchange is energy-consistent by construction (the sensible flux
            # heats p and E_t together, the moisture flux carries C_vv T + geo with its vapor,
            # the drag work is the KE it removes): its R row is second order in the step,
            # against the surface energy flux it carries.
            sfc = m.mynn.active ? m.surface : nothing
            fsh = maximum(abs.(sfc.F_sh)); fq = maximum(abs.(sfc.F_q))
            @test fsh > 1.0
            @test abs(sf) < 1.0e-3 * (fsh + 2.5e6 * fq)
            # The sfc tendency was moved OUT of the bl row, not duplicated: the two
            # operators' tendencies sum to what the closure wrote (bl's capture).
            eb = m.ebudget
            @test any(!=(0.0), view(eb.tn, :, eb.jEt, Scythe.EB_OP_SFC))
        end
    end

    # ── 5. allocation-free ───────────────────────────────────────────────────────────
    @testset "allocation-free column step with the budget on" begin
        mktempdir() do tmp
            for ice in (false, true)
                m, p, mo = eb_tile(tmp, "alloc_$(ice)"; ice = ice, extra = ON,
                    seed! = sheet!(),
                    pp_extra = Dict{Symbol,Float64}(:Kvdiff => 10.0, :Kvdiff_heat => 10.0,
                                                    :Khdiff => 50.0, :alpha => 0.05,
                                                    :z_damp => 8.0e3))
                kDim = mo.grid_params.kDim
                Scythe.moist_compressible_XZ(m, 1, kDim, 2)          # compile
                @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 2)) == 0
                Scythe.moist_compressible_XZ(m, 1, kDim, 4)
                @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 4)) == 0
            end
        end
        isdefined(Main, :MYNNTestFixtures) || include("mynn_test_fixtures.jl")
        F = Main.MYNNTestFixtures
        mktempdir() do tmp
            _, p, mo, gp, z, col = F.make_mynn_tile(tmp; mynn = true, init = :zero,
                                                   num_cells_k = 25, fidelity = nothing)
            mo.options[:energy_budget] = true
            hrm = sparse(Int64[], Int64[], Float64[], size(p.spectral, 1), size(p.spectral, 2))
            m = createModelTile(p, p, mo, hrm)
            F.set_state!(p, mo, gp.kDim, z, col)
            kDim = gp.kDim
            Scythe.moist_compressible_XZ(m, 1, kDim, 3)
            Scythe.moist_compressible_XZ(m, 1, kDim, 3)
            @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 3)) == 0
        end
    end

    # ── 6. registration, the CSV, the refusals ───────────────────────────────────────
    @testset "config key, CSV sidecar, exact_si refused" begin
        @test haskey(Scythe.KNOWN_OPTION_KEYS, :energy_budget)
        @test Scythe.KNOWN_OPTION_KEYS[:energy_budget][1] === Bool
        @test !Scythe.energy_budget_on(Dict{Symbol,Any}())
        @test Scythe.energy_budget_on(Dict{Symbol,Any}(:energy_budget => true))
        mktempdir() do tmp
            path = joinpath(tmp, "eb.toml")
            write(path, "[extra.options]\nenergy_budget = true\n")
            cfg = Scythe.load_config(path)
            @test cfg.extra.options[:energy_budget] === true
            o, _ = Scythe.build_physics(cfg)
            @test o[:energy_budget] === true
            # the wrong type is refused by the registry
            write(path, "[extra.options]\nenergy_budget = 1\n")
            @test_throws Exception Scythe.load_config(path)

            # The CSV: header once, one row per stage per output time, appended.
            m, p, mo = eb_tile(tmp, "csv"; extra = ON, seed! = sheet!())
            dir = mktempdir(tmp)
            for k in 1:2
                eb_run!(m, p, mo, 3; t0 = 3k - 2)
                rows = Scythe.energy_budget_rows([Scythe.energy_budget_snapshot!(m)])
                Scythe.write_energy_budget(dir, 1.5k, rows)
            end
            lines = readlines(joinpath(dir, "energy_budget.csv"))
            @test lines[1] == Scythe.EB_CSV_HEADER
            @test length(lines) == 1 + 2 * (Scythe.EB_NSTAGE + 3)
            f = split(lines[2], ",")
            @test f[1] == "1.5" && f[2] == "refit" && length(f) == 6
            @test all(l -> length(split(l, ",")) == 6, lines)
            @test any(l -> startswith(l, "3.0,closure,"), lines)

            # exact_si: the column step ends outside the marked stages -> refused at setup.
            err = try
                eb_tile(tmp, "xsi";
                        extra = Dict{Symbol,Any}(:energy_budget => true, :exact_si => true))
                nothing
            catch e
                e
            end
            @test err isa ErrorException && occursin("energy_budget", err.msg)
        end
    end
end
