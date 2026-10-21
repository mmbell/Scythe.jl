# test/test_si_water_slaving.jl -- the ν-flux SI slaving of the water slots
# (`options[:si_water_slots]`, Stage 2 of the rho_t retirement; reference/
# RHO_T_RETIREMENT_DESIGN.md §4.2 (i′)/(iii), §6 Stage 2; src/si_water_slaving.jl).
#
# What has to hold (plan §4 Stage 2, tests first):
#   1. per column Δ(ρ_d' + Σ α_k ν_k') = Δρ_t' (slot 3) to rounding on the state-dependent
#      path, the reference path, exact_si phase B and the horizontal sweep — and the same for
#      the staged AI2* histories;
#   2. rest is bitwise: a resting column is a fixed point under :bhyp and :none vapor, every
#      state, tendency and increment exactly as with the slaving off;
#   3. dry is bitwise (every c_k = 0, the −0.0 select);
#   4. the staged `impdot[k]` and the solve's increment are ONE operator (bitwise);
#   5. a mixed vertical wall family across the slaved slots is refused at setup.
# Plus the option and config plumbing, the allocation-free column, and the feeder-1 check
# (the vapor gap no longer grows under the acoustic step).

using Test
using Scythe
using Springsteel
using SparseArrays
using Springsteel: PeriodicBC, NeumannBC, DirichletBC, NaturalBC
import Springsteel.Thermodynamics: gravity

isdefined(Main, :VaporTransformFixtures) || include("vapor_transform_fixtures.jl")

@testset "SI water slaving (options[:si_water_slots])" begin
    F = Main.VaporTransformFixtures
    VMU = 1.0e-9
    ep = eps(Float64)

    # A moist XZ tile like `F.vt_tile` but with a choice of side walls (the exact 2-D solve and
    # the horizontal sweep need rigid walls) and per-slot ground BC overrides.
    function siw_tile(tmpdir, tag; seed! = (p, x, V, col) -> nothing, vtr = :bhyp,
                      ice = false, tr = :bhyp, ts = 0.5, rh = z -> 0.7, walls = false,
                      bcb_override = Dict{String,Any}(), extra = Dict{Symbol,Any}(),
                      pp_extra = Dict{Symbol,Float64}())
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :consistent_qss_reference => true,
                                :hydrostatic_reference => true,
                                :precipitation => ice, :vertical_mixing => false,
                                :condensate_transform => tr, :rain_transform => tr,
                                :vapor_transform => vtr)
        if ice
            opts[:rain_moments] = 2
            opts[:rain_number_transform] = tr
            opts[:ice_microphysics] = :ishmael
            opts[:ice_transform] = tr
        end
        merge!(opts, extra)
        varnames = Scythe.mc_var_names(opts)
        vars = Dict(v => i for (i, v) in enumerate(varnames))
        side_bc = walls ?
            merge(Dict{String,Any}(v => NeumannBC() for v in varnames),
                  Dict{String,Any}("u" => DirichletBC(), "w" => DirichletBC())) :
            Dict{String,Any}(v => PeriodicBC() for v in varnames)
        wall_bc = merge(Dict{String,Any}(v => NeumannBC() for v in varnames),
                        Dict{String,Any}("w" => DirichletBC()))
        bcb = merge(wall_bc, bcb_override)
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = F.SPAN, num_cells_i = F.NCELL,
            kMin = 0.0, kMax = F.ZTOP, num_cells_k = F.NCELL_K,
            BCL = side_bc, BCR = side_bc, BCB = bcb, BCT = wall_bc, vars = vars)
        ref_file = joinpath(tmpdir, "siw_$(tag).ref")
        pp = Dict{Symbol,Float64}(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                                  :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                  :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3,
                                  :f => 0.0, :N_r => 1.0e-3, :l_inf => 80.0)
        merge!(pp, pp_extra)
        model = ModelParameters(ts = ts, integration_time = 10.0, output_interval = 10.0,
            equation_set = "moist_compressible_XZ", ref_state_file = ref_file,
            grid_params = gp, physical_params = pp, options = opts,
            output_dir = joinpath(tmpdir, "out_$(tag)/"), initial_conditions = "ics.csv")
        gp = model.grid_params
        patch = createGrid(gp)
        pts = Scythe.getGridpoints(patch)
        z = pts[1:gp.kDim, end]
        col = F.vt_column(z, rh)
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

    # The derived reference vapor a tile carries its vapor slot against.
    rvbar(m) = m.mc_ref_diag.rho_vbar
    # Seed: the kinked cloud sheet, a prognostic vapor bump (in the tile's own transform), w,
    # and (ice) rain and ice-mass blobs so every water-mass slot carries water.
    function seed_all(m0; ice = false, vtr = :bhyp, tr = :bhyp, vamp = 1.0e-3)
        rvb = rvbar(m0)
        return (patch, pts, V, col) -> begin
            # `vamp`: the vapor bump at 3 km raised in rho_t ONLY (a seeded partition gap)
            F.vt_seed_vapor!(patch, pts, V, col; vtr = vtr, vmu = VMU, rho_vbar = rvb, vamp)
            kDim = length(col.Tk)
            for i in 1:size(patch.physical, 1)
                z = pts[i, end]
                b = exp(-((z - 2500.0) / 800.0)^2) * (1.0 + 0.3 * sin(2pi * pts[i, 1] / F.SPAN))
                rr = 4.0e-4 * b
                patch.physical[i, V["rho_t"], 1] += rr
                patch.physical[i, Scythe.mc_slot(V, "rho_r"), 1] =
                    tr === :none ? rr : Scythe.bhyp(rr, 1.0e-7)
                if ice
                    # all four moments of each species, the ratios of the ice transport test
                    # (mass kg/m3 : number #/m3 : the two volume moments m3/m3)
                    scale = (1.0e-4, 5.0e3, 2.0e-12, 1.0e-12)
                    for sp in 1:3
                        gsp = (0.5 * sp) * exp(-((z - (5000.0 + 800.0 * sp)) / 900.0)^2)
                        for jm in 1:4
                            nm = Scythe.MC_ICE_VARS[4 * (sp - 1) + jm]
                            x = scale[jm] * gsp
                            jm == 1 && (patch.physical[i, V["rho_t"], 1] += x)
                            patch.physical[i, Scythe.mc_slot(V, nm), 1] =
                                tr === :none ? x :
                                Scythe.bhyp(x, Scythe.ice_mu(Dict{Symbol,Any}(), jm))
                        end
                    end
                    # a rain number with the rain mass (2.5e-4 kg per 1e3 drops ~ 1 mm drops)
                    patch.physical[i, Scythe.mc_slot(V, "n_r"), 1] =
                        tr === :none ? 1.0e3 * b : Scythe.bhyp(1.0e3 * b, 1.0)
                end
            end
        end
    end
    # Build a tile twice: once (unseeded) to learn its derived reference vapor, then seeded.
    function seeded_tile(tmpdir, tag; ice = false, vtr = :bhyp, tr = :bhyp, vamp = 1.0e-3, kw...)
        m0, _, _ = siw_tile(tmpdir, tag * "_probe"; ice, vtr, tr, kw...)
        return siw_tile(tmpdir, tag; seed! = seed_all(m0; ice, vtr, tr, vamp), ice, vtr, tr,
                        kw...)
    end

    # The water-mass slots of a tile with their weights α (2 transformed, 1 a density slot).
    function mass_members(m)
        ws = Scythe.si_water_set(m.model.options, m.model.physical_params, m.mc_slots)
        return [(ws.slot[j], ws.alpha[j]) for j in 1:16 if ws.slot[j] > 0 && ws.alpha[j] > 0.0]
    end

    ncols(p, mo) = div(size(p.physical, 1), mo.grid_params.kDim)

    # The identity residual |Δρ_d + Σ α Δν − Δρ_t| at every row, in units of the rounding of
    # the summands: max over rows of residual / (ε · Σ|summands|). Returns (ratio, max|Δρ_t|,
    # max over rows of the water share Σ α|Δν|, so a vacuous zero cannot pass).
    function identity_ratio(D, rows, rd, rt, members)
        worst = 0.0; dmax = 0.0; wmax = 0.0
        for g in rows
            s = D[g, rd]; scale = abs(D[g, rd]) + abs(D[g, rt]); wsum = 0.0
            for (k, a) in members
                s += a * D[g, k]; scale += a * abs(D[g, k]); wsum += a * abs(D[g, k])
            end
            r = abs(s - D[g, rt])
            scale > 0.0 && (worst = max(worst, r / (ep * scale)))
            dmax = max(dmax, abs(D[g, rt])); wmax = max(wmax, wsum)
        end
        return worst, dmax, wmax
    end

    # ── 0. the option ──────────────────────────────────────────────────────────────
    @testset "option, members and registration" begin
        @test Scythe.si_water_slots_mode(Dict{Symbol,Any}()) === :none
        for m in (:none, :mass, :all)
            @test Scythe.si_water_slots_mode(Dict{Symbol,Any}(:si_water_slots => m)) === m
        end
        @test_throws ErrorException Scythe.si_water_slots_mode(
            Dict{Symbol,Any}(:si_water_slots => :vapor))
        @test haskey(Scythe.KNOWN_OPTION_KEYS, :si_water_slots)
        @test haskey(Scythe.KNOWN_OPTION_KEYS, :si_water_mixed_walls)
        @test Scythe.OWNED_OPTION_KEYS[:si_water_slots] == "numerics.si_water_slots"
        @test Scythe.Numerics().si_water_slots === :mass
        IS = Scythe.MCSlots(10, 11, 12:23..., 0)
        o = Dict{Symbol,Any}(:si_water_slots => :mass, :vapor_transform => :bhyp,
                             :condensate_transform => :bhyp, :rain_transform => :none,
                             :ice_microphysics => :ishmael, :ice_transform => :bhyp,
                             :rain_moments => 2)
        ws = Scythe.si_water_set(o, Dict{Symbol,Float64}(), IS)
        @test ws.on
        act = [(ws.slot[j], ws.alpha[j], ws.ref[j]) for j in 1:16 if ws.slot[j] > 0]
        @test act == [(10, 2.0, 1), (9, 2.0, 2), (8, 1.0, 0), (12, 2.0, 0), (16, 2.0, 0),
                      (20, 2.0, 0)]
        o[:si_water_slots] = :all
        ws = Scythe.si_water_set(o, Dict{Symbol,Float64}(), IS)
        act = [(ws.slot[j], ws.alpha[j]) for j in 1:16 if ws.slot[j] > 0]
        @test length(act) == 16
        @test [a for (_, a) in act[7:end]] == zeros(10)
        @test Set(s for (s, _) in act[7:end]) == Set([11, 13, 14, 15, 17, 18, 19, 21, 22, 23])
        @test !Scythe.si_water_set(Dict{Symbol,Any}(), Dict{Symbol,Float64}(), IS).on
    end

    # ── 1. the identity, state-dependent and reference vertical paths ────────────────
    @testset "Δ(ρ_d' + Σα ν') = Δρ_t' per column: vertical solve (sd=$(sd), ice=$(ice), vtr=$(vtr))" for
            sd in (true, false), ice in (false, true), vtr in (:bhyp, :none)
        mktempdir() do tmpdir
            m, p, mo = seeded_tile(tmpdir, "id_$(sd)_$(ice)_$(vtr)"; ice, vtr,
                extra = Dict{Symbol,Any}(:si_water_slots => :mass,
                                         :state_dependent_si => sd))
            F.vt_run!(m, p, mo, 3)
            V = mo.grid_params.vars
            rd = V["rho_d"]; rt = V["rho_t"]
            members = mass_members(m)
            @test length(members) == (ice ? 6 : 3)
            kDim = mo.grid_params.kDim
            t = 4
            worst = 0.0; worst_h = 0.0; dmax = 0.0; wmax = 0.0
            for c in 1:ncols(p, mo)
                cs = (c - 1) * kDim + 1; ce = cs + kDim - 1
                Scythe.advance_column(m, c, t)
                # (a) the staged histories: impdot[2] + Σ α impdot[k] = impdot[3]
                r, _, _ = identity_ratio(m.impdot_n, cs:ce, rd, rt, members)
                worst_h = max(worst_h, r)
                # (b) the solve's slaved increments from a zero predictor (so the advanced
                # value IS the increment, with no rounding of a carried state)
                for (k, _) in members; m.var_np1[cs:ce, k] .= 0.0; end
                m.var_np1[cs:ce, rd] .= 0.0; m.var_np1[cs:ce, rt] .= 0.0
                Scythe.semiimplicit_adjustment_p(m, cs, ce, t; apply_histories = false)
                r, d, w = identity_ratio(m.var_np1, cs:ce, rd, rt, members)
                worst = max(worst, r); dmax = max(dmax, d); wmax = max(wmax, w)
            end
            @info "vertical identity" sd ice vtr worst worst_h dmax wmax
            @test dmax > 0.0 && wmax > 1.0e-6 * dmax      # the water really moved
            # Reference path: below 1 ulp of the summands (the share form cancels in the
            # z-only coefficients). State-dependent path: the quotient-rule z-derivatives
            # c_{d,z} + Σ α c_{k,z} cancel to rounding of terms ~ρ_d ρ̂_z/ρ̂² φ that are larger
            # than the increments they sum to, so the residual is a few ulps of the summands
            # (measured ≤ 13).
            @test worst <= (sd ? 32.0 : 2.0)
            @test worst_h <= (sd ? 32.0 : 2.0)
        end
    end

    # ── 1c/1d. exact 2-D solve phase B and the horizontal sweep ────────────────────
    function xsi_step!(m, p, mo, esd, hfields, t; check_col = nothing)
        kDim = mo.grid_params.kDim
        V = mo.grid_params.vars
        npts = size(p.physical, 1)
        xstar = zeros(npts, 3)
        t > 1 && Scythe.exact_si_load_history!(m, hfields, t, 1)
        for c in 1:ncols(p, mo)
            Scythe.advance_column(m, c, t)
        end
        xstar[:, 1] .= view(m.var_np1, :, V["u"])
        xstar[:, 2] .= view(m.var_np1, :, V["w"])
        xstar[:, 3] .= view(m.var_np1, :, V["p"])
        Scythe.exact_si_solve!(hfields, esd, p, mo, t, xstar)
        out = nothing
        for c in 1:ncols(p, mo)
            cs = (c - 1) * kDim + 1; ce = cs + kDim - 1
            if check_col !== nothing
                for (k, _) in check_col; m.var_np1[cs:ce, k] .= 0.0; end
                m.var_np1[cs:ce, V["rho_d"]] .= 0.0; m.var_np1[cs:ce, V["rho_t"]] .= 0.0
            end
            Scythe.exact_si_apply_column!(m, cs, ce, t, hfields, 1)
        end
        Scythe.calcTendency(m)
        gridTransform!(p)
        return nothing
    end
    function make_esd(m, p, mo)
        Scythe.create_exact_si_data(p, mo, m.mc_ref_diag.Pxi_prof,
            collect(view(Springsteel.ref_rho_t(m.ref_state), :, 1:2)),
            collect(view(Springsteel.ref_rho_d(m.ref_state), :, 1:2)),
            collect(view(Springsteel.ref_total_energy(m.ref_state), :, 1:2) .+
                    view(Springsteel.ref_pressure(m.ref_state), :, 1:2)))
    end

    @testset "Δ(ρ_d' + Σα ν') = Δρ_t' per column: exact_si phase B (vtr=$(vtr))" for vtr in (:bhyp, :none)
        mktempdir() do tmpdir
            m, p, mo = seeded_tile(tmpdir, "xsi_$(vtr)"; vtr, walls = true,
                extra = Dict{Symbol,Any}(:si_water_slots => :mass, :exact_si => true))
            esd = make_esd(m, p, mo)
            hfields = zeros(size(p.physical, 1), Scythe.XSI_NPLANES)
            for t in 1:3
                xsi_step!(m, p, mo, esd, hfields, t)
            end
            V = mo.grid_params.vars
            members = mass_members(m)
            # the staged histories of step 4, then phase B from a zero predictor
            t = 4
            kDim = mo.grid_params.kDim
            Scythe.exact_si_load_history!(m, hfields, t, 1)
            for c in 1:ncols(p, mo); Scythe.advance_column(m, c, t); end
            rh, _, _ = identity_ratio(m.impdot_n, 1:size(p.physical, 1), V["rho_d"], V["rho_t"],
                                      members)
            # the horizontal (hacdot) histories too
            rhh, _, _ = identity_ratio(m.hacdot_n, 1:size(p.physical, 1), V["rho_d"],
                                       V["rho_t"], members)
            npts = size(p.physical, 1)
            xstar = hcat(m.var_np1[:, V["u"]], m.var_np1[:, V["w"]], m.var_np1[:, V["p"]])
            Scythe.exact_si_solve!(hfields, esd, p, mo, t, xstar)
            for (k, _) in members; m.var_np1[:, k] .= 0.0; end
            m.var_np1[:, V["rho_d"]] .= 0.0; m.var_np1[:, V["rho_t"]] .= 0.0
            for c in 1:ncols(p, mo)
                cs = (c - 1) * kDim + 1
                Scythe.exact_si_apply_column!(m, cs, cs + kDim - 1, t, hfields, 1)
            end
            r, d, w = identity_ratio(m.var_np1, 1:npts, V["rho_d"], V["rho_t"], members)
            @info "exact_si identity" vtr r rh rhh d w
            @test d > 0.0 && w > 1.0e-6 * d
            @test r <= 2.0
            @test rh <= 2.0
            @test rhh <= 2.0
        end
    end

    @testset "Δ(ρ_d' + Σα ν') = Δρ_t' per column: horizontal sweep (vtr=$(vtr))" for vtr in (:bhyp, :none)
        mktempdir() do tmpdir
            m, p, mo = seeded_tile(tmpdir, "hsi_$(vtr)"; vtr, walls = true,
                extra = Dict{Symbol,Any}(:si_water_slots => :mass,
                                         :horizontal_semiimplicit => true))
            @test_throws ErrorException Scythe.create_horizontal_solve_data(p, mo,
                m.mc_ref_diag.Pxi_prof,
                collect(view(Springsteel.ref_rho_t(m.ref_state), :, 1)),
                collect(view(Springsteel.ref_rho_d(m.ref_state), :, 1)),
                collect(view(Springsteel.ref_total_energy(m.ref_state), :, 1) .+
                        view(Springsteel.ref_pressure(m.ref_state), :, 1)))
            water = Scythe.si_water_hsi_profiles(m)
            @test water[1] == [m.mc_slots.rho_v]        # cloud-free base: the vapor alone
            @test all(==(vtr === :bhyp ? 0.5 : 1.0), water[2])
            hsd = Scythe.create_horizontal_solve_data(p, mo, m.mc_ref_diag.Pxi_prof,
                collect(view(Springsteel.ref_rho_t(m.ref_state), :, 1)),
                collect(view(Springsteel.ref_rho_d(m.ref_state), :, 1)),
                collect(view(Springsteel.ref_total_energy(m.ref_state), :, 1) .+
                        view(Springsteel.ref_pressure(m.ref_state), :, 1)); water)
            npts = size(p.physical, 1)
            u_incr = zeros(npts, 6)
            V = mo.grid_params.vars
            for t in 1:4
                t > 1 && Scythe.horizontal_si_load_increment!(m, u_incr, t, 1)
                for c in 1:ncols(p, mo); Scythe.advance_column(m, c, t); end
                Scythe.calcTendency(m)
                if t == 4
                    rh, _, _ = identity_ratio(m.hacdot_n, 1:npts, V["rho_d"], V["rho_t"],
                                              [(m.mc_slots.rho_v, vtr === :bhyp ? 2.0 : 1.0)])
                    # the sweep's coefficient increments from a zero carried state
                    sp = p.spectral
                    for v in ("rho_d", "rho_t"); sp[:, V[v]] .= 0.0; end
                    sp[:, m.mc_slots.rho_v] .= 0.0
                    Scythe.horizontal_si_correct!(sp, p, mo, hsd, t)
                    a = vtr === :bhyp ? 2.0 : 1.0
                    # the physical-point correction planes the sweep wrote back: δρ_d (plane
                    # 3), δρ_t (plane 4) and the vapor's (the last water plane, `wlev`)
                    worst = 0.0; dmax = 0.0
                    for i in axes(hsd.wlev, 1), k in axes(hsd.wlev, 2)
                        dd = hsd.dlev[i, k, 3]; dt = hsd.dlev[i, k, 4]; dv = hsd.wlev[i, k]
                        sc = abs(dd) + a * abs(dv) + abs(dt)
                        sc > 0.0 && (worst = max(worst, abs((dd + a * dv) - dt) / (ep * sc)))
                        dmax = max(dmax, abs(dv))
                    end
                    # ...and in B coefficients after the write-back, a linear map per slot
                    # (one basis for the three): to the solve's conditioning, not to an ulp
                    cw = 0.0; cmax = 0.0
                    for r in 1:size(sp, 1)
                        cw = max(cw, abs((sp[r, V["rho_d"]] + a * sp[r, m.mc_slots.rho_v]) -
                                         sp[r, V["rho_t"]]))
                        cmax = max(cmax, abs(sp[r, V["rho_t"]]))
                    end
                    @info "hsi identity" vtr worst rh dmax cw cmax
                    @test dmax > 0.0
                    @test worst <= 2.0
                    @test cw <= 1.0e-10 * cmax
                    @test rh <= 2.0
                else
                    u_incr .= Scythe.horizontal_si_correct!(p.spectral, p, mo, hsd, t)
                end
                gridTransform!(p)
            end
        end
    end

    # ── 2/3. rest and dry are bitwise ─────────────────────────────────────────────
    bits(A) = reinterpret(UInt64, vec(copy(A)))
    function run_pair(tmpdir, tag; nsteps = 6, seeded = false, kw...)
        out = Dict{Symbol,Any}()
        for mode in (:none, :mass, :all)
            extra = merge(get(kw, :extra, Dict{Symbol,Any}()),
                          Dict{Symbol,Any}(:si_water_slots => mode))
            kw2 = merge(Dict{Symbol,Any}(kw), Dict{Symbol,Any}(:extra => extra))
            m, p, mo = seeded ? seeded_tile(tmpdir, "$(tag)_$(mode)"; kw2...) :
                                siw_tile(tmpdir, "$(tag)_$(mode)"; kw2...)
            F.vt_run!(m, p, mo, nsteps)
            out[mode] = (m, p, mo)
        end
        return out
    end
    @testset "rest is bitwise (sd=$(sd), vtr=$(vtr), ice=$(ice))" for sd in (true, false),
            vtr in (:bhyp, :none), ice in (false, true)
        mktempdir() do tmpdir
            r = run_pair(tmpdir, "rest_$(sd)_$(vtr)_$(ice)"; vtr, ice,
                         extra = Dict{Symbol,Any}(:state_dependent_si => sd))
            m0, p0, mo0 = r[:none]
            for mode in (:mass, :all)
                m1, p1, mo1 = r[mode]
                @test bits(p1.physical) == bits(p0.physical)
                @test bits(m1.var_np1) == bits(m0.var_np1)
                @test bits(m1.expdot_n) == bits(m0.expdot_n)
                @test bits(m1.expdot_nm1) == bits(m0.expdot_nm1)
                # the dry legs' histories bitwise; the water rows hold exact zeros
                for v in ("p", "rho_d", "rho_t", "w", "E_t")
                    i = mo0.grid_params.vars[v]
                    @test bits(m1.impdot_n[:, i]) == bits(m0.impdot_n[:, i])
                end
                @test all(iszero, m1.impdot_n)  == all(iszero, m0.impdot_n)
                # a resting column is a fixed point: every perturbation slot is still 0
                @test all(iszero, p1.physical[:, :, 1])
            end
        end
    end

    @testset "dry is bitwise (sd=$(sd))" for sd in (true, false)
        mktempdir() do tmpdir
            dry_seed = (patch, pts, V, col) -> begin
                for i in 1:size(patch.physical, 1)
                    z = pts[i, end]
                    patch.physical[i, V["w"], 1] =
                        0.5 * sin(2pi * pts[i, 1] / F.SPAN) * sin(pi * z / F.ZTOP)
                end
            end
            r = run_pair(tmpdir, "dry_$(sd)"; vtr = :none, rh = z -> 0.0, seed! = dry_seed,
                         nsteps = 20, extra = Dict{Symbol,Any}(:state_dependent_si => sd))
            m0, p0, _ = r[:none]
            @test maximum(abs, p0.physical[:, 5, 1]) > 0.0      # w is live
            for mode in (:mass, :all)
                m1, p1, _ = r[mode]
                @test bits(p1.physical) == bits(p0.physical)
                @test bits(m1.var_np1) == bits(m0.var_np1)
                @test bits(m1.expdot_n) == bits(m0.expdot_n)
            end
        end
    end

    # ── 4. staging and solve are one operator ─────────────────────────────────────
    @testset "impdot[k] and the solve's increment are one operator (sd=$(sd))" for sd in (true, false)
        mktempdir() do tmpdir
            m, p, mo = seeded_tile(tmpdir, "op_$(sd)"; ice = true,
                extra = Dict{Symbol,Any}(:si_water_slots => :mass, :state_dependent_si => sd))
            F.vt_run!(m, p, mo, 2)
            kDim = mo.grid_params.kDim
            V = mo.grid_params.vars
            ws = Scythe.si_water_set(mo.options, mo.physical_params, m.mc_slots)
            members = mass_members(m)
            rc = Springsteel.ref_rho_c(m.ref_state)
            nmatch = 0
            for c in 1:ncols(p, mo)
                cs = (c - 1) * kDim + 1; ce = cs + kDim - 1
                Scythe.advance_column(m, c, 3)
                S = m.mc_scratch[Threads.threadid()]
                # φⁿ through the driver's own chain (w's column basis), as the staging fits it
                wc = Scythe.scratch_column(m, 5)
                rho_hat = sd ? S.rho_t : view(Springsteel.ref_rho_t(m.ref_state), :, 1)
                wc.uMish .= rho_hat .* p.physical[cs:ce, V["w"], 1]
                Springsteel.Btransform!(wc); Springsteel.Atransform!(wc)
                phi = copy(Springsteel.Itransform!(wc))
                phiz = zeros(kDim); Springsteel.Ixtransform(wc, phiz)
                # the solve's operator, applied to φⁿ with Δτ = 1 on a zero state
                cd = zeros(kDim); cdz = zeros(kDim)
                if sd
                    Scythe._siw_sd_cd!(cd, cdz, S.rho_d, S.rho_d_z, p.physical, cs, 4, ws,
                                       Scythe.mc_nu_vbar(m), Scythe.mc_nu_vbar_z(m),
                                       view(rc, :, 1), view(rc, :, 2))
                else
                    rdb = Springsteel.ref_rho_d(m.ref_state); rtb = Springsteel.ref_rho_t(m.ref_state)
                    @. cd = rdb[:, 1] / rtb[:, 1]
                    @. cdz = ((rdb[:, 2] * rtb[:, 1]) - (rdb[:, 1] * rtb[:, 2])) / (rtb[:, 1]^2)
                end
                Z = zeros(size(m.var_np1))
                Scythe._siw_slave!(Z, cs, ws, sd, 1.0, phi, phiz, p.physical, 4,
                                   S.rho_d, S.rho_d_z, cd, cdz, Scythe.mc_nu_vbar(m),
                                   Scythe.mc_nu_vbar_z(m), view(rc, :, 1), view(rc, :, 2))
                for (k, _) in members
                    act = sd || k in (m.mc_slots.rho_v, 9)
                    act || continue
                    # the staged history is −L(φⁿ); the solve applied −Δτ·L(φ) = Z here
                    @test bits(Z[cs:ce, k]) == bits(m.impdot_n[cs:ce, k] .+ 0.0) ||
                          Z[cs:ce, k] == m.impdot_n[cs:ce, k]
                    nmatch += 1
                end
            end
            @test nmatch > 0
        end
    end

    # ── 5. mixed wall families are refused ────────────────────────────────────────
    @testset "a mixed vertical wall family across the slaved slots is refused" begin
        mktempdir() do tmpdir
            rain_free = Dict{String,Any}("nu_r" => NaturalBC())
            err = try
                siw_tile(tmpdir, "mixed"; bcb_override = rain_free,
                         extra = Dict{Symbol,Any}(:si_water_slots => :mass))
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("do not share one vertical operator", err)
            @test occursin("nu_r", err)
            @test occursin("si_water_slots", err)
            # :none never checks; the explicit measurement opt-in builds
            @test siw_tile(tmpdir, "mixed_none"; bcb_override = rain_free,
                           extra = Dict{Symbol,Any}(:si_water_slots => :none))[1] isa Scythe.ModelTile
            @test siw_tile(tmpdir, "mixed_measure"; bcb_override = rain_free,
                           extra = Dict{Symbol,Any}(:si_water_slots => :mass,
                                                    :si_water_mixed_walls => true))[1] isa
                  Scythe.ModelTile
            # a uniform family builds
            @test siw_tile(tmpdir, "uniform"; extra = Dict{Symbol,Any}(:si_water_slots => :mass))[1] isa
                  Scythe.ModelTile
        end
    end

    # ── config plumbing ───────────────────────────────────────────────────────────
    @testset "config: numerics.si_water_slots" begin
        root = normpath(joinpath(@__DIR__, ".."))
        rce = joinpath(root, "simulations", "ocean_2D_rce", "scythe_config.toml")
        mktempdir() do dir
            m = build_model(load_config(rce; set = ["output.dir=$(dir)"]))
            @test m.options[:si_water_slots] === :mass
            m = build_model(load_config(rce; set = ["output.dir=$(dir)",
                                                    "numerics.si_water_slots=none"]))
            @test m.options[:si_water_slots] === :none
            @test_throws Exception build_model(load_config(rce; set = ["output.dir=$(dir)",
                                                           "numerics.si_water_slots=vapor"]))
            ov = joinpath(@__DIR__, "reference_data", "legacy_models", "rce_default.overlay.toml")
            @test build_model(load_config(rce, ov; set = ["output.dir=$(dir)"])).options[:si_water_slots] === :none
        end
    end

    # ── allocation-free, and feeder 1 is gone ─────────────────────────────────────
    @testset "the slaving helpers are allocation-free (sd=$(sd))" for sd in (true, false)
        # (The full column is not allocation-free on this fixture with the slaving OFF either:
        # the per-column state-dependent Helmholtz and the option reads allocate there, which
        # test_allocations.jl's tiles do not exercise. What this stage adds must not.)
        mktempdir() do tmpdir
            m, p, mo = seeded_tile(tmpdir, "alloc_$(sd)"; ice = true,
                extra = Dict{Symbol,Any}(:si_water_slots => :mass, :state_dependent_si => sd))
            F.vt_run!(m, p, mo, 2)
            Scythe.advance_column(m, 1, 3)
            kD = mo.grid_params.kDim
            S = m.mc_scratch[Threads.threadid()]
            rc = Springsteel.ref_rho_c(m.ref_state)
            a = view(rc, :, 1); b = view(rc, :, 2)
            nv = Scythe.mc_nu_vbar(m); nz = Scythe.mc_nu_vbar_z(m)
            set() = Scythe.si_water_set(mo.options, mo.physical_params, m.mc_slots)
            ws = set()
            cd!() = Scythe._siw_sd_cd!(S.si_c_d, S.si_c_d_z, S.rho_d, S.rho_d_z, p.physical,
                                       1, 4, ws, nv, nz, a, b)
            st!() = Scythe._siw_stage_impdot!(m.impdot_n, 1, ws, sd, p.physical, 4, S.rho_d,
                                              S.rho_d_z, S.si_c_d, S.si_c_d_z, S.si_c_e,
                                              S.si_c_e_z, nv, nz, a, b)
            sl!() = Scythe._siw_slave!(m.var_np1, 1, ws, sd, 0.625, S.si_c_e, S.si_c_e_z,
                                       p.physical, 4, S.rho_d, S.rho_d_z, S.si_c_d, S.si_c_d_z,
                                       nv, nz, a, b)
            hi!() = Scythe._siw_histories!(m.var_np1, m.impdot_n, m.impdot_nm1, 1, kD, ws, sd,
                                           3, 0.5)
            ab!() = Scythe._siw_addback!(m.expdot_n, 1, ws, sd, p.physical, 4,
                                         view(p.physical, 1:kD, 5, 1), view(p.physical, 1:kD, 5, 4),
                                         S.rho_t, S.rho_t_z, S.rho_d, S.rho_d_z,
                                         view(Springsteel.ref_rho_t(m.ref_state), :, 1),
                                         view(Springsteel.ref_rho_t(m.ref_state), :, 2),
                                         view(Springsteel.ref_rho_d(m.ref_state), :, 1),
                                         view(Springsteel.ref_rho_d(m.ref_state), :, 2),
                                         nv, nz, a, b, false, S.ADV, m.hacdot_n, false)
            sg!() = Scythe._siw_stage!(m, 1, kD)
            for f in (set, cd!, st!, sl!, hi!, ab!, sg!)
                f()
                @test (@allocations f()) == 0
            end
        end
    end

    @testset "feeder 1: the acoustic step no longer opens the water gap (sd=$(sd))" for sd in (true, false)
        # The gap the slaving closes is the PARTITION gap δ = ρ_t − ρ_d − Σ α ν (the anchor
        # minus the water sum the slots carry). Isolated: smooth vapor and w, no cloud kink
        # (whose refit ringing is the dominant, separate feeder), no condensation, the vapor
        # nudge off (it closes δ by design). With the slots slaved the acoustic step leaves δ
        # at rounding; without, the explicit water compression and the implicit anchor disagree
        # every step (measured: 5.9e-7 at w ~ 0.5 m/s, 5.9e-6 at 5 m/s over 40 steps; slaved
        # ≤ 2e-17).
        mktempdir() do tmpdir
            m0, _, _ = siw_tile(tmpdir, "gap_probe_$(sd)")
            rvb = rvbar(m0)
            gap = Dict{Symbol,Float64}()
            for mode in (:none, :mass)
                m, p, mo = siw_tile(tmpdir, "gap_$(sd)_$(mode)";
                    seed! = (pa, x, V, c) -> F.vt_seed_vapor!(pa, x, V, c; vtr = :bhyp,
                        vmu = VMU, rho_vbar = rvb, amp = 0.0, vamp = 0.0, wamp = 5.0),
                    extra = Dict{Symbol,Any}(:si_water_slots => mode, :state_dependent_si => sd,
                                             :condensation => false),
                    pp_extra = Dict{Symbol,Float64}(:tau_rho_v_rec => 1.0e12))
                V = mo.grid_params.vars
                ws = Scythe.si_water_set(merge(mo.options,
                    Dict{Symbol,Any}(:si_water_slots => :mass)), mo.physical_params, m.mc_slots)
                members = [(ws.slot[j], ws.alpha[j]) for j in 1:16
                           if ws.slot[j] > 0 && ws.alpha[j] > 0.0]
                dgap() = begin
                    d = p.physical[:, V["rho_t"], 1] .- p.physical[:, V["rho_d"], 1]
                    for (k, a) in members; d .-= a .* p.physical[:, k, 1]; end
                    d
                end
                g0 = dgap()
                F.vt_run!(m, p, mo, 40)
                gap[mode] = maximum(abs, dgap() .- g0)
            end
            @info "partition-gap change over 40 steps (acoustic feeder)" sd gap[:none] gap[:mass]
            @test gap[:none] > 1.0e-7
            @test gap[:mass] < 1.0e-14
        end
    end
end
