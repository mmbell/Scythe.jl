# test/test_si_water_slaving.jl -- the ν-flux SI slaving of the water slots
# (`options[:si_water_slots]`, Stage 2 of the rho_t retirement; reference/
# RHO_T_RETIREMENT_DESIGN.md §4.2 (i′)/(iii), §6 Stage 2; src/si_water_slaving.jl).
#
# What has to hold (plan §4 Stage 2, tests first):
#   1. per column Δ(ρ_d' + Σ α_k ν_k') = Δρ_t' to rounding on the state-dependent path, the
#      reference path, exact_si phase B and the horizontal sweep — and the same for the staged
#      AI2* histories. Since Stage 4 of the ρ_t retirement there is no ρ_t slot to read Δρ_t'
#      from: the target is the anchor's own increment, which the solve forms and the deleted
#      slot 3 used to receive bit for bit (−Δτ φ_z, the history −φ_z, exact_si's plane 5, the
#      sweep's plane 4, the fresh horizontal history −ρ̄_t·LDIV);
#   2. rest is bitwise: a resting column is a fixed point under :bhyp and :none vapor, every
#      state, tendency and increment exactly as with the slaving off;
#   3. dry is bitwise (every c_k = 0, the −0.0 select);
#   4. the staged `impdot[k]` and the solve's increment are ONE operator (bitwise);
#   5. a mixed vertical wall family across the slaved slots is refused at setup.
# Plus the option and config plumbing and the allocation-free column. (The feeder-1 check --
# the slot-3-vs-water-sum partition gap under the acoustic step -- went with slot 3: the gap is
# identically zero by construction now.)

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
                      rc_ref = nothing, cqss = true, bct_override = Dict{String,Any}(),
                      side_override = Dict{String,Any}(), lq = Dict("default" => 2.0),
                      sfilt = Dict{String,Any}(), pos = Dict{String,Any}(),
                      bcb_override = Dict{String,Any}(), extra = Dict{Symbol,Any}(),
                      pp_extra = Dict{Symbol,Float64}())
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :consistent_qss_reference => cqss,
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
        bct = merge(wall_bc, bct_override)
        bcl = merge(side_bc, side_override)
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = F.SPAN, num_cells_i = F.NCELL,
            kMin = 0.0, kMax = F.ZTOP, num_cells_k = F.NCELL_K,
            BCL = bcl, BCR = side_bc, BCB = bcb, BCT = bct, vars = vars, l_q = lq,
            spline_filter = sfilt, positivity = pos)
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
        # `rc_ref`: a CLOUDY reference (a saturated layer carrying rho_cbar(z) > 0)
        rcb = rc_ref === nothing ? col.rho_c : rc_ref.(z)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, rcb)
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
            # `vamp`: the vapor-energy bump at 3 km in E_t (the fixture's; until Stage 4 it was
            # also a seeded partition gap in the prognostic rho_t)
            F.vt_seed_vapor!(patch, pts, V, col; vtr = vtr, vmu = VMU, rho_vbar = rvb, vamp)
            kDim = length(col.Tk)
            for i in 1:size(patch.physical, 1)
                z = pts[i, end]
                b = exp(-((z - 2500.0) / 800.0)^2) * (1.0 + 0.3 * sin(2pi * pts[i, 1] / F.SPAN))
                rr = 4.0e-4 * b
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
    # the summands: max over rows of residual / (ε · Σ|summands|). `T[g]` is the anchor's
    # increment Δρ_t at row g (no slot carries it since Stage 4; the caller forms it exactly as
    # the deleted slot 3 received it). Returns (ratio, max|Δρ_t|, max over rows of the water
    # share Σ α|Δν|, so a vacuous zero cannot pass).
    function identity_ratio(D, rows, rd, T, members)
        worst = 0.0; dmax = 0.0; wmax = 0.0
        for g in rows
            s = D[g, rd]; scale = abs(D[g, rd]) + abs(T[g]); wsum = 0.0
            for (k, a) in members
                s += a * D[g, k]; scale += a * abs(D[g, k]); wsum += a * abs(D[g, k])
            end
            r = abs(s - T[g])
            scale > 0.0 && (worst = max(worst, r / (ep * scale)))
            dmax = max(dmax, abs(T[g])); wmax = max(wmax, wsum)
        end
        return worst, dmax, wmax
    end
    # The total-density reference every reader takes: the derived ρ̄_t^w (Stage 4).
    rtbar(m) = Scythe.mc_rho_tbar_ref(m)
    # The fresh horizontal history of the anchor, −ρ̄_t^w·LDIV, per column (the deleted slot 3's
    # `@. -rho_tbar * LDIV` of the driver's `hsi_like` block), from the tile's own state.
    function hacdot_rt(m, p, mo)
        kDim = mo.grid_params.kDim; T = zeros(size(p.physical, 1)); L = zeros(kDim)
        rtb = view(rtbar(m), :, 1)
        for c in 1:ncols(p, mo)
            cs = (c - 1) * kDim + 1; ce = cs + kDim - 1
            Scythe._siw_ldiv!(L, m, cs, ce)
            T[cs:ce] .= @. -rtb * L
        end
        return T
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
        # the Stage-4 XZ layout: the vapor at 3, n_r at 10, the ice at 11-22
        IS = Scythe.MCSlots(3, 10, 11:22..., 0)
        o = Dict{Symbol,Any}(:si_water_slots => :mass, :vapor_transform => :bhyp,
                             :condensate_transform => :bhyp, :rain_transform => :none,
                             :ice_microphysics => :ishmael, :ice_transform => :bhyp,
                             :rain_moments => 2)
        ws = Scythe.si_water_set(o, Dict{Symbol,Float64}(), IS)
        @test ws.on
        act = [(ws.slot[j], ws.alpha[j], ws.ref[j]) for j in 1:16 if ws.slot[j] > 0]
        @test act == [(3, 2.0, 1), (9, 2.0, 2), (8, 1.0, 0), (11, 2.0, 0), (15, 2.0, 0),
                      (19, 2.0, 0)]
        o[:si_water_slots] = :all
        ws = Scythe.si_water_set(o, Dict{Symbol,Float64}(), IS)
        act = [(ws.slot[j], ws.alpha[j]) for j in 1:16 if ws.slot[j] > 0]
        @test length(act) == 16
        @test [a for (_, a) in act[7:end]] == zeros(10)
        @test Set(s for (s, _) in act[7:end]) == Set([10, 12, 13, 14, 16, 17, 18, 20, 21, 22])
        @test !Scythe.si_water_set(Dict{Symbol,Any}(), Dict{Symbol,Float64}(), IS).on

        # The member set is WATER ONLY, in every geometry and under every option set: the
        # momentum components u, v (the tangential/meridional/third wind component, slot 10 of
        # the cylindrical layouts and a wind in any 3-D geometry), w and the non-water scalars
        # p, rho_d, E_t, Q_ss are never slaved or anchored. The set is resolved by the water
        # slot NAMES (rho_v/nu_v, rho_c/nu_c, rho_r/nu_r, the ice masses, and under :all the
        # moments), the same way `mc_slots` resolves them. Author's question 2026-10-23.
        NONWATER = ("p", "rho_d", "u", "v", "w", "E_t", "Q_ss", "rho_e")
        for cyl in (false, true),
            o in (Dict{Symbol,Any}(:si_water_slots => :all, :vapor_transform => :bhyp,
                                   :condensate_transform => :bhyp, :rain_transform => :bhyp,
                                   :rain_number_transform => :bhyp, :ice_microphysics => :ishmael,
                                   :ice_transform => :bhyp, :rain_moments => 2),
                  Dict{Symbol,Any}(:si_water_slots => :mass, :vapor_transform => :none,
                                   :condensate_transform => :none, :rain_transform => :none),
                  Dict{Symbol,Any}(:si_water_slots => :all, :vapor_transform => :bhyp,
                                   :condensate_transform => :bhyp, :rain_transform => :bhyp,
                                   :rain_moments => 1, :mynn => true))
            names = Scythe.mc_var_names(o; cyl = cyl)
            vars = Dict(n => i for (i, n) in enumerate(names))
            rv = Scythe.mc_slot(vars, "rho_v")
            nr = Scythe.rain_moments(o) == 2 ? Scythe.mc_slot(vars, "n_r") : 0
            re = get(o, :mynn, false) === true ? Scythe.mc_slot(vars, "rho_e") : 0
            ISn = Scythe.ice_microphysics(o) === :ishmael ?
                Scythe.MCSlots(rv, nr, ntuple(j -> Scythe.mc_slot(vars, Scythe.MC_ICE_VARS[j]), 12)..., re) :
                Scythe.MCSlots(rv, nr, ntuple(_ -> 0, 12)..., re)
            ws = Scythe.si_water_set(o, Dict{Symbol,Float64}(), ISn)
            members = [names[sl] for sl in ws.slot if sl > 0]
            @test !isempty(members)
            @test all(n -> !(n in NONWATER), members)
            for forbidden in NONWATER
                haskey(vars, forbidden) && @test !(vars[forbidden] in ws.slot)
            end
            cyl && @test vars["v"] == 10 && !(10 in ws.slot)
        end
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
            rd = V["rho_d"]
            members = mass_members(m)
            @test length(members) == (ice ? 6 : 3)
            kDim = mo.grid_params.kDim
            t = 4
            T = zeros(size(p.physical, 1))
            worst = 0.0; worst_h = 0.0; dmax = 0.0; wmax = 0.0
            for c in 1:ncols(p, mo)
                cs = (c - 1) * kDim + 1; ce = cs + kDim - 1
                Scythe.advance_column(m, c, t)
                S = m.mc_scratch[Threads.threadid()]
                # (a) the staged histories: impdot[2] + Σ α impdot[k] = −φ_z (the anchor's
                # staged history, the deleted impdot[3]: `@. -imp_phi_z`)
                T[cs:ce] .= @. -S.imp_phi_z
                r, _, _ = identity_ratio(m.impdot_n, cs:ce, rd, T, members)
                worst_h = max(worst_h, r)
                # (b) the solve's slaved increments from a zero predictor (so the advanced
                # value IS the increment, with no rounding of a carried state), against the
                # anchor's −Δτ φ_z (the deleted slot 3 from a zero predictor, bit for bit)
                for (k, _) in members; m.var_np1[cs:ce, k] .= 0.0; end
                m.var_np1[cs:ce, rd] .= 0.0
                Scythe.semiimplicit_adjustment_p(m, cs, ce, t; apply_histories = false)
                T[cs:ce] .= 0.0 .- (1.25 * mo.ts .* S.si_phi_z)
                r, d, w = identity_ratio(m.var_np1, cs:ce, rd, T, members)
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
                m.var_np1[cs:ce, V["rho_d"]] .= 0.0
            end
            Scythe.exact_si_apply_column!(m, cs, ce, t, hfields, 1)
        end
        Scythe.calcTendency(m)
        gridTransform!(p)
        return nothing
    end
    function make_esd(m, p, mo)
        Scythe.create_exact_si_data(p, mo, m.mc_ref_diag.Pxi_prof,
            collect(view(rtbar(m), :, 1:2)),
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
            Th = zeros(size(p.physical, 1))
            for c in 1:ncols(p, mo)
                Scythe.advance_column(m, c, t)
                cs = (c - 1) * kDim + 1
                Th[cs:(cs + kDim - 1)] .= @. -m.mc_scratch[Threads.threadid()].imp_phi_z
            end
            rh, _, _ = identity_ratio(m.impdot_n, 1:size(p.physical, 1), V["rho_d"], Th,
                                      members)
            # the horizontal (hacdot) histories too
            rhh, _, _ = identity_ratio(m.hacdot_n, 1:size(p.physical, 1), V["rho_d"],
                                       hacdot_rt(m, p, mo), members)
            npts = size(p.physical, 1)
            xstar = hcat(m.var_np1[:, V["u"]], m.var_np1[:, V["w"]], m.var_np1[:, V["p"]])
            Scythe.exact_si_solve!(hfields, esd, p, mo, t, xstar)
            for (k, _) in members; m.var_np1[:, k] .= 0.0; end
            m.var_np1[:, V["rho_d"]] .= 0.0
            for c in 1:ncols(p, mo)
                cs = (c - 1) * kDim + 1
                Scythe.exact_si_apply_column!(m, cs, cs + kDim - 1, t, hfields, 1)
            end
            # the anchor's increment is plane 5 (δρ_t = −Δτ div^{n+1}; the deleted slot 3
            # received it onto a zero predictor)
            r, d, w = identity_ratio(m.var_np1, 1:npts, V["rho_d"], 0.0 .+ hfields[:, 5], members)
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
                collect(view(rtbar(m), :, 1)),
                collect(view(Springsteel.ref_rho_d(m.ref_state), :, 1)),
                collect(view(Springsteel.ref_total_energy(m.ref_state), :, 1) .+
                        view(Springsteel.ref_pressure(m.ref_state), :, 1)))
            water = Scythe.si_water_hsi_profiles(m)
            @test water[1] == [m.mc_slots.rho_v]        # cloud-free base: the vapor alone
            @test all(==(vtr === :bhyp ? 0.5 : 1.0), water[2])
            hsd = Scythe.create_horizontal_solve_data(p, mo, m.mc_ref_diag.Pxi_prof,
                collect(view(rtbar(m), :, 1)),
                collect(view(Springsteel.ref_rho_d(m.ref_state), :, 1)),
                collect(view(Springsteel.ref_total_energy(m.ref_state), :, 1) .+
                        view(Springsteel.ref_pressure(m.ref_state), :, 1)); water)
            npts = size(p.physical, 1)
            u_incr = zeros(npts, 6)
            V = mo.grid_params.vars
            for t in 1:4
                t > 1 && Scythe.horizontal_si_load_increment!(m, u_incr, t, 1)
                for c in 1:ncols(p, mo); Scythe.advance_column(m, c, t); end
                # the anchor's fresh history from the state n the columns read (before
                # calcTendency, which writes the advanced state into the tile)
                Th = hacdot_rt(m, p, mo)
                Scythe.calcTendency(m)
                if t == 4
                    rh, _, _ = identity_ratio(m.hacdot_n, 1:npts, V["rho_d"], Th,
                                              [(m.mc_slots.rho_v, vtr === :bhyp ? 2.0 : 1.0)])
                    # the sweep's coefficient increments from a zero carried state
                    sp = p.spectral
                    sp[:, V["rho_d"]] .= 0.0
                    sp[:, m.mc_slots.rho_v] .= 0.0
                    Scythe.horizontal_si_correct!(sp, p, mo, hsd, t)
                    # the anchor's plane 4 (δρ_t) in B coefficients, as the deleted slot 3
                    # received it: the same write-back, onto a zero column of ρ_d's basis
                    spt = zeros(size(sp))
                    Scythe._hsi_add_increment!(spt, hsd, p, V["rho_d"], view(hsd.dlev, :, :, 4))
                    sprt = spt[:, V["rho_d"]]
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
                                         sprt[r]))
                        cmax = max(cmax, abs(sprt[r]))
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

    # ── 1e. a CLOUDY reference: the share-form z-terms are live ──────────────────
    # Every other fixture has a cloud-free base, where s_v = 1/α_v exactly and s_{v,z} = 0, so
    # the reference path's `s_{k,z}(1 − c_d) − s_k c_{d,z}` and exact_si's
    # `−Δτ s_{k,z}(1 − c_d) φ^{n+1}` were never exercised (and Σ α s_{k,z} = 0 hides a wrong
    # coefficient from the identity). Here the base carries a cloud layer rho_cbar(z) > 0 with a
    # gradient, and each species' increment is checked against the derivation computed
    # independently from the reference profiles, not only the sum.
    cl_rc(z) = 3.0e-4 * exp(-((z - 2250.0) / 700.0)^2)
    cloudy(tmpdir, tag; kw...) = seeded_tile(tmpdir, tag; rc_ref = cl_rc, cqss = false, kw...)
    # The reference shares and their derivatives, independently: s_k = ν̄_k/ρ̄_w with
    # ν̄_v = bhyp(ρ̄_v) (the tile's derived ρ̄_v), ν̄_c = bhyp(ρ̄_c), α = 2 for both.
    function shares(m)
        rcm = Springsteel.ref_rho_c(m.ref_state)
        rcb = rcm[:, 1]; rcbz = rcm[:, 2]
        mu_c = 1.0e-7
        rvb = m.mc_ref_diag.rho_vbar; rvbz = m.mc_ref_diag.rho_vbar_z
        nv = [Scythe.bhyp(x, VMU) for x in rvb]; nvz = [Scythe.dbhyp(x, VMU) for x in rvb] .* rvbz
        nc = [Scythe.bhyp(x, mu_c) for x in rcb]; ncz = [Scythe.dbhyp(x, mu_c) for x in rcb] .* rcbz
        W = 2 .* (nv .+ nc); Wz = 2 .* (nvz .+ ncz)
        sv = nv ./ W; svz = (nvz .* W .- nv .* Wz) ./ W .^ 2
        sc = nc ./ W; scz = (ncz .* W .- nc .* Wz) ./ W .^ 2
        # the derived reference ρ̄_t^w = ρ̄_d + W̄ (Stage 4: every reader's), from the same W
        rdb = Springsteel.ref_rho_d(m.ref_state)
        rt = rdb[:, 1] .+ W; rtz = rdb[:, 2] .+ Wz
        cd = rdb[:, 1] ./ rt
        cdz = (rdb[:, 2] .* rt .- rdb[:, 1] .* rtz) ./ rt .^ 2
        return (; sv, svz, sc, scz, cd, cdz, rt)
    end
    rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), floatmin())

    @testset "cloudy reference: reference vertical path, per species and the identity" begin
        mktempdir() do tmpdir
            m, p, mo = cloudy(tmpdir, "cl_ref";
                extra = Dict{Symbol,Any}(:si_water_slots => :mass, :state_dependent_si => false))
            @test maximum(Springsteel.ref_rho_c(m.ref_state)[:, 1]) > 1.0e-4
            F.vt_run!(m, p, mo, 3)
            V = mo.grid_params.vars
            sh = shares(m)
            @test maximum(abs, sh.scz) > 0.0 && maximum(abs, sh.svz) > 0.0   # the z-terms live
            kDim = mo.grid_params.kDim
            t = 4; tt = 1.25 * mo.ts
            rv = m.mc_slots.rho_v; members = mass_members(m)
            worst = 0.0; ev = 0.0; ec = 0.0
            T = zeros(size(p.physical, 1))
            for c in 1:ncols(p, mo)
                cs = (c - 1) * kDim + 1; ce = cs + kDim - 1
                Scythe.advance_column(m, c, t)
                for (k, _) in members; m.var_np1[cs:ce, k] .= 0.0; end
                m.var_np1[cs:ce, V["rho_d"]] .= 0.0
                Scythe.semiimplicit_adjustment_p(m, cs, ce, t; apply_histories = false)
                S = m.mc_scratch[Threads.threadid()]
                T[cs:ce] .= 0.0 .- (tt .* S.si_phi_z)
                worst = max(worst, first(identity_ratio(m.var_np1, cs:ce, V["rho_d"],
                                                        T, members)))
                phi = Scythe.scratch_column(m, V["w"]).uMish; phiz = S.si_phi_z
                om = 1.0 .- sh.cd
                for (k, s_, sz) in ((rv, sh.sv, sh.svz), (9, sh.sc, sh.scz))
                    ck = s_ .* om; ckz = sz .* om .- s_ .* sh.cdz
                    want = .-(tt .* (ck .* phiz .+ ckz .* phi))
                    e = rel(m.var_np1[cs:ce, k], want)
                    k == rv ? (ev = max(ev, e)) : (ec = max(ec, e))
                end
            end
            @info "cloudy reference vertical" worst ev ec
            @test worst <= 2.0
            @test ev <= 1.0e-12 && ec <= 1.0e-12
        end
    end

    @testset "cloudy reference: exact_si phase B, per species from the stored planes" begin
        mktempdir() do tmpdir
            m, p, mo = cloudy(tmpdir, "cl_xsi"; walls = true,
                extra = Dict{Symbol,Any}(:si_water_slots => :mass, :exact_si => true))
            esd = make_esd(m, p, mo)
            hfields = zeros(size(p.physical, 1), Scythe.XSI_NPLANES)
            for t in 1:3; xsi_step!(m, p, mo, esd, hfields, t); end
            V = mo.grid_params.vars
            sh = shares(m)
            members = mass_members(m); rv = m.mc_slots.rho_v
            t = 4; tt = 1.25 * mo.ts; kDim = mo.grid_params.kDim
            Scythe.exact_si_load_history!(m, hfields, t, 1)
            for c in 1:ncols(p, mo); Scythe.advance_column(m, c, t); end
            npts = size(p.physical, 1)
            xstar = hcat(m.var_np1[:, V["u"]], m.var_np1[:, V["w"]], m.var_np1[:, V["p"]])
            Scythe.exact_si_solve!(hfields, esd, p, mo, t, xstar)
            for (k, _) in members; m.var_np1[:, k] .= 0.0; end
            m.var_np1[:, V["rho_d"]] .= 0.0
            for c in 1:ncols(p, mo)
                cs = (c - 1) * kDim + 1
                Scythe.exact_si_apply_column!(m, cs, cs + kDim - 1, t, hfields, 1)
            end
            r, _, _ = identity_ratio(m.var_np1, 1:npts, V["rho_d"], 0.0 .+ hfields[:, 5], members)
            # δν_k = s_k (δρ_t − δρ_d) − Δτ s_{k,z} (1 − c_d) φ^{n+1}, φ^{n+1} = ρ̄_t w^{n+1}
            lev = [mod1(g, kDim) for g in 1:npts]
            dw = hfields[:, 5] .- hfields[:, 4]
            phin1 = sh.rt[lev] .* m.var_np1[:, V["w"]]
            corr = 0.0
            errs = map(((rv, sh.sv, sh.svz), (9, sh.sc, sh.scz))) do (k, s_, sz)
                ztrm = tt .* (sz[lev] .* (1.0 .- sh.cd[lev])) .* phin1
                corr = max(corr, maximum(abs, ztrm))
                rel(m.var_np1[:, k], s_[lev] .* dw .- ztrm)
            end
            @info "cloudy reference exact_si" r errs corr
            @test corr > 0.0                    # the flux-form correction is live
            @test r <= 2.0
            @test all(<=(1.0e-12), errs)
        end
    end

    @testset "cloudy reference: horizontal sweep, per species and the identity" begin
        mktempdir() do tmpdir
            m, p, mo = cloudy(tmpdir, "cl_hsi"; walls = true,
                extra = Dict{Symbol,Any}(:si_water_slots => :mass,
                                         :horizontal_semiimplicit => true))
            water = Scythe.si_water_hsi_profiles(m)
            @test water[1] == [m.mc_slots.rho_v, 9]      # vapor and cloud both carry a share
            sh = shares(m)
            @test rel(water[2][:, 1], sh.sv) <= 1.0e-14 && rel(water[2][:, 2], sh.sc) <= 1.0e-14
            hsd = Scythe.create_horizontal_solve_data(p, mo, m.mc_ref_diag.Pxi_prof,
                collect(view(rtbar(m), :, 1)),
                collect(view(Springsteel.ref_rho_d(m.ref_state), :, 1)),
                collect(view(Springsteel.ref_total_energy(m.ref_state), :, 1) .+
                        view(Springsteel.ref_pressure(m.ref_state), :, 1)); water)
            npts = size(p.physical, 1); V = mo.grid_params.vars
            u_incr = zeros(npts, 6)
            for t in 1:4
                t > 1 && Scythe.horizontal_si_load_increment!(m, u_incr, t, 1)
                for c in 1:ncols(p, mo); Scythe.advance_column(m, c, t); end
                Scythe.calcTendency(m)
                if t == 4
                    sp = p.spectral
                    for v in (V["rho_d"], m.mc_slots.rho_v, 9); sp[:, v] .= 0.0; end
                    Scythe.horizontal_si_correct!(sp, p, mo, hsd, t)
                    # the anchor's plane 4 in B coefficients (the deleted slot 3's write-back)
                    spt = zeros(size(sp))
                    Scythe._hsi_add_increment!(spt, hsd, p, V["rho_d"], view(hsd.dlev, :, :, 4))
                    sprt = spt[:, V["rho_d"]]
                    # the last water plane is the cloud's: s_c(z)·(δρ_t − δρ_d)
                    want = [sh.sc[k] * (hsd.dlev[i, k, 4] - hsd.dlev[i, k, 3])
                            for i in axes(hsd.wlev, 1), k in axes(hsd.wlev, 2)]
                    ec = rel(hsd.wlev, want)
                    cw = maximum(abs, (sp[:, V["rho_d"]] .+ 2 .* sp[:, m.mc_slots.rho_v] .+
                                       2 .* sp[:, 9]) .- sprt)
                    @info "cloudy reference hsi" ec cw
                    @test ec <= 1.0e-15
                    @test maximum(abs, sp[:, 9]) > 0.0
                    @test cw <= 1.0e-10 * maximum(abs, sprt)
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
                for v in ("p", "rho_d", "w", "E_t")
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
                rho_hat = sd ? S.rho_t : view(rtbar(m), :, 1)
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
                    rdb = Springsteel.ref_rho_d(m.ref_state); rtb = rtbar(m)
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
    @testset "a slaved set that does not share the anchor's fit operator is refused" begin
        mktempdir() do tmpdir
            msg(; kw...) = try
                siw_tile(tmpdir, "ref_$(hash(kw))"; kw...)
                ""
            catch e
                sprint(showerror, e)
            end
            on = Dict{Symbol,Any}(:si_water_slots => :mass)
            nat = NaturalBC()
            branches = [
                ("ground BC (BCB)", "nu_r", (; bcb_override = Dict{String,Any}("nu_r" => nat))),
                ("lid BC (BCT)", "nu_c", (; bct_override = Dict{String,Any}("nu_c" => nat))),
                ("inner/left BC (BCL)", "nu_v",
                 (; walls = true, side_override = Dict{String,Any}("nu_v" => nat))),
                ("vertical l_q", "nu_c", (; lq = Dict("default" => 2.0, "nu_c_k" => 1.0))),
                ("horizontal l_q", "nu_r",
                 (; lq = Dict("default" => 2.0, "nu_r" => 1.0, "nu_r_k" => 2.0))),
                # the ANCHOR's dry slot is compared too (its rho_t branch went with slot 3)
                ("vertical l_q", "rho_d", (; lq = Dict("default" => 2.0, "rho_d_k" => 1.0)))]
            for (label, v, kw) in branches
                err = msg(; extra = on, kw...)
                @test occursin("do not share one fit operator", err)
                @test occursin(label, err)
                @test occursin("\"$(v)\"", err)
                # :none never checks; the measurement opt-in builds
                @test msg(; extra = Dict{Symbol,Any}(:si_water_slots => :none), kw...) == ""
                @test msg(; extra = merge(on, Dict{Symbol,Any}(:si_water_mixed_walls => true)),
                          kw...) == ""
            end
            # spline_filter: Springsteel accepts no spline_filter on the RiRk directions, so the
            # branch is driven on a built model's grid parameters directly
            mt, _, mo = siw_tile(tmpdir, "sf"; extra = on)
            mo.grid_params.spline_filter["nu_c"] = Dict(:k => Springsteel.GaussianFilter(sigma = 2.0))
            err = try Scythe.validate_si_water_slots(mo); "" catch e; sprint(showerror, e) end
            @test occursin("spline_filter", err) && occursin("\"nu_c\"", err)
            # a positivity bound (an untransformed rho_r under positivity = auto)
            err = msg(; tr = :none, extra = on,
                      pos = Dict{String,Any}("rho_r" => Dict(:k => 0.0)))
            @test occursin("positivity bound", err) && occursin("\"rho_r\"", err)
            @test msg(; tr = :none, extra = Dict{Symbol,Any}(:si_water_slots => :none),
                      pos = Dict{String,Any}("rho_r" => Dict(:k => 0.0))) == ""
            # a uniform family builds
            @test msg(; extra = on) == ""
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
                                         view(rtbar(m), :, 1), view(rtbar(m), :, 2),
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

    # "feeder 1: the acoustic step no longer opens the water gap" was deleted at Stage 4 of the
    # rho_t retirement: it measured the partition gap rho_t - rho_d - sum alpha nu between the
    # prognostic slot 3 and the water sum the slots carry, which is identically zero now that
    # rho_t is derived from that sum.
end
