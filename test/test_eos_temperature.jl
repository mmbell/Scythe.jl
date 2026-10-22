# test/test_eos_temperature.jl -- the model temperature switch `options[:temperature]`
# (`numerics.temperature`): `:eos`, T = p/(rho_d R_d + rho_v R_v) from the prognostic pressure
# and vapor (the default since 2026-10-14), against `:energy`, the closed-form total-energy
# retrieval every consumer read before. The retrieval reads the condensates through their
# RECOVERED densities, which the `:bhyp` transforms clip at zero, while rho_t and E_t carry the
# signed lobe: wherever a lobe is clipped it sees liquid that is not there and warms by
# L_v/D per unit lobe, and the pressure (which the dynamics feels) does not. See
# `eos_temperature` in src/moist_compressible.jl.
#
# What has to hold:
#   1. the definitions (the EOS, the mode switch and its refusal, the implied energy);
#   2. `:energy` is BITWISE the pre-switch code (a golden capture at 9e81c33);
#   3. on a clear column the two temperatures agree, to the reference's fit error;
#   4. on a clipped lobe the retrieval warms by -L_v rho_t'/D and the EOS does not move;
#   5. the resting reference is a discrete fixed point under `:eos` (T = Tbar bitwise);
#   6. the output diagnostics `T_split`, `E_residual`, `column_E_residual` are what they say;
#   7. the column step stays allocation-free under `:eos`.
#
# The fixture is the Stage L one (test_lobe_relaxation.jl): a periodic XZ slice with 500 m
# vertical cells (the RCE's dz), every water slot under `:bhyp`, horizontally uniform seeds,
# here over a HYDROSTATIC reference so `:hydrostatic_reference` and `:consistent_qss_reference`
# — the production reference build — can be on.

using Test
using Scythe
using Springsteel
using SparseArrays
using Springsteel: PeriodicBC, NeumannBC, DirichletBC

isdefined(Main, :VaporTransformFixtures) || include("vapor_transform_fixtures.jl")

@testset "EOS temperature (options[:temperature])" begin
    EOS_F = Main.VaporTransformFixtures

    import Springsteel.Thermodynamics: rho_v_sat, internal_energy_bf02, L_v, L_s,
        Rd, Rv, Cpd, Cpv, Cl, Ci, gravity

    EOS_SPAN = 8.0e3
    EOS_NCELL = 4
    EOS_ZTOP = 10.0e3
    EOS_NCELL_K = 20
    EOS_MU = 1.0e-7

    """Reference column: 6.5 K/km lapse rate, RH `rh(z)`, cloud mixing ratio `q_l` (0 = clear),
    HYDROSTATIC (dp/dz = -g rho_t by RK4 on 10 m sub-steps), so `:hydrostatic_reference`
    accepts it."""
    function eos_column(z, rh; q_l = 0.0)
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

    """The periodic XZ tile. `temperature` is `options[:temperature]`, or `nothing` to leave
    the key absent (the default). `seed!(patch, pts, vars, col)` writes the perturbation
    state, which is then refit (spectral + grid transform) so the slots hold what the model's
    own refit makes of it, lobes included."""
    function eos_tile(tmpdir, tag; seed! = (p, x, V, col) -> nothing, temperature = nothing,
                      ice = false, tr = :bhyp, ts = 0.5, rh = z -> 0.7, q_l = 0.0,
                      cqss = true, hydro = true, extra = Dict{Symbol,Any}(),
                      pp_extra = Dict{Symbol,Float64}())
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :consistent_qss_reference => cqss,
                                :hydrostatic_reference => hydro,
                                :precipitation => ice, :vertical_mixing => false,
                                :condensate_transform => tr, :rain_transform => tr)
        temperature === nothing || (opts[:temperature] = temperature)
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
            iMin = 0.0, iMax = EOS_SPAN, num_cells_i = EOS_NCELL,
            kMin = 0.0, kMax = EOS_ZTOP, num_cells_k = EOS_NCELL_K,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
        ref_file = joinpath(tmpdir, "eos_$(tag).ref")
        pp = merge(Dict{Symbol,Float64}(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                                        :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                        :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3,
                                        :f => 0.0, :N_r => 1.0e-3), pp_extra)
        model = ModelParameters(ts = ts, integration_time = 10.0, output_interval = 10.0,
            equation_set = "moist_compressible_XZ", ref_state_file = ref_file,
            grid_params = gp, physical_params = pp, options = opts)
        gp = model.grid_params
        patch = createGrid(gp)
        pts = Scythe.getGridpoints(patch)
        z = pts[1:gp.kDim, end]
        col = eos_column(z, rh; q_l)
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

    eos_ncols(patch, model) = div(size(patch.physical, 1), model.grid_params.kDim)

    function eos_run!(m, p, mo, nsteps)
        for t in 1:nsteps
            for c in 1:eos_ncols(p, mo)
                Scythe.advance_column(m, c, t)
            end
            Scythe.calcTendency(m)
            gridTransform!(p)
        end
    end

    """A kinked cloud sheet 200-1150 m of amplitude `amp`, added at FIXED T: p unchanged (a
    condensate exerts no partial pressure), E_t carrying the cloud's BF02 energy
    rho_c (C_pv T - L_v(T) + g z) and the derived total rho_t = rho_d + sum_k alpha_k nu_k its
    mass (until Stage 4 of the rho_t retirement the mass was also written into the prognostic
    rho_t slot, which no reader of the derived anchor took). Where the sheet is resolved the
    state is therefore thermodynamically consistent; its top is sharper than one 500 m cell, so
    the refit rings the cloud slot (and with it the derived rho_t) and E_t below zero in the air
    above — a clipped lobe of exactly the kind the convective burst makes. `wamp` adds a smooth
    w."""
    eos_kink(z; zb = 200.0, zt = 1150.0, amp = 1.0e-3) =
        max(0.0, amp * (z - zb) * (zt - z) / ((zt - zb) / 2)^2)

    function eos_seed_sheet!(patch, pts, V, col; amp = 1.0e-3, wamp = 0.0)
        rc_i = Scythe.mc_slot(V, "rho_c")
        kDim = length(col.Tk)
        for i in 1:size(patch.physical, 1)
            z = pts[i, end]; k = mod1(i, kDim)
            rc = eos_kink(z; amp)
            T = col.Tk[k]
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rc, 0.0, :bhyp, EOS_MU)
            patch.physical[i, V["E_t"], 1] = rc * ((Cpv * T) - L_v(T) + (gravity * z))
            patch.physical[i, V["w"], 1] = wamp * sin(2pi * pts[i, 1] / EOS_SPAN) *
                                           sin(pi * z / EOS_ZTOP)
        end
    end

    "Order-sensitive exact checksum of a Float64 array (bit patterns, position-weighted)."
    eos_checksum(A) = sum(UInt128(reinterpret(UInt64, x)) * UInt128(j)
                          for (j, x) in enumerate(vec(A)))

    # ── 1. definitions ───────────────────────────────────────────────────────────────
    @testset "eos_temperature and the mode switch" begin
        for (p, rd, rv) in ((1.0e5, 1.15, 0.018), (3.0e4, 0.45, 1.0e-4), (8.0e4, 1.0, 0.0),
                            (5.0e4, 0.7, -2.0e-6))          # a negative (unfloored) vapor
            @test Scythe.eos_temperature(p, rd, rv) === p / ((rd * Rd) + (rv * Rv))
        end
        # The default is :eos; :energy is the other; anything else is refused loudly.
        @test Scythe.temperature_mode(Dict{Symbol,Any}()) === :eos
        @test Scythe.temperature_mode(Dict{Symbol,Any}(:temperature => :energy)) === :energy
        @test Scythe.temperature_mode(Dict{Symbol,Any}(:temperature => :eos)) === :eos
        @test_throws ErrorException Scythe.temperature_mode(Dict{Symbol,Any}(:temperature => :retrieval))
        @test_throws ErrorException Scythe.temperature_mode(Dict{Symbol,Any}(:temperature => "eos"))
        # model_temperature IS the two calls, bitwise.
        a = (2.6e5, 9.0e4, 1.05, 1.07, 0.012, 1.0e-3, 2.0e-4)
        M, p, rd, rt, rv, rl, ri = a
        @test Scythe.model_temperature(true, M, p, rd, rt, rv, rl, ri) ===
              Scythe.eos_temperature(p, rd, rv)
        @test Scythe.model_temperature(false, M, p, rd, rt, rv, rl, ri) ===
              Scythe.retrieve_temperature(M, rd, rt, rl, ri)
        # The config section, the option table and the build all carry it.
        @test haskey(Scythe.KNOWN_OPTION_KEYS, :temperature)
        @test Scythe.OWNED_OPTION_KEYS[:temperature] == "numerics.temperature"
        @test Scythe.Numerics().temperature === :eos
        @test Tuple(Scythe.TEMPERATURE_CHOICES) == Scythe.TEMPERATURE_MODES
    end

    @testset "implied_total_energy: a consistent state has T_eos == T_energy and no residual" begin
        # Forward-build (T, p, vapor, liquid, ice, winds, z) states in the model's own energy
        # definition; on such a state every temperature and the energy agree to rounding.
        for (Tk, p, sat, q_l, q_i, u, w, z) in ((300.0, 1.0e5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                                (285.0, 9.0e4, 1.0, 1.0e-3, 0.0, 3.0, 1.0, 1000.0),
                                                (250.0, 5.0e4, 0.9, 2.0e-4, 5.0e-4, 15.0, 5.0, 6000.0),
                                                (220.0, 2.5e4, 0.5, 0.0, 1.0e-4, 30.0, -2.0, 11000.0))
            rho_v = sat * rho_v_sat(Tk, p / 100.0)
            rho_d = (p - Rv * Tk * rho_v) / (Rd * Tk)
            rho_l = q_l * rho_d; rho_i = q_i * rho_d
            rho_t = rho_d + rho_v + rho_l + rho_i
            geo = 0.5 * (u^2 + w^2) + gravity * z
            E_t = Scythe.implied_total_energy(Tk, rho_d, rho_v, rho_l, rho_i, rho_t, geo)
            # It IS the reference build's definition (Springsteel `_pressure_reference`), plus
            # the retrieval's ice term.
            @test E_t ≈ rho_d * internal_energy_bf02(Tk, rho_v / rho_d, q_l) +
                        rho_i * (Cpv * Tk - L_s(Tk)) + rho_t * geo rtol = 1e-15
            M = p + E_t - rho_t * geo
            T_ret = Scythe.retrieve_temperature(M, rho_d, rho_t, rho_l, rho_i)
            T_eos = Scythe.eos_temperature(p, rho_d, rho_v)
            @test T_ret ≈ Tk rtol = 1e-12
            @test T_eos ≈ Tk rtol = 1e-14
            @test abs(E_t - Scythe.implied_total_energy(T_eos, rho_d, rho_v, rho_l, rho_i,
                                                        rho_t, geo)) < 1.0e-12 * abs(E_t)
            # A phantom liquid of 1e-3 kg/m^3 in the RETRIEVAL's input (the clipped lobe) warms
            # T_ret by L_v/D per kg/m^3 and moves neither T_eos nor the implied energy; the
            # residual is then D (T_ret - T_eos), the docstring's identity.
            dl = 1.0e-3
            T_ph = Scythe.retrieve_temperature(M, rho_d, rho_t, rho_l + dl, rho_i)
            D = (rho_d * Cpd + (rho_t - rho_d) * Cpv) - (rho_l + dl) * (Cpv - Cl) -
                rho_i * (Cpv - Ci)
            @test T_ph - Tk ≈ dl * L_v(Tk) / D rtol = 1e-9
            @test T_ph - Tk > 2.0                  # >= ~2 K per 1e-3 kg/m^3, more aloft
        end
    end

    # ── 2. :energy is the pre-switch code, bitwise ───────────────────────────────────
    @testset ":energy is bitwise bf577b2 under the derived anchor (golden capture)" begin
        # Three full steps of the seeded sheet with a w perturbation, warm and with ice, under
        # the production reference build (hydrostatic + consistent_qss) and the `:energy`
        # retrieval. First captured at 9e81c33, before `options[:temperature]` existed (that
        # capture, on the prognostic rho_t slot, is in git history); RE-SEEDED at bf577b2 under
        # the derived water anchor for Stage 4 of the rho_t retirement, and summed in the
        # Stage-4 layout (`s4_checksum`: slot 3, the passive shadow, left out). The cloud
        # evaporation bound, the exchange realization and the refit partner stay pinned off as
        # in the original capture (each is bitwise its own switch-off), so the fixture still
        # isolates the retrieval; the density vapor (`:vapor_transform => :none`) as before.
        golden = Dict("warm" => (UInt128(161122335957649786769422633),
                                 UInt128(4162737462651284037572250466)),
                      "ice" => (UInt128(457800712936917242225469130),
                                UInt128(16736897797215895292863035670)))
        got = Dict{String,Any}()
        mktempdir() do tmp
            for (tag, ice) in (("warm", false), ("ice", true))
                m, p, mo = eos_tile(tmp, tag; temperature = :energy, ice = ice,
                    extra = Dict{Symbol,Any}(:cloud_evap_realization => false,
                                             :exchange_realization => false,
                                             :refit_partner => false,
                                             :vapor_transform => :none),
                    seed! = (p, x, V, c) -> eos_seed_sheet!(p, x, V, c; wamp = 0.5))
                eos_run!(m, p, mo, 3)
                @test all(isfinite, m.var_np1)
                V = mo.grid_params.vars
                got[tag] = (EOS_F.s4_checksum(m.var_np1, V), EOS_F.s4_checksum(p.physical, V))
                @test got[tag][1] == golden[tag][1]
                @test got[tag][2] == golden[tag][2]
            end
        end
        get(ENV, "SCYTHE_PRINT_GOLDENS", "") == "1" &&
            foreach(k -> println("GOLDEN eos $(k) => $(got[k])"), sort(collect(keys(got))))
    end

    @testset "the driver's T is each mode's own formula, bitwise" begin
        mktempdir() do tmp
            for mode in (:energy, :eos)
                m, p, mo = eos_tile(tmp, "f_$(mode)"; temperature = mode,
                    seed! = (p, x, V, c) -> eos_seed_sheet!(p, x, V, c; wamp = 0.5))
                eos_run!(m, p, mo, 2)
                Scythe.advance_column(m, 2, 3)        # serial: the scratch is column 2's
                S = m.mc_scratch[Threads.threadid()]
                ref = mode === :eos ? Scythe.eos_temperature.(S.p, S.rho_d, S.rho_v) :
                      Scythe.retrieve_temperature.(S.M, S.rho_d, S.rho_t, S.rho_liq_t,
                                                   S.rho_ice_t)
                @test all(S.Tk .=== ref)
                # ...and the vapor it reads is the prognostic slot, unfloored.
                kDim = mo.grid_params.kDim
                rv_i = m.mc_slots.rho_v
                @test S.rho_v == p.physical[(kDim + 1):(2kDim), rv_i, 1] .+ m.mc_ref_diag.rho_vbar
            end
        end
    end

    # ── 3. a clear column ────────────────────────────────────────────────────────────
    @testset ":eos and :energy on a clear column: the reference fit, and the scheme" begin
        # No condensate anywhere and no clipped lobe. Pointwise, on a consistent state, the two
        # temperatures agree to rounding (testset "implied_total_energy" above). On the model's
        # FITTED reference they cannot: p̄, Ē_t, ρ̄_d and ρ̄_v are four independent spline fits of
        # one sounding, so the two resting temperatures differ by the fit error.
        #
        # (a) The PLAIN fitted reference (`hydrostatic = false`): fit error only, measured
        #     <= 6e-4 K on this 500 m column.
        # (b) The HYDROSTATIC reference re-integrates p̄ from the fitted ρ̄_t and keeps ρ̄_d, ρ̄_v
        #     and Ē_t, so its p̄ departs from the EOS of the sounding by the re-integration (up
        #     to Springsteel's 1 % gate). The retrieval reads that δp̄ at 1/(ρ C_p), the EOS at
        #     1/(ρ R) — 3.5 times harder — and the resting temperatures differ by up to ~1 K on
        #     this coarse 10 km column (measured -0.97 K at 9 km). That is a property of the
        #     reference build, not of the switch; under :eos the resting T is the EOS one, and
        #     `consistent_qss_reference` builds Q̄_ss on it, so rest is still exact (testset 5).
        # (c) A MOVING clear state (two steps of a smooth w): both formulas on the SAME state
        #     differ by (a) plus what the pressure and energy equations' discretizations
        #     disagree about — the non-conservation of the p equation that the output's
        #     `E_residual` measures. Measured 0.7 % of the perturbation after three steps.
        mktempdir() do tmp
            dTbar = Dict{Bool,Vector{Float64}}()
            for hydro in (false, true)
                Tb = Dict{Symbol,Vector{Float64}}()
                for mode in (:energy, :eos)
                    m, _, _ = eos_tile(tmp, "rest_$(hydro)_$(mode)"; temperature = mode,
                                       hydro = hydro)
                    Tb[mode] = copy(m.mc_ref_diag.Tbar)
                end
                dTbar[hydro] = Tb[:eos] .- Tb[:energy]
            end
            @test maximum(abs.(dTbar[false])) < 1.0e-3                  # (a)
            @test 0.1 < maximum(abs.(dTbar[true])) < 1.5                 # (b)

            m, p, mo = eos_tile(tmp, "clear_move"; hydro = false,
                seed! = (p, x, V, c) -> eos_seed_sheet!(p, x, V, c; amp = 0.0, wamp = 0.3))
            eos_run!(m, p, mo, 2)
            Scythe.advance_column(m, 2, 3)
            S = m.mc_scratch[Threads.threadid()]
            @test maximum(S.rho_liq) == 0.0 && minimum(S.nu_c) == 0.0   # clear, no lobe
            T_ret = Scythe.retrieve_temperature.(S.M, S.rho_d, S.rho_t, S.rho_liq_t, S.rho_ice_t)
            dT = S.Tk .- m.mc_ref_diag.Tbar
            dd = maximum(abs.((S.Tk .- T_ret) .- dTbar[false]))         # (c)
            @test maximum(abs.(dT)) > 1.0e-3                             # the state moved
            @test dd < 0.02 * maximum(abs.(dT))
        end
    end

    # ── 4. a clipped lobe ────────────────────────────────────────────────────────────
    @testset "a clipped lobe warms :energy and not :eos" begin
        mktempdir() do tmp
            res = Dict{Symbol,Any}()
            for mode in (:energy, :eos)
                m, p, mo = eos_tile(tmp, "lobe_$(mode)"; temperature = mode,
                                    seed! = eos_seed_sheet!)
                kDim = mo.grid_params.kDim
                Scythe.advance_column(m, 1, 1)
                S = m.mc_scratch[Threads.threadid()]
                # rho_t' is DERIVED (Stage 4 of the rho_t retirement): rho_d' + W', the
                # signed water sum of the slots, so the cloud slot's ringing lobe is in it.
                ws = Scythe.water_anchor_set(mo.options, mo.physical_params, m.mc_slots)
                rtp = [Scythe.anchor_rho_tp(p.physical[g, mo.grid_params.vars["rho_d"], 1],
                                            Scythe.anchor_Wp(p.physical, g, 1, ws))
                       for g in 1:kDim]
                D = (S.rho_d .* Cpd .+ (S.rho_t .- S.rho_d) .* Cpv) .-
                    S.rho_liq_t .* (Cpv - Cl)
                res[mode] = (dT = S.Tk .- m.mc_ref_diag.Tbar, rtp = rtp,
                             nu_c = copy(S.nu_c), rho_c = copy(S.rho_c), D = D,
                             Lv = L_v.(m.mc_ref_diag.Tbar),
                             z = Scythe.getGridpoints(p)[1:kDim, end])
            end
            e = res[:energy]; o = res[:eos]
            # The lobe: the refit rang the cloud slot (and so the derived rho_t) below zero above
            # the sheet; the recovery clips the slot to 0 there, rho_t keeps the lobe.
            lobe = findall((e.nu_c .< 0.0) .& (e.rtp .< -1.0e-6))
            @test length(lobe) >= 2
            @test all(e.rho_c[lobe] .== 0.0)
            # :energy credits the clipped lobe with phantom latent heat. The seed puts the
            # lobe's mass (through the cloud slot) AND its BF02 energy into rho_t and E_t, so the
            # retrieval sees
            # M' = rho_t'(C_pv T - L_v(T)) and C_f' = rho_t' C_pv with no liquid to hold the
            # latent part: dT = -L_v(Tbar) rho_t'/D > 0, D the state's own denominator. The
            # ringing of E_t and of rho_t are separate fits, hence the tolerance.
            pred = .-e.Lv[lobe] .* e.rtp[lobe] ./ e.D[lobe]
            @test all(e.dT[lobe] .> 0.0)
            @test all(isapprox.(e.dT[lobe], pred; rtol = 0.05))
            @test maximum(e.dT[lobe]) > 1.0e-2                # a measurable warming
            # :eos does not move AT ALL: p', rho_d' and the vapor slot are exactly zero, so
            # T == Tbar bit for bit at every level, lobe or not.
            @test all(o.dT .== 0.0)
        end
    end

    @testset "state_minima_trace reports max|T_split|" begin
        mktempdir() do tmp
            m, p, mo = eos_tile(tmp, "trace"; seed! = eos_seed_sheet!,
                                extra = Dict{Symbol,Any}(:state_minima_trace => 1))
            out = mktemp() do path, io
                redirect_stdout(io) do
                    Scythe.state_minima_trace(m, 1)
                end
                flush(io)
                read(path, String)
            end
            @test occursin("[T=eos]", out)
            mm = match(r"max\|T_split\|=([-0-9.e]+) K", out)
            @test mm !== nothing && parse(Float64, mm.captures[1]) > 1.0e-2
        end
    end

    # ── 5. the resting reference ─────────────────────────────────────────────────────
    @testset "resting reference: a discrete fixed point under :eos" begin
        mktempdir() do tmp
            for mode in (nothing, :eos, :energy)       # nothing = the default (:eos)
                m, p, mo = eos_tile(tmp, "rest_$(mode)"; temperature = mode)
                eos_run!(m, p, mo, 5)
                # EVERY prognostic slot stays EXACTLY zero: not small, zero.
                @test maximum(abs.(p.physical)) == 0.0
                @test maximum(abs.(m.var_np1)) == 0.0
                # The driver's T is the tile's resting Tbar, bitwise, and under :eos that is
                # the EOS temperature of (p̄, ρ̄_d, ρ̄_v) with the derived vapor.
                Scythe.advance_column(m, 1, 6)
                S = m.mc_scratch[Threads.threadid()]
                @test S.Tk == m.mc_ref_diag.Tbar
                if mode !== :energy
                    pbar = Springsteel.ref_pressure(m.ref_state)[:, 1]
                    rdbar = Springsteel.ref_rho_d(m.ref_state)[:, 1]
                    @test m.mc_ref_diag.Tbar == Scythe.eos_temperature.(pbar, rdbar,
                                                                        m.mc_ref_diag.rho_vbar)
                end
            end
            # A reference built for one mode is NOT a fixed point of the other: the builder
            # must take the run's option (that is why every caller passes it).
            m, p, mo = eos_tile(tmp, "cross"; temperature = :eos)
            ref_en = Scythe.consistent_qss_reference(
                Springsteel.exact_pressure_reference_state(mo.ref_state_file,
                    Scythe.getGridpoints(p)[1:mo.grid_params.kDim, end],
                    Scythe.reference_column(p, mo.grid_params); hydrostatic = true),
                Scythe.getGridpoints(p)[1:mo.grid_params.kDim, end],
                Scythe.reference_column(p, mo.grid_params); temperature = :energy)
            @test Springsteel.ref_qss(ref_en)[:, 1] != Springsteel.ref_qss(m.ref_state)[:, 1]
        end
        # A SATURATED (cloudy) base: the :eos Newton keeps the cloud on the Q_ss = 0 manifold
        # and the column holds to rounding (the same bar the :energy branch is held to in
        # test_moist_compressible.jl), with no transform (a cloudy reference refuses :bhyp).
        mktempdir() do tmp
            m, p, mo = eos_tile(tmp, "cloudy"; tr = :none, rh = z -> 1.0, q_l = 1.0e-3)
            @test all(Springsteel.ref_qss(m.ref_state)[:, 1] .== 0.0)
            @test all(Springsteel.ref_rho_c(m.ref_state)[:, 1] .> 0.0)
            # Saturated in the run's own (EOS) arithmetic, to the Newton tolerance.
            rs = m.ref_state
            Tb = m.mc_ref_diag.Tbar
            rvs = rho_v_sat.(Tb, Springsteel.ref_pressure(rs)[:, 1] ./ 100.0)
            @test maximum(abs.(m.mc_ref_diag.rho_vbar .- rvs) ./ rvs) < 1.0e-10
            eos_run!(m, p, mo, 5)
            V = mo.grid_params.vars
            @test maximum(abs.(p.physical[:, V["p"], 1])) < 1.0e-9
            @test maximum(abs.(p.physical[:, V["E_t"], 1])) < 1.0e-6
            # (no "rho_t": the slot was deleted at Stage 4; the derived total is rho_d plus the
            # water slots, every one of which is held here)
            for nm in ("rho_d", "u", "w", "Q_ss", "rho_r", "rho_c", "rho_v")
                @test maximum(abs.(p.physical[:, V[nm], 1])) < 1.0e-12
            end
        end
    end

    # ── 6. the output diagnostics ────────────────────────────────────────────────────
    @testset "NetCDF diagnostics: T_energy, T_eos, T_split, E_residual" begin
        # A 3-column state built CONSISTENTLY (E_t from the implied energy at the EOS T) plus a
        # clipped lobe: the cloud slot is a negative nu_c = -5e-5 (recovered 0) and the derived
        # rho_t = rho_d + rho_v + 2 nu_c carries its 2 nu_c = -1e-4. (Until Stage 4 of the rho_t
        # retirement the lobe was written into a prognostic rho_t slot directly.)
        z = collect(range(0.0, 12000.0; length = 13))
        n_i = 3; n_k = length(z)
        Tf(i, zz) = 300.0 - 0.0065 * zz + 0.5 * i
        p = [1.0e5 * exp(-z[k] / 8000.0) for i in 1:n_i, k in 1:n_k]
        rv = [0.7 * rho_v_sat(Tf(i, z[k]), p[i, k] / 100.0) for i in 1:n_i, k in 1:n_k]
        rd = [(p[i, k] - Rv * Tf(i, z[k]) * rv[i, k]) / (Rd * Tf(i, z[k])) for i in 1:n_i, k in 1:n_k]
        rt = rd .+ rv
        u = fill(3.0, n_i, n_k); w = fill(0.2, n_i, n_k)
        geo = 0.5 .* (u .^ 2 .+ w .^ 2) .+ gravity .* reshape(z, 1, n_k)
        Tk = [Tf(i, z[k]) for i in 1:n_i, k in 1:n_k]
        E = Scythe.implied_total_energy.(Tk, rd, rv, 0.0, 0.0, rt, geo)
        cl = zeros(n_i, n_k); cl[2, 7] = -5.0e-5
        lobe = 2.0 .* cl
        # The writer's own association (`mc_derived_fields`: rho_d .+ (rho_v .+ 2 nu_c) on the
        # flat zero reference), so the T_energy / E_residual comparisons below stay bitwise.
        rt_l = rd .+ (rv .+ lobe)
        E_l = E .+ lobe .* ((Cpv .* Tk) .- L_v.(Tk) .+ geo)       # the lobe's own energy
        zp = zeros(n_k)
        ref = Scythe.RegularReferenceProfiles(z, zp, zp, zp, zp, zp, zp, zp, zp)
        slots = (; p, rho_d = rd, E_t = E_l, Q_ss = zeros(n_i, n_k), rho_v = rv,
                   u, w, rain = zeros(n_i, n_k), cloud = cl)
        for mode in (:eos, :energy)
            cfg = Scythe.DerivedConfig(:bhyp, 1.0e-7, :bhyp, 1.0e-7, :none, 1.0, :none,
                                       (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16), 1, false, false,
                                       8.0e6, 100.0, "test", false, mode)
            d = Scythe.mc_derived_fields(z, slots, ref, cfg)
            @test all(d.T_eos .=== Scythe.eos_temperature.(p, rd, rv))
            @test all(d.T_energy .=== Scythe.retrieve_temperature.(d.M, rd, rt_l, d.rho_liq, 0.0))
            @test all(d.T .=== (mode === :eos ? d.T_eos : d.T_energy))
            @test all(d.T_split .=== d.T_energy .- d.T_eos)
            @test d.E_residual == E_l .- Scythe.implied_total_energy.(d.T_eos, rd, rv, 0.0, 0.0,
                                                                     rt_l, d.ke .+ gravity .*
                                                                     reshape(z, 1, n_k))
            @test d.column_E_residual == Scythe._column_integral(d.E_residual, z)
            @test d.column_E_t == Scythe._column_integral(E_l, z)
            # Consistent everywhere but the lobe point: split and residual vanish to rounding.
            mask = trues(n_i, n_k); mask[2, 7] = false
            @test maximum(abs.(d.T_split[mask])) < 1.0e-9
            @test maximum(abs.(d.E_residual[mask]) ./ abs.(E[mask])) < 1.0e-12
            # At the lobe the retrieval warms by -L_v lobe/D, and the residual obeys the exact
            # identity of `implied_total_energy`: D·T_split + X C_pv T_eos, with
            # X = rho_t - rho_d - rho_v - rho_liq - rho_ice the mass the gas+condensate
            # partition does not account for (here the lobe itself).
            Dl = (rd[2, 7] * Cpd + (rt_l[2, 7] - rd[2, 7]) * Cpv)
            @test d.T_split[2, 7] ≈ -L_v(Tk[2, 7]) * lobe[2, 7] / Dl rtol = 1.0e-9
            @test d.T_split[2, 7] > 0.2
            X = rt_l[2, 7] - rd[2, 7] - rv[2, 7]
            @test d.E_residual[2, 7] ≈ Dl * d.T_split[2, 7] + X * Cpv * d.T_eos[2, 7] rtol = 1.0e-9
            s = Scythe.temperature_split_summary(d, collect(1.0:n_i), z)
            @test s.max_abs == abs(d.T_split[2, 7]) && s.x_max == 2.0 && s.z_max == z[7]
            @test s.colres_mean ≈ sum(d.column_E_residual) / n_i
        end
    end

    # ── 7. allocation-free ───────────────────────────────────────────────────────────
    @testset "allocation-free column step under :eos (and the diffusion star state)" begin
        mktempdir() do tmp
            for ice in (false, true)
                m, p, mo = eos_tile(tmp, "alloc_$(ice)"; ice = ice,
                    seed! = (p, x, V, c) -> eos_seed_sheet!(p, x, V, c; wamp = 0.5),
                    pp_extra = Dict{Symbol,Float64}(:Kvdiff => 10.0, :Kvdiff_heat => 10.0))
                kDim = mo.grid_params.kDim
                Scythe.moist_compressible_XZ(m, 1, kDim, 2)          # compile
                @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 2)) == 0
                Scythe.diffusion_timestep_mc(m, 1, kDim, 2)
                @test (@allocations Scythe.diffusion_timestep_mc(m, 1, kDim, 2)) == 0
            end
        end
    end
end
