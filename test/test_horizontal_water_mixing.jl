# test/test_horizontal_water_mixing.jl -- the horizontal water-species mixing leg of
# `mc_driver!` (`physical_params[:Khdiff_water]`, `physics.horizontal_mixing.water`): EVERY
# water species mixes with ONE coefficient K_water = K/Sc_t (Smagorinsky or constant) --
# total water (slot 3), the prognostic vapor, cloud and rain (and the rain number), and
# under ISHMAEL all three ice species' mass AND number and habit moments, so crystals are
# not rescaled by the operator.
#
# Every assertion is on the DIFFERENCE of two tiles advanced from the same state with the
# mixing on and off, so advection, the pressure gradient, buoyancy and every microphysical
# source cancel identically and what is left is the mixing itself. The tile is a periodic
# XZ slice (the RCE geometry), so the horizontal integral of a Laplacian has no wall flux to
# hide behind.

using Test
using Scythe
using Springsteel
using SparseArrays
using Springsteel: PeriodicBC, NeumannBC, DirichletBC

@testset "horizontal water-species mixing" begin

    # ── fixture ──────────────────────────────────────────────────────────────────────
    const_L = 48.0e3                      # periodic span [m]: 24 cells of 2 km
    NCELL = 24
    DX = const_L / NCELL
    MU_C = 1.0e-7                         # cloud / rain widths (the shipped defaults)
    MU_NR = 1.0                           # rain-number width (the shipped default)

    """Moist, CLOUD-FREE, subsaturated reference column (RH 70 %), cold enough aloft for
    ice. Cloud-free so every condensate slot is its own total (bhyp(0) = 0)."""
    function hwm_column(z)
        Tk = @. 300.0 - 0.0065 * z
        p_Pa = @. 100000.0 * exp(-z / 8000.0)
        rho_v = 0.7 .* Scythe.rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Scythe.Rv .* Tk .* rho_v)) ./ (Scythe.Rd .* Tk)
        return (; Tk, p_Pa, rho_d, rho_v, rho_c = zeros(length(z)))
    end

    gauss(x, x0, s) = exp(-((x - x0) / s)^2)

    # Per-crystal mass [kg], volume moments a, c per crystal [m^3] of the three species.
    # The seeds put q = m n, a = alpha n, c = gamma n with the same spatial shape, so the
    # ratios q/n, a/n, c/n are uniform -- and a mixing that rescales crystals shows up as a
    # non-uniform ratio after one step.
    ICE_M = (1.0e-10, 3.0e-11, 5.0e-10)
    ICE_A = (1.25e-13, 8.0e-14, 4.0e-13)
    ICE_C = (2.5e-13, 6.0e-14, 9.0e-13)

    """A periodic XZ RiRk tile with ISHMAEL ice, two-moment rain and every water slot
    transformed by `tr` (`:none` or `:bhyp`), seeded with sharp vapor, cloud, rain and ice
    features that carry x-structure, and a periodic strain u = 5 sin(2 pi x/L)."""
    function hwm_tile(tmpdir, tag; tr = :bhyp, ice = true, Khdiff_water = 0.0,
                      Sc_t = 1.0, C_s = 0.0, K_min = 5.0, ts = 0.5, seed = true)
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :precipitation => true, :vertical_mixing => false,
                                :rain_moments => 2,
                                :condensate_transform => tr, :rain_transform => tr,
                                :rain_number_transform => tr)
        if ice
            opts[:ice_microphysics] = :ishmael
            opts[:ice_transform] = tr
        end
        varnames = Scythe.mc_var_names(opts)
        vars = Dict(v => i for (i, v) in enumerate(varnames))
        side_bc = Dict{String,Any}(v => PeriodicBC() for v in varnames)
        wall_bc = merge(Dict{String,Any}(v => NeumannBC() for v in varnames),
                        Dict{String,Any}("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = const_L, num_cells_i = NCELL,
            kMin = 0.0, kMax = 10.0e3, num_cells_k = 10,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
        ref_file = joinpath(tmpdir, "hwm_$(tag).ref")
        pp = Dict{Symbol,Float64}(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                                  :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                  :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3,
                                  :f => 0.0, :N_r => 1.0e-3, :K_min => K_min,
                                  :Khdiff_water => Khdiff_water, :Sc_t => Sc_t)
        C_s > 0.0 && (pp[:C_s] = C_s)
        model = ModelParameters(ts = ts, integration_time = 10.0, output_interval = 10.0,
            equation_set = "moist_compressible_XZ", ref_state_file = ref_file,
            grid_params = gp, physical_params = pp, options = opts)
        gp = model.grid_params
        patch = createGrid(gp)
        pts = Scythe.getGridpoints(patch)
        z = pts[1:gp.kDim, end]
        col = hwm_column(z)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        if seed
            V = gp.vars
            rv_i = Scythe.mc_slot(V, "rho_v"); rc_i = Scythe.mc_slot(V, "rho_c")
            rr_i = Scythe.mc_slot(V, "rho_r"); nr_i = Scythe.mc_slot(V, "n_r")
            ice_i = ice ? Scythe.mc_ice_slot_indices(V) : ntuple(_ -> 0, 12)
            x0 = 0.5 * const_L
            for i in 1:size(patch.physical, 1)
                x = pts[i, 1]; zz = pts[i, end]
                patch.physical[i, V["u"], 1] = 5.0 * sin(2pi * x / const_L)
                # sharp features: 2-3 km wide against 2 km cells
                rv = 1.0e-3 * gauss(x, x0, 3.0e3) * gauss(zz, 3000.0, 1000.0)
                rc = 5.0e-4 * gauss(x, x0 + 2.0e3, 2.0e3) * gauss(zz, 4000.0, 700.0)
                rr = 3.0e-4 * gauss(x, x0 - 3.0e3, 2.0e3) * gauss(zz, 1500.0, 700.0)
                nr = rr / 5.0e-7
                patch.physical[i, rv_i, 1] = rv
                patch.physical[i, V["Q_ss"], 1] = -1.0e-5 * gauss(x, x0, 3.0e3) *
                                                  gauss(zz, 3000.0, 1000.0)
                patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rc, 0.0, tr, MU_C)
                patch.physical[i, rr_i, 1] = Scythe.rain_slot(rr, tr, MU_C)
                patch.physical[i, nr_i, 1] = Scythe.rain_number_slot(nr, tr, MU_NR)
                ri_sum = 0.0
                if ice
                    for k in 1:3
                        q = (1.0e-4 / k) * gauss(x, x0 + (k - 2) * 4.0e3, 2.5e3) *
                            gauss(zz, 6500.0 + 500.0 * k, 800.0)
                        n = q / ICE_M[k]
                        vals = (q, n, ICE_A[k] * n, ICE_C[k] * n)
                        for j in 1:4
                            jj = 4 * (k - 1) + j
                            patch.physical[i, ice_i[jj], 1] =
                                Scythe.total_slot(vals[j], tr,
                                                  Scythe.MC_ICE_MU_DEFAULTS[jj])
                        end
                        ri_sum += q
                    end
                end
                # Total water carries every species, so the partition starts closed.
                patch.physical[i, V["rho_t"], 1] = rv + rc + rr + ri_sum
            end
        end
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

    """expdot(on) − expdot(off) from the same state: the mixing contribution, plus the on
    tile's handles."""
    function mix_increment(tmpdir, tag; Kw, kw...)
        m1, p1, mo1 = hwm_tile(tmpdir, "$(tag)_on"; Khdiff_water = Kw, kw...)
        m0, p0, mo0 = hwm_tile(tmpdir, "$(tag)_off"; Khdiff_water = 0.0, kw...)
        advance_all!(m1, p1, mo1)
        advance_all!(m0, p0, mo0)
        return m1.expdot_n .- m0.expdot_n, m1, p1, mo1
    end

    """The comparison floor for slot `s`: the difference D is taken between two whole
    tendencies, so it carries a few ulps of THEIR magnitude (advection, sources), not of
    the mixing term's."""
    tol(m, s, expect) = 1.0e-12 * maximum(abs.(expect)) +
                        1.0e-13 * maximum(abs.(m.expdot_n[:, s]))

    """Horizontal Gauss-Legendre weights of the x-mish (3 points per cell), per gridpoint."""
    function xweights(patch, model)
        kDim = model.grid_params.kDim
        gw = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0) .* DX
        return [gw[mod1(div(i - 1, kDim) + 1, 3)] for i in 1:size(patch.physical, 1)]
    end

    """The water slots the leg mixes, by role (slot 3 is handled separately)."""
    function water_slots(model; ice = true)
        V = model.grid_params.vars
        s = [Scythe.mc_slot(V, r) for r in ("rho_v", "rho_c", "rho_r", "n_r", "Q_ss")]
        ice && append!(s, collect(Scythe.mc_ice_slot_indices(V)))
        return s
    end

    @testset "every species receives K/Sc_t times its own Laplacian (constant K)" begin
        mktempdir() do tmp
            K = 2000.0
            for tr in (:none, :bhyp)
                D, m, p, mo = mix_increment(tmp, "kc_$(tr)"; Kw = K, tr = tr)
                @test all(isfinite, D)
                V = mo.grid_params.vars
                for s in water_slots(mo)
                    expect = K .* p.physical[:, s, 3]
                    sc = maximum(abs.(expect))
                    @test sc > 0.0                       # the seed really has x-structure
                    @test isapprox(D[:, s], expect; rtol = 0.0, atol = tol(m, s, expect))
                end
                # Slot 3 mixes WATER only: rho_t' - rho_d' (dry air is not mixed).
                e3 = K .* (p.physical[:, V["rho_t"], 3] .- p.physical[:, V["rho_d"], 3])
                @test isapprox(D[:, V["rho_t"]], e3; rtol = 0.0,
                               atol = tol(m, V["rho_t"], e3))
                # Nothing else moves: no dry air, momentum, energy, pressure.
                for nm in ("rho_d", "u", "w", "E_t", "p")
                    @test all(D[:, V[nm]] .== 0.0)
                end
            end
        end
    end

    @testset "Smagorinsky: one K_smag/Sc_t for every species, from the tile's own u_x" begin
        mktempdir() do tmp
            Sc_t = 0.7
            D, m, p, mo = mix_increment(tmp, "ks"; Kw = -1.0, C_s = 0.2, Sc_t = Sc_t)
            V = mo.grid_params.vars
            Kexp = max.((0.2 * DX)^2 * sqrt(2.0) .* abs.(p.physical[:, V["u"], 2]), 5.0)
            @test maximum(Kexp) > 10.0 * 5.0          # the strain lifts K off the floor
            for s in water_slots(mo)
                expect = (Kexp ./ Sc_t) .* p.physical[:, s, 3]
                @test isapprox(D[:, s], expect; rtol = 0.0, atol = tol(m, s, expect))
            end
            # The sentinel without a length is still refused (Ls = 0 and no C_s).
            mk, pk, mok = hwm_tile(tmp, "ks_nolen"; Khdiff_water = -1.0)
            @test_throws ErrorException Scythe.advance_column(mk, 1, 1)
        end
    end

    @testset "per-species mass closure under periodic x" begin
        # ∫ K ∂xx φ dx = 0 for constant K on a periodic span; the spline's φ_xx is piecewise
        # linear, so the 3-point Gauss mish integrates it exactly and the discrete sum is
        # zero to rounding, LEVEL BY LEVEL. Under `:none` the slot IS the density, so this
        # is the mass of each species; for variable K it is ∫ K φ_xx = −∫ K_x φ_x ≠ 0, the
        # known non-conservation of the K∇² (rather than ∇·K∇) form, and is reported.
        # Integrated on the operator K φ_xx itself, which the first two testsets show is
        # what each slot receives: the on-off difference D carries ulps of the whole
        # tendency (Q_ss's is O(1e4) larger than its mixing), which would measure the
        # subtraction rather than the operator.
        mktempdir() do tmp
            function closure(Kv, p, mo, slots)
                w = xweights(p, mo)
                kDim = mo.grid_params.kDim
                V = mo.grid_params.vars
                worst = 0.0
                for s in slots, k in 1:kDim
                    idx = k:kDim:size(p.physical, 1)
                    phixx = s == V["rho_t"] ?
                        p.physical[idx, s, 3] .- p.physical[idx, V["rho_d"], 3] :
                        p.physical[idx, s, 3]
                    E = Kv[idx] .* phixx
                    gross = sum(w[idx] .* abs.(E))
                    gross > 0.0 && (worst = max(worst, abs(sum(w[idx] .* E)) / gross))
                end
                return worst
            end
            _, _, p, mo = mix_increment(tmp, "mc_const"; Kw = 2000.0, tr = :none)
            slots = vcat(water_slots(mo), mo.grid_params.vars["rho_t"])
            rel_const = closure(fill(2000.0, size(p.physical, 1)), p, mo, slots)
            @test rel_const < 1.0e-12
            _, _, p, mo = mix_increment(tmp, "mc_smag"; Kw = -1.0, C_s = 0.2, tr = :none)
            Kexp = max.((0.2 * DX)^2 * sqrt(2.0) .* abs.(p.physical[:, mo.grid_params.vars["u"], 2]), 5.0)
            rel_smag = closure(Kexp, p, mo, slots)
            @info "horizontal water mixing: worst per-level |∫K∂xxφ| / ∫|K∂xxφ| over all species" rel_const rel_smag
            @test rel_smag > 1.0e3 * rel_const       # variable K: K∇² is not flux form
            @test rel_smag < 1.0                     # ...but it is still a mixing
        end
    end

    @testset "ice moments mix with the mass: crystals are not rescaled" begin
        # One forward step of the mixing ALONE (the on-off difference): q, n, a, c of each
        # species start in fixed ratio with the same shape, so with one K the ratios q/n,
        # a/n, c/n must stay uniform. Under `:none` that is exact to rounding (the operator
        # is linear and the four Laplacians are proportional). Under `:bhyp` the slots are
        # ν and the identity holds to the affine accuracy of bhyp above the knee.
        mktempdir() do tmp
            dt = 50.0
            for (tr, tol) in ((:none, 1.0e-12), (:bhyp, 1.0e-3))
                for (tag, kw) in (("c", (Kw = 2000.0,)), ("s", (Kw = -1.0, C_s = 0.2)))
                    D, _, p, mo = mix_increment(tmp, "im_$(tr)_$(tag)"; tr = tr, kw...)
                    ii = Scythe.mc_ice_slot_indices(mo.grid_params.vars)
                    worst = 0.0
                    for k in 1:3
                        sl = ii[4k-3:4k]
                        mus = Scythe.MC_ICE_MU_DEFAULTS[4k-3:4k]
                        new = [Scythe.recover_total.(p.physical[:, sl[j], 1] .+
                                                     dt .* D[:, sl[j]], tr, mus[j])
                               for j in 1:4]
                        @test maximum(abs.(D[:, sl[1]])) > 0.0
                        # Only where the crystals are meaningfully present: 1e-3 of the peak.
                        live = new[2] .> 1.0e-3 * maximum(new[2])
                        for (j, ref) in ((1, ICE_M[k]), (3, ICE_A[k]), (4, ICE_C[k]))
                            r = new[j][live] ./ new[2][live]
                            worst = max(worst, maximum(abs.(r ./ ref .- 1.0)))
                        end
                    end
                    @info "ice per-crystal ratio drift after a mixing-only step" tr tag worst
                    @test worst < tol
                end
            end
        end
    end

    @testset "the prognostic vapor is mixed: the plume spreads" begin
        mktempdir() do tmp
            D, m, p, mo = mix_increment(tmp, "rv"; Kw = 2000.0, tr = :none, ice = false)
            rv = Scythe.mc_slot(mo.grid_params.vars, "rho_v")
            phi = p.physical[:, rv, 1]
            imax = argmax(phi)
            @test D[imax, rv] < 0.0                  # the peak drains...
            # ...into the horizontal flanks (same level, a few km out): perturbation form,
            # the reference ρ̄_v(z) has no x-structure and contributes nothing.
            kDim = mo.grid_params.kDim
            flank = imax + 3 * 3 * kDim               # three cells further in x
            @test phi[flank] < phi[imax] && D[flank, rv] > 0.0
            e = 2000.0 .* p.physical[:, rv, 3]
            @test isapprox(D[:, rv], e; rtol = 0.0, atol = tol(m, rv, e))
        end
    end

    @testset "the species sum stays on slot 3: rho_v + Σ condensates = rho_t − rho_d" begin
        # gap = (ρ_t' − ρ_d') − (ρ_v' + ρ_c + ρ_r + Σρ_i): with every species mixed by the
        # same K, the mixing's contribution to the gap is zero under `:none` (one linear
        # operator on a sum). Under `:bhyp` slot 3 mixes in DENSITY and the condensates in
        # ν, and the recovered density rate is f'(ν)·K∇²ν: equal to K∇²ρ to O(μ²/ρ²) above
        # the knee, but ZERO where the spline fit of a sharp feature rings into ν < 0 (`ahyp`
        # is floored there, f' = 0). At those LOBE points the mass slot 3 receives does not
        # reach any species — it goes into lifting the negative ν toward zero, i.e. into
        # shrinking the lobe pool Σ 2 min(ν, 0) — so the gap is measured separately on and
        # off the lobes, and the pool is checked to shrink. Without the vapor term — the leg
        # as it was — the gap would be the whole vapor Laplacian (`old`).
        mktempdir() do tmp
            dt = 50.0
            for tr in (:none, :bhyp)
                D, _, p, mo = mix_increment(tmp, "gap_$(tr)"; Kw = 2000.0, tr = tr)
                V = mo.grid_params.vars
                ii = Scythe.mc_ice_slot_indices(V)
                mass = [(Scythe.mc_slot(V, "rho_c"), MU_C), (Scythe.mc_slot(V, "rho_r"), MU_C)]
                for k in 1:3
                    push!(mass, (ii[4k-3], Scythe.MC_ICE_MU_DEFAULTS[4k-3]))
                end
                nu(s, d) = p.physical[:, s, 1] .+ d .* D[:, s]
                function gap(d)
                    g = nu(V["rho_t"], d) .- nu(V["rho_v"], d)
                    for (s, mu) in mass
                        g .-= Scythe.recover_total.(nu(s, d), tr, mu)
                    end
                    return g
                end
                # Under `:none` nothing is floored, so there is no lobe class: every point
                # must close to rounding.
                lobe = falses(size(D, 1))
                if tr !== :none
                    for (s, _) in mass
                        lobe .|= (nu(s, 0.0) .< 0.0) .| (nu(s, dt) .< 0.0)
                    end
                end
                dgap = gap(dt) .- gap(0.0)
                moved = maximum(dt .* abs.(D[:, V["rho_t"]]))
                rel = maximum(abs.(dgap[.!lobe])) / moved
                rel_lobe = any(lobe) ? maximum(abs.(dgap[lobe])) / moved : 0.0
                old = maximum(abs.(dt .* D[:, V["rho_v"]])) / moved
                pool(d) = sum(sum(2.0 .* min.(nu(s, d), 0.0)) for (s, _) in mass)
                @info "species-sum gap from one mixing step / max water moved" tr rel rel_lobe old n_lobe = count(lobe) pool0 = pool(0.0) pool1 = pool(dt)
                @test old > 0.1                          # the vapor term is not negligible
                # `:none`: rounding. `:bhyp`, off the lobes: the knee points (0 < ν ≲ μ, where
                # f' runs from 1 to 2) carry an ABSOLUTE error of order μ per step — the O(μ)
                # convention error, measured 0.3 μ here (1.4e-3 of the water moved).
                if tr === :none
                    @test rel < 1.0e-12
                else
                    @test maximum(abs.(dgap[.!lobe])) < MU_C
                end
                if tr === :bhyp
                    @test any(lobe)                      # the sharp seeds do ring
                    @test pool(dt) > pool(0.0)           # mixing shrinks the lobe pool
                end
            end
        end
    end

    @testset "the species-sum gap does not grow over several full steps" begin
        mktempdir() do tmp
            nsteps = 10
            res = Dict{Float64,Float64}()
            for Kw in (0.0, 2000.0)
                m, p, mo = hwm_tile(tmp, "gs_$(Kw)"; Khdiff_water = Kw, tr = :bhyp)
                V = mo.grid_params.vars
                ii = Scythe.mc_ice_slot_indices(V)
                function gapmax()
                    g = p.physical[:, V["rho_t"], 1] .- p.physical[:, V["rho_d"], 1] .-
                        p.physical[:, V["rho_v"], 1] .-
                        Scythe.recover_total.(p.physical[:, V["nu_c"], 1], :bhyp, MU_C) .-
                        Scythe.recover_total.(p.physical[:, V["nu_r"], 1], :bhyp, MU_C)
                    for k in 1:3
                        g .-= Scythe.recover_total.(p.physical[:, ii[4k-3], 1], :bhyp,
                                                    Scythe.MC_ICE_MU_DEFAULTS[4k-3])
                    end
                    return maximum(abs.(g))
                end
                g0 = gapmax()
                for t in 1:nsteps
                    advance_all!(m, p, mo; t = t)
                    Scythe.calcTendency(m)
                    gridTransform!(p)
                end
                @test all(isfinite, p.physical[:, :, 1])
                res[Kw] = gapmax() - g0
            end
            @info "species-sum gap growth over 10 steps [kg/m^3]" off = res[0.0] on = res[2000.0]
            # The mixing adds at most O(μ) per step of gap on top of what the unmixed run
            # already carries (advection of each species on its own fit, the nudges).
            @test abs(res[2000.0] - res[0.0]) < nsteps * MU_C
        end
    end

    @testset "water = false is bitwise inert; ice + water builds from config" begin
        mktempdir() do tmp
            D, _, _, _ = mix_increment(tmp, "inert"; Kw = 0.0)
            @test all(D .== 0.0)
            # The absent key is the same as 0.0.
            m1, p1, mo1 = hwm_tile(tmp, "absent")
            delete!(mo1.physical_params, :Khdiff_water)
            m0, p0, mo0 = hwm_tile(tmp, "zero")
            advance_all!(m1, p1, mo1); advance_all!(m0, p0, mo0)
            @test m1.expdot_n == m0.expdot_n
            # Config: water = true with ISHMAEL is no longer refused.
            owb = joinpath(pkgdir(Scythe), "simulations", "ocean_warm_bubble",
                           "scythe_config.toml")
            cfg = load_config(owb; set = ["output.dir=$(tmp)",
                                          "physics.horizontal_mixing.scheme=smagorinsky",
                                          "physics.horizontal_mixing.water=true"])
            @test cfg.physics.microphysics.ice === :ishmael
            m = build_model(cfg)
            @test m.physical_params[:Khdiff_water] == -1.0
        end
    end
end
