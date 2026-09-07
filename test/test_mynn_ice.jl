using Test
using Scythe
using Springsteel
using SparseArrays

# Tests for the MYNN-EDMF ice legs (src/mc_mynn_bl.jl, plan stage S8): the twelve ISHMAEL
# ice moments and the two-moment rain number mixed by the boundary layer, and the ice's
# share of the flux-form water energy carry.
#
# The idiom is test_mynn_bl.jl's and test_louis_bl.jl's: the SAME physical state is run
# through `advance_column` with the closure on and off and the explicit tendencies are
# differenced. Both arms here carry the FULL ice configuration, so the ISHMAEL sources,
# the sedimentation fluxes and the partition reconciliation are bitwise identical between
# them and cancel out of the difference exactly — what is left in an ice slot is the
# boundary layer's own leg and nothing else.

@testset "MYNN-EDMF ice legs (mc)" begin

    import Springsteel.Thermodynamics: Rd, Rv, Cpd, Cpv, gravity, L_s, L_v

    # ──────────────────────────────────────────────
    # Fixtures
    # ──────────────────────────────────────────────

    """Stably stratified dry column with an exact hydrostatic pressure for a linear
    temperature profile (test_mynn_bl.jl's, reproduced so the two files do not share a
    fixture that a later edit could move under one of them)."""
    function stable_column(z; T0 = 300.0, lapse = 0.004, p0 = 100000.0)
        Tk = @. T0 - lapse * z
        p_Pa = @. p0 * (Tk / T0)^(gravity / (Rd * lapse))
        rho_d = p_Pa ./ (Rd .* Tk)
        n = length(z)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    """Boundary-layer jet with zero value and zero slope at both ends (see
    test_mynn_bl.jl): real shear in the lowest ~2 zs, no momentum boundary term."""
    bl_jet(z, V0, zs) = @. V0 * (z / zs)^2 * exp(-z / zs)

    """Tile for an ice + MYNN arm, or the matching closure-off control.

    Every arm carries `:ice_microphysics = :ishmael` and the `:rain_moments = 2` it
    requires, so the ice slot indices are the same on both tiles and the ISHMAEL physics
    is identical between them. `condensation = false` switches off the phase changes —
    both the warm closure and `mc_ice_sources!`/`_ice_population_reconcile!` — which is
    what makes the zero-ice INERTNESS test below an equality rather than a tolerance:
    with the sources live, an ice-registered column nucleates and reconciles regardless of
    the boundary layer and the two configurations are not the same run.

    NEUMANN side walls for every variable, for the reason test_mynn_bl.jl gives: the
    `rho_e` slot has no partner to difference against and the test relies on its transport
    being identically zero."""
    function make_ice_tile(tmpdir; mynn = true, ice = true, num_cells_k = 25,
                           kMax = 25.0e3, ts = 0.5, Cd = -1.0, fluxes = true,
                           SST = 302.65, condensation = false, itrans = :none,
                           water_carry = :flux, mix_numbers = true, lapse = 0.004,
                           imu = (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16))
        opts_names = Dict{Symbol,Any}(:rain_moments => 2)
        ice && (opts_names[:ice_microphysics] = :ishmael)
        ice && (opts_names[:ice_transform] = itrans)
        mynn && (opts_names[:mynn] = true)
        varlist = Scythe.mc_var_names(opts_names; cyl = false)
        rain_name = Scythe.rain_var_name(opts_names)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = scalar_bc
        bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
        top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 24.0e3, num_cells_i = 4,
            kMin = 0.0, kMax = kMax, num_cells_k = num_cells_k,
            BCL = side_bc, BCR = side_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
        ref_file = joinpath(tmpdir,
            "mynn_ice_$(num_cells_k)_$(kMax)_$(mynn)_$(ice)_$(itrans).ref")
        options = Dict{Symbol,Any}(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => true,
                                   :condensation => condensation,
                                   :rain_moments => 2,
                                   :surface_fluxes => (mynn && fluxes))
        if ice
            options[:ice_microphysics] = :ishmael
            options[:ice_transform] = itrans
        end
        if mynn
            options[:mynn] = true
            options[:mynn_init] = :zero
            options[:mynn_interval] = 20.0
            options[:mynn_water_carry] = water_carry
            options[:mynn_mix_numbers] = mix_numbers
            options[:mynn_trace] = false
        end
        model = ModelParameters(
            ts = ts, integration_time = 10.0 * ts, output_interval = 10.0 * ts,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict{Symbol,Any}(
                :Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                :Kvdiff_water => 0.0, :tau_qss => 10.0, :alpha => 0.0,
                :z_damp => 20.0e3, :f => 0.0, :Cd => Cd, :Ls => 0.0,
                :Ck => 1.0e-3, :U_min => 1.0, :l_inf => 80.0, :SST => SST,
                :N_r => 1.0e-3, :mynn_K_max => Inf,
                :mu_ice => imu[1], :mu_ice_n => imu[2],
                :mu_ice_a => imu[3], :mu_ice_c => imu[4]),
            options = options)
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, end]
        col = stable_column(z; lapse = lapse)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model, gp, z, col
    end

    """3-point Gauss-Legendre cell weights on the RiRk mish (mubar = 3)."""
    function gauss_weights(kDim, num_cells_k, L)
        @assert kDim == 3 * num_cells_k
        dz = L / num_cells_k
        w = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0) .* dz
        return repeat(collect(w), num_cells_k)
    end

    "Advance every column once on a tile."
    function step_all!(mtile, patch, kDim, t = 1)
        ncols = div(size(patch.physical, 1), kDim)
        for c in 1:ncols
            Scythe.advance_column(mtile, c, t)
        end
        return ncols
    end

    """The wind, TKE and warm-water state test_mynn_bl.jl uses, plus the E_t' = rho_t ke
    compensation so the imposed wind is not also a thermal perturbation."""
    function set_warm!(patch, model, kDim, z, col; V0 = 12.0, zs = 500.0, e0 = 0.4,
                       ze = 1500.0, qpert = 2.0e-3)
        vars = model.grid_params.vars
        ui = vars["u"]; wi = vars["w"]; ei = vars["E_t"]
        rei = get(vars, "rho_e", 0)
        u = bl_jet(z, V0, zs)
        rho_t = col.rho_d .+ col.rho_v .+ col.rho_c
        npts = size(patch.physical, 1)
        for j in 1:npts
            k = mod1(j, kDim)
            patch.physical[j, ui, 1] = u[k]
            patch.physical[j, wi, 1] = 0.0
            bump = qpert * exp(-((z[k] - 800.0) / 600.0)^2)
            patch.physical[j, vars["rho_v"], 1] = rho_t[k] * bump
            patch.physical[j, vars["rho_t"], 1] = rho_t[k] * bump
            patch.physical[j, vars["rho_c"], 1] = rho_t[k] * 0.2 * bump
            rei > 0 && (patch.physical[j, rei, 1] = rho_t[k] * e0 * exp(-z[k] / ze))
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        for j in 1:npts
            k = mod1(j, kDim)
            ke = 0.5 * (patch.physical[j, ui, 1]^2 + patch.physical[j, wi, 1]^2)
            patch.physical[j, ei, 1] += rho_t[k] * ke
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        return nothing
    end

    """The ice blob of species 1, and the rain number.

    Within a species the four moments are the SAME profile times four constants, which is
    what makes the ratio assertion below sharp: mixing four extensive moments with one
    `K_h` on one (Neumann) basis gives four increments in the same proportion, so `rho/n`,
    `a/n` and `c/n` are unchanged. The amplitudes are all far ABOVE their transform widths
    (`mu_ice = 1e-12`, `mu_ice_n = 1e-2`, `mu_ice_a/c = 1e-16`), the regime where `bhyp`
    is exactly affine with slope 1/2 — the same regime test_louis_bl.jl's condensate
    transform test uses, and the reason the two ice-transform arms can be compared at
    1e-9 rather than at fit accuracy.

    `shape` is bounded away from zero and has zero end slope, so it is representable on
    the Neumann basis and the fit is not fighting the boundary conditions.

    The four amplitudes are a CONSISTENT crystal population — `rho/n = 1e-8 kg` per
    particle and `a_i/n = c_i/n = <a^2 c> = 2.6e-12 m^3`, which is that mass at the density
    of ice — so the state the ISHMAEL fall speeds and habit machinery see is a real
    distribution and not a 260x-overdense one.

    `tf` turns a DENSITY into the slot value, so the same profile can be seeded on the
    untransformed and on the transformed tile."""
    function set_ice!(patch, model, kDim, z, col;
                      amp = (1.0e-4, 1.0e4, 2.6e-8, 2.6e-8),
                      H = 20.0e3, n_r = 1.0e3, tf = :none,
                      imu = (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16))
        vars = model.grid_params.vars
        IS = Scythe.mc_slots(model)
        # Two DIFFERENT profiles, so species 1 and species 2 cannot be confused for one
        # another by a mis-wired slot index, and species 3 is left empty as the control
        # that an absent species costs exactly zero.
        shape(zk) = 1.0 + (0.5 * cos(pi * zk / H))
        shape2(zk) = 1.0 + (0.4 * cos(2.0 * pi * zk / H))
        slots = (IS.i1_q, IS.i1_n, IS.i1_a, IS.i1_c)
        slots2 = (IS.i2_q, IS.i2_n, IS.i2_a, IS.i2_c)
        npts = size(patch.physical, 1)
        for j in 1:npts
            k = mod1(j, kDim)
            sh = shape(z[k])
            sh2 = shape2(z[k])
            for m in 1:4
                patch.physical[j, slots[m], 1] =
                    Scythe.total_slot(amp[m] * sh, tf, imu[m])
                patch.physical[j, slots2[m], 1] =
                    Scythe.total_slot(0.3 * amp[m] * sh2, tf, imu[m])
            end
            patch.physical[j, IS.n_r, 1] = n_r * sh
            # The ice mass goes into rho_t and E_t too, not only into the ice slots: ice
            # above the headroom `(rho_t - rho_d) - rho_liq` is a DETACHED partition, the
            # anchor share drops below 1 and the state the test is measuring on is one the
            # reconciliation is actively undoing. `C_pv T - L_s(T) + g z` is the model's
            # own ice specific energy (the `E_sed_i` coefficient) and is exactly what
            # leaves `retrieve_temperature` invariant when mass is added to `rho_t` and
            # `rho_ice` together, so the sounding's temperature is unchanged by the blob.
            rho_i = (amp[1] * sh) + (0.3 * amp[1] * sh2)
            patch.physical[j, vars["rho_t"], 1] += rho_i
            patch.physical[j, vars["E_t"], 1] +=
                (((Cpv * col.Tk[k]) - L_s(col.Tk[k])) + (gravity * z[k])) * rho_i
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        return nothing
    end

    """The fitted `K_h ∂z(density)` flux divergence of one TOTAL-form moment slot, built
    here rather than read out of the model: the slot's OWN spline column, the model's own
    fitted slot gradient divided by the slot's Jacobian, times `K_h` at the mish."""
    function direct_leg(mtile, patch, slot, Kh, rng, tf, mu)
        nu = patch.physical[rng, slot, 1]
        nu_z = patch.physical[rng, slot, 4]          # zslot(MCCartesianXZ) == 4
        J = tf === :none ? ones(length(rng)) :
            [Scythe.dbhyp(max(Scythe.recover_total(nu[i], tf, mu), 0.0), mu)
             for i in eachindex(nu)]
        c = Scythe.scratch_column(mtile, slot)
        c.uMish .= Kh .* (nu_z ./ J)
        Btransform!(c)
        Atransform!(c)
        out = zeros(length(rng))
        Ixtransform(c, out)
        return out, J
    end

    # ──────────────────────────────────────────────
    # 1. Warm inertness: zero ice must not perturb anything
    # ──────────────────────────────────────────────
    @testset "ice registered but identically zero: warm increments are ===" begin
        mktempdir() do tmp
            m_ice, p_ice, mo_i, gpo, z, col = make_ice_tile(tmp; mynn = true, ice = true)
            m_no, p_no, mo_n, _, _, _ = make_ice_tile(tmp; mynn = true, ice = false)
            kDim = gpo.kDim
            set_warm!(p_ice, mo_i, kDim, z, col)
            set_warm!(p_no, mo_n, kDim, z, col)
            step_all!(m_ice, p_ice, kDim)
            step_all!(m_no, p_no, kDim)
            # Slots 1..11 are the nine fixed slots, rho_v and n_r, and they carry the same
            # index on both tiles (the twelve ice slots are appended after n_r and rho_e
            # after them). The ice legs are exact zeros in every one of them, and that is
            # an EQUALITY: `x - 0.0 === x` for every double, `x + 0.0` is not, and the S8
            # arithmetic is written so that only the first form appears.
            for s in 1:11
                @test all(m_ice.expdot_n[:, s] .=== m_no.expdot_n[:, s])
            end
            @test all(m_ice.expdot_n[:, mo_i.grid_params.vars["rho_e"]] .===
                      m_no.expdot_n[:, mo_n.grid_params.vars["rho_e"]])
            # ...and the closure itself saw the same column: with rho_ice_t == 0 the
            # `q_i` handed to theta_l and the cloud PDF is an exact zero.
            @test maximum(abs.(m_ice.mynn.K_h .- m_no.mynn.K_h)) == 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 2. The twelve ice legs and the rain number
    # ──────────────────────────────────────────────
    @testset "ice moments are mixed as the fitted K_h dz flux divergence" begin
        # A STRONG TKE (`e0 = 60 m^2/s^2`, `K_h ~ 37 m^2/s`) on purpose. The boundary
        # layer's ice leg is being read as the difference between two tendencies whose
        # OTHER, identical, ice terms are seven orders of magnitude larger, so the
        # difference is only as accurate as `ulp` of those terms: at the default
        # `e0 = 0.4` the cancellation floor is ~7e-9 relative and nothing tighter can be
        # asserted honestly. Thirty times the diffusivity buys back the digits — the
        # measured agreement below is ~2e-11 — and the leg is linear in `K_h`, so nothing
        # about what is being tested changes.
        mktempdir() do tmp
            e0 = 60.0
            m_on, p_on, mo, gpo, z, col = make_ice_tile(tmp; mynn = true, ice = true)
            m_off, p_off, mf, _, _, _ = make_ice_tile(tmp; mynn = false, ice = true)
            kDim = gpo.kDim
            for (pp, mm) in ((p_on, mo), (p_off, mf))
                set_warm!(pp, mm, kDim, z, col; e0 = e0)
                set_ice!(pp, mm, kDim, z, col)
            end
            step_all!(m_on, p_on, kDim)
            step_all!(m_off, p_off, kDim)
            IS = Scythe.mc_slots(mo)
            c = 3
            rng = ((c - 1) * kDim + 1):(c * kDim)
            Kh = m_on.mynn.K_h[rng]
            @test maximum(Kh) > 0.0                 # the closure is actually mixing

            D = m_on.expdot_n[:, 1:size(m_off.expdot_n, 2)] .- m_off.expdot_n
            islots = (IS.i1_q, IS.i1_n, IS.i1_a, IS.i1_c,
                      IS.i2_q, IS.i2_n, IS.i2_a, IS.i2_c,
                      IS.i3_q, IS.i3_n, IS.i3_a, IS.i3_c)
            imu = (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16,
                   1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16,
                   1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16)
            legs = Vector{Vector{Float64}}(undef, 12)
            worst_leg = 0.0
            for (m, sl) in enumerate(islots)
                expect, _ = direct_leg(m_on, p_on, sl, Kh, rng, :none, imu[m])
                legs[m] = expect
                got = D[rng, sl]
                sc = maximum(abs.(expect))
                sc == 0.0 && continue
                rel = maximum(abs.(got .- expect)) / sc
                worst_leg = max(worst_leg, rel)
                @test rel < 1.0e-9
            end
            @info "MYNN ice: worst leg-vs-direct-fit residual" rel = worst_leg
            # Species 1 and 2 carry blobs of DIFFERENT shape (so a mis-wired slot index
            # would show up as one species receiving the other's divergence, which the
            # per-slot fit comparison above would catch); species 3 is empty and its four
            # legs are exact zeros, which is what says an absent species costs nothing.
            @test maximum(abs.(D[rng, IS.i1_q])) > 0.0
            @test maximum(abs.(D[rng, IS.i2_q])) > 0.0
            for sl in (IS.i3_q, IS.i3_n, IS.i3_a, IS.i3_c)
                @test all(D[rng, sl] .== 0.0)
            end

            # The rain NUMBER, on the same K_h and its own column.
            nr_exp, _ = direct_leg(m_on, p_on, IS.n_r, Kh, rng, :none, 1.0)
            @test maximum(abs.(nr_exp)) > 0.0
            @test maximum(abs.(D[rng, IS.n_r] .- nr_exp)) <
                  1.0e-9 * maximum(abs.(nr_exp))

            # THE MEAN PARTICLE DOES NOT MOVE. The four moments were seeded as one profile
            # times four constants, so mixing them with one K_h on one basis must leave
            # rho/n, a/n and c/n unchanged: d(X/n) = (dX·n - X·dn)/n^2, and the numerator
            # is zero when dX/dn == X/n.
            rho_i = p_on.physical[rng, IS.i1_q, 1]
            n_i   = p_on.physical[rng, IS.i1_n, 1]
            a_i   = p_on.physical[rng, IS.i1_a, 1]
            c_i   = p_on.physical[rng, IS.i1_c, 1]
            dn = D[rng, IS.i1_n]
            worst = 0.0
            for (X, dX) in ((rho_i, D[rng, IS.i1_q]), (a_i, D[rng, IS.i1_a]),
                            (c_i, D[rng, IS.i1_c]))
                num = (dX .* n_i) .- (X .* dn)
                rel = maximum(abs.(num)) / maximum(abs.(dX .* n_i))
                worst = max(worst, rel)
                @test rel < 1.0e-9
            end
            @info "MYNN ice: worst moment-ratio drift" rel = worst
        end
    end

    # ──────────────────────────────────────────────
    # 3. The control-variable transform
    # ──────────────────────────────────────────────
    @testset "ice_transform = :bhyp mixes the DENSITY and carries the Jacobian" begin
        mktempdir() do tmp
            arms = Dict{Symbol,Any}()
            for tf in (:none, :bhyp)
                m_on, p_on, mo, gpo, z, col = make_ice_tile(tmp; mynn = true, ice = true,
                                                            itrans = tf)
                m_off, p_off, mf, _, _, _ = make_ice_tile(tmp; mynn = false, ice = true,
                                                          itrans = tf)
                kDim = gpo.kDim
                for (pp, mm) in ((p_on, mo), (p_off, mf))
                    set_warm!(pp, mm, kDim, z, col; e0 = 60.0)
                    set_ice!(pp, mm, kDim, z, col; tf = tf)
                end
                step_all!(m_on, p_on, kDim)
                step_all!(m_off, p_off, kDim)
                arms[tf] = (m_on.expdot_n[:, 1:size(m_off.expdot_n, 2)] .-
                            m_off.expdot_n, Scythe.mc_slots(mo), kDim, m_on, p_on)
            end
            Dn, IS, kDim, m_n, p_n = arms[:none]
            Db, _, _, m_b, p_b = arms[:bhyp]
            c = 3
            rng = ((c - 1) * kDim + 1):(c * kDim)
            # Every seeded moment is far above its own `mu`, where `bhyp` is affine with
            # slope 1/2 exactly (test_louis_bl.jl's condensate-transform regime). So the
            # SLOT increment is half the untransformed one — the Jacobian, applied once
            # and in the right direction — and the DENSITY increment (slot rate over J)
            # is the same number in both arms.
            islots = (IS.i1_q, IS.i1_n, IS.i1_a, IS.i1_c)
            for sl in islots
                sc = maximum(abs.(Dn[rng, sl]))
                @test sc > 0.0
                @test isapprox(Db[rng, sl], 0.5 .* Dn[rng, sl]; rtol = 1.0e-9,
                               atol = 1.0e-9 * sc)
            end
            # The warm slots see the same water and the same energy: the ice's `-L_f`
            # share of `S_Ew` is built from the DENSITY gradient in both arms.
            for s in (1, 3, 6, 7, 9)
                sc = maximum(abs.(Dn[rng, s]))
                @test isapprox(Db[rng, s], Dn[rng, s]; rtol = 1.0e-9,
                               atol = 1.0e-9 * sc + 1.0e-300)
            end
        end
    end

    # ──────────────────────────────────────────────
    # 4. :mynn_mix_numbers
    # ──────────────────────────────────────────────
    @testset "mynn_mix_numbers = false leaves n_r alone and nothing else" begin
        mktempdir() do tmp
            m_a, p_a, mo_a, gpo, z, col = make_ice_tile(tmp; mynn = true, ice = true)
            m_b, p_b, mo_b, _, _, _ = make_ice_tile(tmp; mynn = true, ice = true,
                                                    mix_numbers = false)
            kDim = gpo.kDim
            for (pp, mm) in ((p_a, mo_a), (p_b, mo_b))
                set_warm!(pp, mm, kDim, z, col)
                set_ice!(pp, mm, kDim, z, col)
            end
            step_all!(m_a, p_a, kDim)
            step_all!(m_b, p_b, kDim)
            IS = Scythe.mc_slots(mo_a)
            @test maximum(abs.(m_a.expdot_n[:, IS.n_r] .-
                               m_b.expdot_n[:, IS.n_r])) > 0.0
            # Everything else, the twelve ice moments included, is untouched by the knob:
            # it gates the rain NUMBER and nothing more.
            for s in 1:size(m_a.expdot_n, 2)
                s == IS.n_r && continue
                @test all(m_a.expdot_n[:, s] .=== m_b.expdot_n[:, s])
            end
        end
    end

    # ──────────────────────────────────────────────
    # 5. The D3 column energy identity with the ice carry
    # ──────────────────────────────────────────────
    @testset "column energy budget still closes with an ice blob" begin
        mktempdir() do tmp
            for nk in (25, 50), ts in (0.5, 0.25)
                m_on, p_on, mo, gpo, z, col = make_ice_tile(tmp; mynn = true, ice = true,
                                                            num_cells_k = nk, ts = ts)
                m_off, p_off, mf, _, _, _ = make_ice_tile(tmp; mynn = false, ice = true,
                                                          num_cells_k = nk, ts = ts)
                kDim = gpo.kDim
                for (pp, mm) in ((p_on, mo), (p_off, mf))
                    set_warm!(pp, mm, kDim, z, col)
                    set_ice!(pp, mm, kDim, z, col)
                end
                step_all!(m_on, p_on, kDim)
                step_all!(m_off, p_off, kDim)
                re_i = mo.grid_params.vars["rho_e"]
                wq = gauss_weights(kDim, nk, gpo.kMax - gpo.kMin)
                c = 3
                rng = ((c - 1) * kDim + 1):(c * kDim)
                dE = m_on.expdot_n[rng, 6] .- m_off.expdot_n[rng, 6]
                de = m_on.expdot_n[rng, re_i]
                lhs = sum(wq .* (dE .+ de))
                rhs = m_on.mynn.bdry_E[c]
                scale = sum(wq .* abs.(dE)) + sum(wq .* abs.(de))
                @test isfinite(lhs - rhs)
                @test abs(lhs - rhs) <= 1.0e-12 * scale
            end
        end
    end

    # ──────────────────────────────────────────────
    # 6. The refusals that are left
    # ──────────────────────────────────────────────
    @testset "ice + :fixed_T water carry is refused; ice + MYNN is not" begin
        mktempdir() do tmp
            m, p, mo, gpo, z, col = make_ice_tile(tmp; mynn = true, ice = true,
                                                  water_carry = :fixed_T)
            set_warm!(p, mo, gpo.kDim, z, col)
            set_ice!(p, mo, gpo.kDim, z, col)
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
            # ...while the default :flux carry runs.
            m2, p2, mo2, gp2, z2, col2 = make_ice_tile(tmp; mynn = true, ice = true)
            set_warm!(p2, mo2, gp2.kDim, z2, col2)
            set_ice!(p2, mo2, gp2.kDim, z2, col2)
            step_all!(m2, p2, gp2.kDim)
            @test all(isfinite, m2.expdot_n)
        end
    end
end
