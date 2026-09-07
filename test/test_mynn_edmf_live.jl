using Test
using Scythe
using Springsteel
using SparseArrays

# Tests for the LIVE EDMF mass-flux coupling (src/mc_mynn_bl.jl, plan stage S7 / D6):
# `options[:mynn_edmf] = 1`, `DMP_mf` on the closure cadence, and the plume term folded
# into the same fitted flux columns the eddy diffusivity uses.
#
# test/test_mynn_edmf.jl is the PARITY test of `dmp_mf!` itself against the Fortran; this
# file never looks at a Fortran number. What it pins is the COUPLING:
#
#   1. turning the plumes on changes NOTHING, bit for bit, on a column where they do not
#      fire -- which is the only way a knob this size can be trusted;
#   2. on a column where they DO fire the D3 column energy identity still closes to
#      round-off, because the mass flux rides inside the fitted columns rather than beside
#      them (src/mc_mynn_bl.jl, `_mynn_add_mf!`);
#   3. the plume heat flux has the sign a rising plume has: it cools the layer it leaves
#      and warms the one it reaches;
#   4. and how thin the first model layer has to be before the plumes leave the ground at
#      all -- printed, not asserted, because it is a property of the grid.

@testset "MYNN-EDMF mass flux, live (mc)" begin

    import Springsteel.Thermodynamics: rho_v_sat, Rd, Rv, Cpd, Cpv, gravity

    # ──────────────────────────────────────────────
    # Fixtures
    # ──────────────────────────────────────────────

    """Stably stratified dry column, exactly test_mynn_bl.jl's: `p = p0 (T/T0)^{g/(R_d
    Gamma)}` for a linear `T`."""
    function stable_column(z; T0 = 300.0, lapse = 0.004, p0 = 100000.0)
        Tk = @. T0 - lapse * z
        p_Pa = @. p0 * (Tk / T0)^(gravity / (Rd * lapse))
        rho_d = p_Pa ./ (Rd .* Tk)
        n = length(z)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    """Convective column: a DRY ADIABATIC mixed layer to `zi` under a stable free
    troposphere, hydrostatic in closed form in each branch and continuous across the
    inversion. This is what a mass-flux scheme is for -- with `dtheta/dz = 0` below `zi` a
    surface plume keeps its buoyancy as it entrains, which the stable fixture above denies
    it by construction.

    `lapse_ml` defaults to `gravity/Cpd` EXACTLY, the model's own constants, so `theta` is
    constant to round-off rather than to two digits. That is not fussiness: the plume's
    surface excess is `exc_fac*w_up*sigma_th/sigma_w`, tens of a kelvin at a 200 W/m^2
    surface flux, and a mixed layer that is off the adiabat by even 1e-3 K/m eats it
    within one model layer -- which is exactly how the first attempt at this fixture
    (lapse 0.0098, `theta` rising 7e-4 K/m) stalled every plume at the first wall. The
    reference column of the Fortran harness, `columns/case4_convective.txt`, carries
    `theta = 300.0` to every printed digit through 1 km for the same reason.

    `dlapse` is how far ABOVE the dry adiabat the mixed layer sits: `1e-3 K/m` is
    `dtheta/dz ~ -1 K/km`, a genuinely heated convective layer rather than the neutral
    limit. It is not decoration either. At `dlapse = 0` (exactly neutral) whether the
    plumes leave the ground is a knife-edge on the buoyancy of the WEAKEST of the eight at
    the first wall, and the one that fails takes the whole column with it (`nup2 = 0`,
    :6288, harness README item 12): measured on this fixture, 25 and 200 cells fire and 50
    and 100 do not, with no monotone dependence on anything. At `1e-3` every spacing from
    25 to 200 cells fires with `Sigma_aw ~ 0.16 m/s`, so the resolution study below is
    reading the FIRST LAYER THICKNESS rather than that knife-edge."""
    function convective_column(z; T0 = 300.0, zi = 1500.0, dlapse = 1.0e-3,
                               lapse_ml = (gravity / Cpd) + dlapse, lapse_ft = 0.004,
                               p0 = 100000.0)
        n = length(z)
        Tk = similar(z); p_Pa = similar(z)
        T_i = T0 - lapse_ml * zi
        p_i = p0 * (T_i / T0)^(gravity / (Rd * lapse_ml))
        @inbounds for k in 1:n
            if z[k] <= zi
                Tk[k] = T0 - lapse_ml * z[k]
                p_Pa[k] = p0 * (Tk[k] / T0)^(gravity / (Rd * lapse_ml))
            else
                Tk[k] = T_i - lapse_ft * (z[k] - zi)
                p_Pa[k] = p_i * (Tk[k] / T_i)^(gravity / (Rd * lapse_ft))
            end
        end
        rho_d = p_Pa ./ (Rd .* Tk)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    bl_jet(z, V0, zs) = @. V0 * (z / zs)^2 * exp(-z / zs)

    # The convective arm's surface forcing. `surface_exchange` gives
    # `F_sh = rho_d C_pd C_k U (T_s - T_1)`, so `C_k = 1.2e-3` with `U_min = 10 m/s` over a
    # sea 15 K above the lowest mish level is ~210 W/m^2 -- the 200 W/m^2 of the harness's
    # `case4_convective`, reached through Scythe's own bulk layer rather than prescribed.
    # `CONV_QPERT` is deliberately 10x smaller than the stable fixture's: the water bump
    # has to make the four water legs live without tilting `theta_v` by more than the
    # plume's own excess (2e-3 of vapour is 0.36 K of virtual temperature, which would
    # rebuild the stable layer the mixed layer exists to remove).
    CONV_CK = 1.2e-3;
    CONV_U = 10.0;
    CONV_QPERT = 2.0e-4;

    """Tile for the EDMF arm, or the matching `:mynn_edmf = 0` control. The grid, the
    reference and every physical parameter are identical between the two -- only
    `options[:mynn_edmf]` differs -- so a difference in `expdot` is the plumes and
    nothing else."""
    function make_edmf_tile(tmpdir; edmf = 1, edmf_mom = true, num_cells_k = 50,
                            kMax = 25.0e3, ts = 0.5, Cd = -1.0, fluxes = true,
                            SST = 302.65, init = :zero, interval = 20.0,
                            profile = :stable, lapse = 0.004, zi = 1500.0,
                            Ck = 1.0e-3, U_min = 1.0, dlapse = 1.0e-3)
        opts_names = Dict{Symbol,Any}(:mynn => true)
        varlist = Scythe.mc_var_names(opts_names; cyl = false)
        rain_name = Scythe.rain_var_name(opts_names)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = scalar_bc                       # see test_mynn_bl.jl for WHY Neumann
        bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
        top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 24.0e3, num_cells_i = 4,
            kMin = 0.0, kMax = kMax, num_cells_k = num_cells_k,
            BCL = side_bc, BCR = side_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
        ref_file = joinpath(tmpdir,
            "mynn_edmf_$(num_cells_k)_$(kMax)_$(profile)_$(lapse)_$(zi)_$(dlapse).ref")
        options = Dict{Symbol,Any}(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => false,
                                   :surface_fluxes => fluxes,
                                   :mynn => true,
                                   :mynn_init => init,
                                   :mynn_interval => interval,
                                   :mynn_edmf => edmf,
                                   :mynn_edmf_mom => edmf_mom,
                                   :mynn_trace => false,
                                   # See the identical note in test_mynn_bl.jl's fixture:
                                   # `:mynn_output` defaults to true (S9) and this fixture
                                   # has no `output_dir` of its own.
                                   :mynn_output => false)
        model = ModelParameters(
            ts = ts, integration_time = 10.0 * ts, output_interval = 10.0 * ts,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict{Symbol,Any}(
                :Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                :Kvdiff_water => 0.0, :tau_qss => 10.0, :alpha => 0.0,
                :z_damp => 20.0e3, :f => 0.0, :Cd => Cd, :Ls => 0.0,
                :Ck => Ck, :U_min => U_min, :l_inf => 80.0, :SST => SST,
                :mynn_K_max => Inf),
            options = options)
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, end]
        col = profile === :stable ? stable_column(z; lapse = lapse) :
                                    convective_column(z; zi = zi, dlapse = dlapse)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model, gp, z, col
    end

    """3-point Gauss-Legendre cell weights on the RiRk mish (mubar = 3), reimplemented
    here so the budget assertion shares no code with the model."""
    function gauss_weights(kDim, num_cells_k, L)
        @assert kDim == 3 * num_cells_k
        dz = L / num_cells_k
        w = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0) .* dz
        return repeat(collect(w), num_cells_k)
    end

    "Impose u(z), rho_e(z) and the E_t' = rho_t ke compensation (test_mynn_bl.jl's)."
    function set_state!(patch, model, kDim, z, col; V0 = 12.0, zs = 500.0, e0 = 0.4,
                        ze = 1500.0, W0 = 0.0, qpert = 2.0e-3, qshape = :bump)
        vars = model.grid_params.vars
        ui = vars["u"]; wi = vars["w"]; ei = vars["E_t"]
        rei = vars["rho_e"]
        u = bl_jet(z, V0, zs)
        rho_t = col.rho_d .+ col.rho_v .+ col.rho_c
        ztop = model.grid_params.kMax
        npts = size(patch.physical, 1)
        for j in 1:npts
            k = mod1(j, kDim)
            patch.physical[j, ui, 1] = u[k]
            patch.physical[j, wi, 1] = W0 * sin(pi * z[k] / ztop)
            # `:bump` is the S5 fixture's mid-BL Gaussian, which makes every water leg
            # live; `:decreasing` is the profile a boundary layer actually has -- water
            # maximum at the surface -- and the convective arm needs it, because with the
            # bump the environment is MOISTER above the plume than at its root, the plume
            # water flux points DOWN, and the sign of the transport says more about the
            # fixture than about the scheme.
            bump = qshape === :bump ? qpert * exp(-((z[k] - 800.0)/600.0)^2) :
                                      qpert * exp(-z[k]/2000.0)
            patch.physical[j, vars["rho_v"], 1] = rho_t[k] * bump
            patch.physical[j, vars["rho_t"], 1] = rho_t[k] * bump
            patch.physical[j, vars["rho_c"], 1] = rho_t[k] * 0.2 * bump
            patch.physical[j, vars["rho_r"], 1] = rho_t[k] * 0.1 * bump
            patch.physical[j, rei, 1] = rho_t[k] * e0 * exp(-z[k] / ze)
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

    function step_all!(mtile, patch, kDim, t = 1)
        ncols = div(size(patch.physical, 1), kDim)
        for c in 1:ncols
            Scythe.advance_column(mtile, c, t)
        end
        return ncols
    end

    # ──────────────────────────────────────────────
    # 1. Off/on inertness: no plume, no difference — bitwise
    # ──────────────────────────────────────────────
    @testset "no plume fires: :mynn_edmf = 1 is BITWISE :mynn_edmf = 0" begin
        mktempdir() do tmp
            # (a) The S5 resting fixture: sea at the sounding's own T_1, no wind, no TKE.
            _, _, _, _, z0, col0 = make_edmf_tile(tmp; edmf = 0, num_cells_k = 25)
            T1 = col0.Tk[1]
            R = Matrix{Float64}[]
            for ed in (0, 1)
                m, p, mo, gp, z, col = make_edmf_tile(tmp; edmf = ed, num_cells_k = 25,
                                                      SST = T1)
                step_all!(m, p, gp.kDim)
                push!(R, copy(m.expdot_n))
                ed == 1 && (@test m.mynn.edmf == 1)
            end
            @test size(R[1]) == size(R[2])
            @test all(i -> R[1][i] === R[2][i], eachindex(R[1]))

            # (b) The stable sheared fixture with weak fluxes: the closure is fully live
            # (K_m, K_h > 0, every leg nonzero) and the plumes still never fire, because a
            # stably stratified column is not superadiabatic. This is the case that
            # actually exercises `_mynn_add_mf!` -- it runs, on exact zeros, and must be
            # the identity.
            E = Matrix{Float64}[]
            for ed in (0, 1)
                m, p, mo, gp, z, col = make_edmf_tile(tmp; edmf = ed, num_cells_k = 50,
                                                      SST = T1 + 0.05)
                set_state!(p, mo, gp.kDim, z, col; V0 = 12.0, e0 = 0.4)
                step_all!(m, p, gp.kDim)
                push!(E, copy(m.expdot_n))
                if ed == 1
                    @test all(iszero, m.mynn.s_aw)
                    @test all(iszero, m.mynn.s_aw_st)
                    @test maximum(m.mynn.plume_ktop) == 0
                end
                @test maximum(m.mynn.K_h) > 0.0          # the closure IS live
            end
            @test all(i -> E[1][i] === E[2][i], eachindex(E[1]))
        end
    end

    # ──────────────────────────────────────────────
    # 2. A plume-firing column
    # ──────────────────────────────────────────────
    @testset "a convective column fires plumes, and the D3 budget still closes" begin
        mktempdir() do tmp
            # 100 cells over 25 km: a ~77 m first model layer, the order of the TC
            # nests' own lowest layer. The resolution study below is where that choice
            # comes from -- at 638 m (12 cells) every plume stalls at the first wall and
            # the column produces nothing, which is a statement about the grid rather than
            # about the scheme.
            nk = 100
            _, _, _, _, zc, colc = make_edmf_tile(tmp; edmf = 0, num_cells_k = nk,
                                                  profile = :convective)
            SST = colc.Tk[1] + 15.0        # ~210 W/m^2 through the model's own bulk layer
            res = Dict{Int,Any}()
            for ed in (0, 1)
                m, p, mo, gp, z, col = make_edmf_tile(tmp; edmf = ed, num_cells_k = nk,
                                                      SST = SST, profile = :convective,
                                                      Ck = CONV_CK, U_min = CONV_U)
                kDim = gp.kDim
                set_state!(p, mo, kDim, z, col; V0 = 6.0, e0 = 0.4,
                           qpert = CONV_QPERT, qshape = :decreasing)
                step_all!(m, p, kDim)
                res[ed] = (copy(m.expdot_n), m, mo, gp, kDim)
            end
            E0, m0, _, gp0, kDim = res[0]
            E1, m1, mo1, _, _ = res[1]

            # ...the plumes fired.
            @test maximum(m1.mynn.plume_ktop) > 0
            @test maximum(m1.mynn.aw_max) > 0.0
            @test maximum(m1.mynn.plume_ztop) > 0.0
            @test sum(m1.mynn.n_plume_col) > 0
            # ...and `Sigma_aw` is the tenths-of-m/s the plan's explicit-advection
            # argument rests on: at `ts = 0.5 s` over a 77 m layer that is a plume Courant
            # number of ~1e-3, which is why the Fortran's implicit environment half and
            # its `khdz` diagonal-dominance floor are dropped here.
            aw = m1.mynn.s_aw
            @test 0.0 < maximum(aw) < 1.0
            @test maximum(aw) * 0.5 / minimum(m1.mynn.dz) < 0.01
            # ...and they moved the state.
            @test E0 != E1

            # THE ACCEPTANCE: the column energy identity of D3, with plumes.
            wq = gauss_weights(kDim, nk, gp0.kMax - gp0.kMin)
            re_i = mo1.grid_params.vars["rho_e"]
            for c in (2, 3, 4)
                rng = ((c - 1) * kDim + 1):(c * kDim)
                # `w == 0` and an x-uniform state make the rho_e transport identically
                # zero, so `expdot[rho_e]` IS the closure's own increment (the premise is
                # measured in test_mynn_bl.jl and re-measured here).
                div_col = m1.tile.physical[rng, mo1.grid_params.vars["u"], 2]
                @test maximum(abs.(div_col)) < 1.0e-10
                dEt = m1.expdot_n[rng, 6]
                de = m1.expdot_n[rng, re_i]
                # The non-boundary-layer part of dE_t is common to the plumes-on and
                # plumes-off tiles, so it cancels in the DIFFERENCE of the two identities
                # and what is left on each side is the closure's own.
                dEt0 = m0.expdot_n[rng, 6]
                de0 = m0.expdot_n[rng, re_i]
                lhs = sum(wq .* ((dEt .- dEt0) .+ (de .- de0)))
                rhs = m1.mynn.bdry_E[c] - m0.mynn.bdry_E[c]
                scale = sum(wq .* abs.(dEt .- dEt0)) + sum(wq .* abs.(de .- de0))
                @test isfinite(lhs - rhs)
                @test abs(lhs - rhs) <= 1.0e-12 * scale
                if !(abs(lhs - rhs) <= 1.0e-12 * scale)
                    @info "EDMF budget residual" c lhs rhs scale rel=abs(lhs-rhs)/scale
                end
            end

            # THE SIGN: an upward plume flux cools and dries what it leaves and warms
            # and moistens what it reaches. `dE_t` is the WHOLE boundary-layer energy
            # difference, not the heat leg alone -- but on this fixture every plume leg
            # points the same way (`s_t` and `q_t` both decrease upward, and the vapour
            # reaches `E_t` through `c_w + c_v = (C_pv - R_v)T + ke + gz > 0`), so its
            # sign IS the sign of the transport. The water slot below is the same
            # statement on a channel with no thermodynamic factor at all.
            c = 3
            rng = ((c - 1) * kDim + 1):(c * kDim)
            dQ = m1.expdot_n[rng, 6] .- m0.expdot_n[rng, 6]
            ztop = maximum(m1.mynn.plume_ztop)
            zc2 = m1.mynn.z_lay
            below = findall(k -> zc2[k] < 0.4 * ztop && aw[(c-1)*kDim + k] > 0.0, 1:kDim)
            above = findall(k -> zc2[k] > 0.7 * ztop && zc2[k] < 1.4 * ztop, 1:kDim)
            @test !isempty(below) && !isempty(above)
            @info "EDMF plume transport: below vs above the plume layer" ztop dE_below=sum(dQ[below]) dE_above=sum(dQ[above])
            @test sum(dQ[below]) < 0.0        # the plume takes energy out down here
            @test sum(dQ[above]) > 0.0        # ...and puts it in up there
            # ...and the water goes the same way, which is the same statement about the
            # sign of `M_w` on a column whose moisture decreases upward.
            dW = m1.expdot_n[rng, 3] .- m0.expdot_n[rng, 3]
            @test sum(dW[below]) < 0.0
            @test sum(dW[above]) > 0.0
            # The plume lifts LOW-momentum surface air, so `u_up - bar u < 0` aloft: the
            # flux is downward and the divergence accelerates the bottom and brakes the
            # top -- the opposite sign to heat and water, and the check that the momentum
            # leg is not just a copy of them.
            dU = m1.expdot_n[rng, 4] .- m0.expdot_n[rng, 4]
            @test sum(dU[below]) > 0.0
            @test sum(dU[above]) < 0.0

            # ...and `:mynn_edmf_mom = false` leaves the momentum slot alone while the
            # heat and water legs still move.
            m2, p2, mo2, gp2, z2, col2 = make_edmf_tile(tmp; edmf = 1, edmf_mom = false,
                num_cells_k = nk, SST = SST, profile = :convective,
                Ck = CONV_CK, U_min = CONV_U)
            set_state!(p2, mo2, gp2.kDim, z2, col2; V0 = 6.0, e0 = 0.4,
                       qpert = CONV_QPERT, qshape = :decreasing)
            step_all!(m2, p2, gp2.kDim)
            @test maximum(m2.mynn.plume_ktop) > 0
            # The plumes still fire and still move the heat, but the momentum signature
            # they stamped on `u` is gone -- strictly weaker, not merely different.
            dU2 = m2.expdot_n[rng, 4] .- m0.expdot_n[rng, 4]
            @test abs(sum(dU2[below])) < abs(sum(dU[below]))
            @test abs(sum(dU2[above])) < abs(sum(dU[above]))
            dQ2 = m2.expdot_n[rng, 6] .- m0.expdot_n[rng, 6]
            @test sum(dQ2[below]) < 0.0
            @test sum(dQ2[above]) > 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 3. The stall is a property of the FIRST LAYER THICKNESS (informational)
    # ──────────────────────────────────────────────
    @testset "vertical resolution study: when do the plumes leave the ground?" begin
        mktempdir() do tmp
            println("\nEDMF resolution study (25 km lid, convective column, " *
                    "SST = T_1 + 15 K):")
            println("  ", rpad("cells", 8), rpad("kDim", 7), rpad("dz1 [m]", 10),
                    rpad("z_1 [m]", 10), rpad("gate", 7), rpad("stall", 7),
                    rpad("plume", 7), rpad("ktop", 6), rpad("ztop [m]", 10),
                    "max Sigma_aw [m/s]")
            fired = 0
            for nk in (12, 25, 50, 100)
                _, _, _, _, zc, colc = make_edmf_tile(tmp; edmf = 0, num_cells_k = nk,
                                                      profile = :convective)
                SST = colc.Tk[1] + 15.0
                m, p, mo, gp, z, col = make_edmf_tile(tmp; edmf = 1, num_cells_k = nk,
                                                      SST = SST, profile = :convective,
                                                      Ck = CONV_CK, U_min = CONV_U)
                set_state!(p, mo, gp.kDim, z, col; V0 = 6.0, e0 = 0.4,
                           qpert = CONV_QPERT, qshape = :decreasing)
                step_all!(m, p, gp.kDim)
                MY = m.mynn
                println("  ", rpad(nk, 8), rpad(gp.kDim, 7),
                        rpad(round(MY.dz[1]; digits = 1), 10),
                        rpad(round(z[1]; digits = 1), 10),
                        rpad(sum(MY.n_gate_col), 7), rpad(sum(MY.n_stall_col), 7),
                        rpad(sum(MY.n_plume_col), 7),
                        rpad(maximum(MY.plume_ktop), 6),
                        rpad(round(maximum(MY.plume_ztop); digits = 1), 10),
                        round(maximum(MY.aw_max); sigdigits = 4))
                @test sum(MY.n_gate_col) > 0          # the gate passes at every spacing
                maximum(MY.plume_ktop) > 0 && (fired += 1)
            end
            # The point of the table: the gate is not the story. It passes at EVERY
            # spacing, including the one where nothing comes of it -- what decides is
            # whether the weakest of the eight plumes can accelerate out of the first
            # model layer, and at 638 m it cannot.
            @test fired >= 2
        end
    end

    # ──────────────────────────────────────────────
    # 4. Census / options
    # ──────────────────────────────────────────────
    @testset "options and census" begin
        mktempdir() do tmp
            m, p, mo, gp, z, col = make_edmf_tile(tmp; edmf = 1, num_cells_k = 25)
            @test m.mynn.edmf == 1
            @test m.mynn.edmf_mom
            @test length(m.mynn.ework) == Threads.maxthreadid()
            step_all!(m, p, gp.kDim)
            Scythe.mynn_write_final!(m)
            line = Scythe.mynn_census_line(m.mynn)
            @test occursin("mynn_plume_active_frac=", line)
            @test occursin("mynn_max_mass_flux=", line)
            @test occursin("mynn_max_ztop_m=", line)
            @test occursin("mynn_n_stall=", line)
            # ...and the off arm allocates no EDMF scratch and prints no plume fields.
            m0, _, _, _, _, _ = make_edmf_tile(tmp; edmf = 0, num_cells_k = 25)
            @test isempty(m0.mynn.ework)
            @test !occursin("mynn_plume", Scythe.mynn_census_line(m0.mynn))
            # Bad values are refused loudly.
            @test_throws ErrorException Scythe.validate_mynn_options(
                Dict{Symbol,Any}(:mynn => true, :mynn_edmf => 2),
                Dict{Symbol,Any}(), "moist_compressible_XZ", 0.5, 75)
            @test_throws ErrorException Scythe.validate_mynn_options(
                Dict{Symbol,Any}(:mynn => true, :mynn_edmf => 1),
                Dict{Symbol,Any}(), "moist_compressible_XZ", 0.5, 3)
            cfg = Scythe.validate_mynn_options(
                Dict{Symbol,Any}(:mynn => true, :mynn_edmf => 1,
                                 :mynn_edmf_mom => false),
                Dict{Symbol,Any}(), "moist_compressible_XZ", 0.5, 75)
            @test cfg.edmf == 1 && cfg.edmf_mom == false
        end
    end
end
