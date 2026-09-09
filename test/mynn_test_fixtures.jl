# Shared fixtures for the wired MYNN-EDMF tests (test_mynn_bl.jl, test_mynn_fidelity.jl).
#
# These moved OUT of test_mynn_bl.jl so the fidelity stage can build the same tile, the
# same state and the same D3 column-budget residual without either file copying the
# other. The D3 rule they still obey is the one that matters: a fixture must not share
# code with the MODEL -- `gauss_weights` is deliberately a reimplementation of the mish
# quadrature and not an import of `benchmarks/common` or of anything in `src/`. Sharing
# between TESTS is the opposite of that risk: two copies of `make_mynn_tile` drifting
# apart is how a "bitwise identical" claim quietly stops being about the same tile.
#
# A module, not a bag of globals: `set_state!`/`step_all!`/`stable_column` are names other
# test files already use for their own LOCAL fixtures (test_mynn_ice.jl,
# test_mynn_edmf_live.jl, test_netcdf_output.jl), and those must go on shadowing whatever
# is in Main.
module MYNNTestFixtures

using Test
using Scythe
using Springsteel
using SparseArrays

import Springsteel.Thermodynamics: Rd, gravity

export stable_column, bl_jet, make_mynn_tile, gauss_weights, set_state!, step_all!,
       budget_residual

"""Stably stratified dry column with an exact hydrostatic pressure for a linear
temperature profile: `p = p0 (T/T0)^{g/(R_d Gamma)}`. Dry (rho_v = 0) so the
diagnostic vapour is pinned at zero and retrieval-level wiggles cannot leak into the
water channels, whose `(L_v - R_v T)` energy factor amplifies them (the note in
test_louis_bl.jl)."""
function stable_column(z; T0 = 300.0, lapse = 0.004, p0 = 100000.0)
    Tk = @. T0 - lapse * z
    p_Pa = @. p0 * (Tk / T0)^(gravity / (Rd * lapse))
    rho_d = p_Pa ./ (Rd .* Tk)
    n = length(z)
    return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
end

"""Boundary-layer jet `u = V0 (z/zs)^2 exp(-z/zs)`: zero VALUE and zero SLOPE at the
ground and (to round-off) at the lid, so it is representable on the Neumann spline
AND both momentum boundary terms of the D3 identity vanish, while the shear between
0 and ~2 zs is a real boundary-layer shear rather than a domain-deep ramp."""
bl_jet(z, V0, zs) = @. V0 * (z / zs)^2 * exp(-z / zs)

"""Tile for the MYNN arm (or the matching closure-off control).

`mynn = false` builds the SAME grid and reference with `options[:mynn]` absent, which
is the control the tendency difference is taken against: the `rho_e` slot is appended
after every fixed slot and after the vapour, so slots 1-9 and `rho_v` keep the same
indices on both tiles.

`fidelity` (F1) is `nothing` for "no `:mynn_fidelity` key at all", `:fortran` for the
key set to its default, or a `Vector{Symbol}` of `Scythe.MYNN_DEVIATIONS` names.
`sfc_stability` turns the Monin-Obukhov surface layer on -- which `:rmol_sfc` requires,
and which is set OUTSIDE the `mynn` branch because the surface layer is shared. `Ck` is
the enthalpy exchange coefficient, exposed so a test can drive the surface heat flux past
the Fortran wrapper's clip."""
function make_mynn_tile(tmpdir; mynn = true, num_cells_k = 50, kMax = 25.0e3,
                        ts = 0.5, Cd = -1.0, fluxes = true, SST = 302.65,
                        init = :zero, interval = 20.0, K_max = Inf,
                        water_carry = :flux, ctrans = :none, rtrans = :none,
                        l_inf = 80.0, lapse = 0.004, Ck = 1.0e-3,
                        fidelity = nothing, sfc_stability = false, trace = false)
    opts_names = Dict{Symbol,Any}(:condensate_transform => ctrans,
                                  :rain_transform => rtrans)
    mynn && (opts_names[:mynn] = true)
    varlist = Scythe.mc_var_names(opts_names; cyl = false)
    rain_name = Scythe.rain_var_name(opts_names)
    vars = Dict(v => i for (i, v) in enumerate(varlist))
    scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
    # NEUMANN on the side walls for EVERY variable, `u` included. That is not the
    # production configuration; it is what makes this test's premise true. The
    # `rho_e` slot exists only on the closure-on tile, so its tendency has no partner
    # to difference against and the test relies on its TRANSPORT being identically
    # zero: `u = u(z)` uniform in x, `w = 0`, `rho_e = rho_e(z)`. With a Dirichlet `u`
    # the spline is forced to zero at the walls, an x-uniform wind is NOT
    # representable, and the resulting `du/dx` feeds `-rho_e div(u)` a term worth ~10 %
    # of the budget -- which is exactly what this fixture measured before the BC was
    # changed. A constant IS exactly representable on the Neumann basis (the filter
    # penalises the third derivative, which a constant does not have), so the
    # transport is zero to round-off and what is left in the slot is the closure's.
    side_bc = scalar_bc
    bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
    top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
    gp = GridParameters(geometry = "RiRk",
        iMin = 0.0, iMax = 24.0e3, num_cells_i = 4,
        kMin = 0.0, kMax = kMax, num_cells_k = num_cells_k,
        BCL = side_bc, BCR = side_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
    # The reference state is a function of the SOUNDING and the grid only -- not of the
    # closure's options -- so the fidelity and the surface-layer switch stay out of the
    # cache key on purpose: two arms that differ only in a deviation must start from the
    # same reference file, bit for bit.
    ref_file = joinpath(tmpdir,
        "mynn_bl_$(num_cells_k)_$(kMax)_$(lapse)_$(mynn).ref")
    options = Dict{Symbol,Any}(:semiimplicit => true,
                               :exact_reference_state => true,
                               :precipitation => false,
                               # `options[:surface_fluxes]` REQUIRES a boundary-layer
                               # closure (the driver refuses it otherwise), so the
                               # control tile never carries them -- which is right:
                               # the surface fluxes ARE part of what the closure
                               # contributes, and the difference must contain them.
                               :surface_fluxes => (mynn && fluxes),
                               :condensate_transform => ctrans,
                               :rain_transform => rtrans)
    # `options[:sfc_stability]` is the SURFACE LAYER's, not the closure's, so it is set
    # outside the `mynn` branch -- and only when asked for, so the default tile's options
    # dict is byte-identical to a pre-F1 one. The `:rmol_sfc` deviation needs it.
    sfc_stability && (options[:sfc_stability] = true)
    if mynn
        options[:mynn] = true
        options[:mynn_init] = init
        # `nothing` (the default) adds NO key at all, and `:fortran` adds the key with
        # the default value: the bitwise arm has to be able to show that those two are
        # the same run, so they cannot be spelled the same way here.
        fidelity === nothing || (options[:mynn_fidelity] = fidelity)
        options[:mynn_interval] = interval
        options[:mynn_water_carry] = water_carry
        options[:mynn_trace] = trace
        # `:mynn_output` now defaults to true (S9); this fixture has no `output_dir`
        # of its own (it falls back to `ModelParameters`'s "./output/" default), so
        # every caller that reaches `mynn_write_final!` would otherwise write a
        # sidecar into the repo's working directory. Off here; test_mynn_io.jl is
        # where the sidecar itself is tested, with its own `mktempdir` output_dir.
        options[:mynn_output] = false
    end
    model = ModelParameters(
        ts = ts, integration_time = 10.0 * ts, output_interval = 10.0 * ts,
        equation_set = "moist_compressible_XZ",
        ref_state_file = ref_file, grid_params = gp,
        physical_params = Dict{Symbol,Any}(
            :Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
            :Kvdiff_water => 0.0, :tau_qss => 10.0, :alpha => 0.0,
            :z_damp => 20.0e3, :f => 0.0, :Cd => Cd, :Ls => 0.0,
            :Ck => Ck, :U_min => 1.0, :l_inf => l_inf, :SST => SST,
            :mynn_K_max => K_max),
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

"""3-point Gauss-Legendre cell weights on the RiRk mish (mubar = 3). Deliberately
reimplemented here rather than imported from benchmarks/common: the budget assertion
must not share code with anything the model uses."""
function gauss_weights(kDim, num_cells_k, L)
    @assert kDim == 3 * num_cells_k
    dz = L / num_cells_k
    w = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0) .* dz
    return repeat(collect(w), num_cells_k)
end

"""Impose u(z), rho_e(z) (both uniform in x) and the E_t' = rho_t ke compensation, so
the imposed wind does not double as a thermal perturbation (the retrieval reads E_t
minus the kinetic energy)."""
function set_state!(patch, model, kDim, z, col; V0 = 12.0, zs = 500.0, e0 = 0.4,
                    ze = 1500.0, W0 = 0.0, qpert = 2.0e-3)
    vars = model.grid_params.vars
    ui = vars["u"]; wi = vars["w"]; ei = vars["E_t"]
    rei = get(vars, "rho_e", 0)
    u = bl_jet(z, V0, zs)
    rho_t = col.rho_d .+ col.rho_v .+ col.rho_c
    ztop = model.grid_params.kMax
    npts = size(patch.physical, 1)
    for j in 1:npts
        k = mod1(j, kDim)
        patch.physical[j, ui, 1] = u[k]
        # `w` is OPTIONAL and defaults to zero. The budget tests need it zero: with a
        # nonzero `w` the divergence is nonzero and the TKE slot's own transport stops
        # being zero, which is the premise those tests rest on (see the fixture's BC
        # note). The fold test turns it on because slot 5 has to be shown to move.
        patch.physical[j, wi, 1] = W0 * sin(pi * z[k] / ztop)
        # Water PERTURBATIONS. The mixing acts on perturbations from the reference, so
        # a dry reference with zero water slots leaves every D7 leg identically zero
        # and untested. These make the vapour, total-water, cloud and rain fluxes -- and
        # with them the flux-form energy carry `S_Ew` -- live.
        bump = qpert * exp(-((z[k] - 800.0)/600.0)^2)
        patch.physical[j, vars["rho_v"], 1] = rho_t[k] * bump
        patch.physical[j, vars["rho_t"], 1] = rho_t[k] * bump
        patch.physical[j, vars["rho_c"], 1] = rho_t[k] * 0.2 * bump
        patch.physical[j, vars["rho_r"], 1] = rho_t[k] * 0.1 * bump
        rei > 0 && (patch.physical[j, rei, 1] = rho_t[k] * e0 * exp(-z[k] / ze))
    end
    # Fit first, then compensate with the FITTED kinetic energy (a raw-value
    # compensation leaves a ~1 % thermal residual from the fit of the wind itself).
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

"Advance every column once on a tile and return it."
function step_all!(mtile, patch, kDim, t = 1)
    ncols = div(size(patch.physical, 1), kDim)
    for c in 1:ncols
        Scythe.advance_column(mtile, c, t)
    end
    return ncols
end

"""Column budget residual on one grid/timestep/surface configuration.

Returns `(residual, scale, ratio)` where `residual` is
`sum_k w_k (dE_t + drho_e) - bdry_E` and `scale` is `sum_k w_k |dE_t|`."""
function budget_residual(tmp; num_cells_k, ts, Cd, fluxes, water_carry = :flux,
                         V0 = 12.0, fidelity = :fortran, sfc_stability = false)
    m_on, p_on, mo, gpo, z, col = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                 num_cells_k = num_cells_k, ts = ts,
                                                 Cd = Cd, fluxes = fluxes,
                                                 water_carry = water_carry,
                                                 fidelity = fidelity,
                                                 sfc_stability = sfc_stability)
    # The CONTROL never carries the closure, so it never carries a fidelity either; the
    # surface layer is SHARED, so it does carry that.
    m_off, p_off, mf, _, _, _ = make_mynn_tile(tmp; mynn = false,
                                               num_cells_k = num_cells_k, ts = ts,
                                               Cd = Cd, fluxes = fluxes,
                                               sfc_stability = sfc_stability)
    kDim = gpo.kDim
    set_state!(p_on, mo, kDim, z, col; V0 = V0)
    set_state!(p_off, mf, kDim, z, col; V0 = V0)
    ncols = step_all!(m_on, p_on, kDim)
    step_all!(m_off, p_off, kDim)
    re_i = mo.grid_params.vars["rho_e"]
    wq = gauss_weights(kDim, num_cells_k, gpo.kMax - gpo.kMin)
    c = 6                                      # an interior column, away from both walls
    rng = ((c - 1) * kDim + 1):(c * kDim)
    dE = m_on.expdot_n[rng, 6] .- m_off.expdot_n[rng, 6]
    de = m_on.expdot_n[rng, re_i]
    lhs = sum(wq .* (dE .+ de))
    rhs = m_on.mynn.bdry_E[c]
    # The premise: with an x-uniform wind on a Neumann horizontal basis the TKE
    # slot's own transport is identically zero, so `de` IS the closure's increment.
    # Measured, not assumed -- it is the term that hid an 11 % budget residual until
    # the side-wall BC was fixed.
    div_col = m_on.tile.physical[rng, mo.grid_params.vars["u"], 2]
    @test maximum(abs.(div_col)) < 1.0e-10
    scale = sum(wq .* abs.(dE)) + sum(wq .* abs.(de))
    return (lhs - rhs, scale, m_on, mo, gpo, rng, wq, dE, de)
end

end # module
