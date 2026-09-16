using Test
using Scythe
using Springsteel
using SparseArrays
using SharedArrays
using LinearAlgebra

include("distributed_test_helpers.jl")

# PERIODIC lateral boundary conditions in x, on the RiRk geometry (cubic B-spline in
# BOTH i and k), driven through a real Scythe time integration — single-process and
# with several worker tiles.
#
# Why this file exists: `PeriodicBC()` maps to `CubicBSpline.PERIODIC`, whose Γ ladder
# wraps the first/last B-coefficients into each other and forces a DENSE Cholesky of
# (P+Q) (the wraparound breaks the half-bandwidth-3 structure every other BC keeps).
# Nothing in the suite exercised that path on RiRk, let alone through the tiled loop,
# and a periodic RCE driver is about to depend on it.
#
# What the tiled path actually does, and why periodicity is expected to survive it:
#   A → physical   `tileTransform!(::RiRk_Grid)` — pure basis evaluation of the tile's
#                  window of the PATCH A-coefficients; no BC enters.
#   physical → B   `SBtransform!` — Gauss quadrature against the basis; BC-free.
#   B assembly     each tile writes its inner B rows and hands its 3-row right halo to
#                  its neighbour (the last tile's goes to the master), so the patch
#                  B-vector is the same partition-of-quadrature sum the single-process
#                  run computes in one go. No first↔last wrap is needed here: the wrap
#                  lives in the Γ operator, not in the quadrature.
#   B → A          the 3-arg `splineTransform!(shared, patch, tile)` solves on the
#                  PATCH splines — i.e. the periodic solve is always global, never
#                  per-tile — and the tile then copies its window out.
# Testsets 1 and 2 verify that reasoning end to end rather than asserting it.

@testset "Periodic RiRk (distributed)" begin

# ════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════

"""Exact 2-D Gauss-quadrature weights on the RiRk mish (i outer, k inner).

The dot product with a mish-sampled field is the exact integral of the field's cubic
spline representation — the same quadrature the Galerkin solver uses. Mirrors
`gauss_cell_weights` in benchmarks/common/diagnostics.jl, tensored in k.
"""
function rirk_mish_weights(params)
    _, qw_i = Springsteel.CubicBSpline._quadrature_rule(params.mubar, params.quadrature)
    _, qw_k = Springsteel.CubicBSpline._quadrature_rule(params.mubar, params.quadrature)
    Wi = repeat(qw_i .* ((params.iMax - params.iMin) / params.num_cells),
                outer = params.num_cells)
    Wk = repeat(qw_k .* ((params.kMax - params.kMin) / params.num_cells_k),
                outer = params.num_cells_k)
    W = zeros(Float64, params.iDim * params.kDim)
    g = 1
    for r in 1:params.iDim, z in 1:params.kDim
        W[g] = Wi[r] * Wk[z]
        g += 1
    end
    return W
end

"""Field value and ∂x at the two domain ENDPOINTS, x = iMin and x = iMax.

The Gauss mish never lands on an endpoint (`calcMishPoints` uses interior quadrature
nodes), so "the first and last x column" are two DIFFERENT physical locations and
comparing them says nothing about periodicity. `x = iMin` and `x = iMax` ARE the same
physical location under periodicity, so the spline is evaluated there directly.
Returns `(value_L, value_R, dx_L, dx_R)`, each a column over `k_pts`.
"""
function endpoint_pair(patch, v::Int, k_pts::Vector{Float64})
    rg = regularGridTransform(patch, [patch.params.iMin, patch.params.iMax], k_pts)
    nk = length(k_pts)
    return (rg[1:nk, v, 1], rg[nk+1:2nk, v, 1], rg[1:nk, v, 2], rg[nk+1:2nk, v, 2])
end

# ════════════════════════════════════════════════════════════════════════════
# 1. Linear advection straight across the periodic boundary
# ════════════════════════════════════════════════════════════════════════════
@testset "Linear advection across the periodic seam" begin

    LX = 100.0            # domain length [m]
    X0 = 70.0             # initial blob centre
    SIG = 5.0             # blob width
    U0 = 1.0              # advecting speed [m/s]

    """LinearAdvectionRZ (src/testModels.jl) on a periodic-in-x RiRk grid.

    `K = 0` kills the diffusion term, which is what carries the cylindrical `hr/r`
    factor; `iMin = 0` is safe with it because the Gauss mish is strictly interior
    (`calcMishPoints`: x = xmin + mc·DX + qpts[mu]·DX with qpts ∈ (0,1)), so r > 0 at
    every point the equation set ever touches — verified below.
    """
    function make_model(; num_cells = 50, num_cells_k = 4, ts = 0.1, num_ts = 600)
        vars = Dict("h" => 1, "u" => 2, "v" => 3, "w" => 4)
        # PERIODIC must be declared on BOTH sides — the Γ ladder builds the left
        # (rank 1) and right (rank 2) wraparound rows independently.
        periodic = Dict(v => PeriodicBC() for v in keys(vars))
        natural  = Dict(v => NaturalBC()  for v in keys(vars))
        gp = GridParameters(
            geometry = "RiRk",
            iMin = 0.0, iMax = LX, num_cells = num_cells,
            kMin = 0.0, kMax = 10.0, num_cells_k = num_cells_k,
            BCL = periodic, BCR = periodic, BCB = natural, BCT = natural,
            vars = vars,
        )
        model = ModelParameters(
            ts = ts,
            integration_time = num_ts * ts,
            output_interval = num_ts * ts,
            equation_set = "LinearAdvectionRZ",
            initial_conditions = "",
            grid_params = gp,
            physical_params = Dict(:K => 0.0),   # no diffusion: no hr/r term
        )
        return model, num_ts
    end

    """The analytic solution: a Gaussian on the CIRCLE of circumference LX."""
    function wrapped_gaussian(x; x0 = X0, sigma = SIG)
        d = mod(x - x0 + LX / 2, LX) - LX / 2     # signed periodic distance
        return exp(-d^2 / (2 * sigma^2))
    end

    function blob_ic(model)
        patch = createGrid(model.grid_params)
        gridpoints = getGridpoints(patch)
        for i in 1:size(patch.physical, 1)
            patch.physical[i, 1, 1] = wrapped_gaussian(gridpoints[i, 1])
            patch.physical[i, 2, 1] = U0     # u
            patch.physical[i, 3, 1] = 0.0    # v
            patch.physical[i, 4, 1] = 0.0    # w
        end
        spectralTransform!(patch)
        return copy(patch.spectral), gridpoints, patch
    end

    model, num_ts = make_model()
    ic_spectral, gridpoints, ic_patch = blob_ic(model)
    gridTransform!(ic_patch)

    # The r = 0 hazard the equation set would have: never realised on the mish.
    @test minimum(gridpoints[:, 1]) > 0.0

    W = rirk_mish_weights(ic_patch.params)
    integral_0 = dot(W, ic_patch.physical[:, 1, 1])

    single, _ = run_single_process_simulation(model, ic_spectral, num_ts)

    # ── (i) no reflection, no wall: the blob is where the WRAPPED analytic
    #        solution says it is. A reflecting or rigid side would leave a bump
    #        piled near x = 100 and nothing at all at x = 30.
    shift = U0 * num_ts * model.ts                       # 60 m; 70 + 60 = 130 ≡ 30
    analytic = [wrapped_gaussian(gridpoints[i, 1]; x0 = X0 + shift)
                for i in 1:size(gridpoints, 1)]
    l2_error = norm(single[:, 1] .- analytic) / norm(analytic)
    @test l2_error < 0.05

    peak = maximum(single[:, 1])
    # The blob really did land at x ≈ 30, not stall at the wall.
    peak_x = gridpoints[argmax(single[:, 1]), 1]
    @test abs(peak_x - 30.0) < 2.0

    # ── (ii) nothing left behind at the outflow edge.
    right_edge = findall(x -> x >= 85.0, view(gridpoints, :, 1))
    edge_max = maximum(abs.(single[right_edge, 1]))
    @test edge_max < 1.0e-3 * peak

    # ── (iii) mass. d/dt ∫h = -u[h]₀^L = 0 under periodicity, exactly.
    integral_1 = dot(W, single[:, 1])
    mass_drift = abs(integral_1 - integral_0) / abs(integral_0)
    @test mass_drift < 1.0e-10

    # u is untouched by LinearAdvectionRZ; a periodic spline must carry the
    # constant exactly, or the "advecting speed" would drift under its own fit.
    @test maximum(abs.(single[:, 2] .- U0)) < 1.0e-10

    println("  periodic RiRk advection: L2 err = $(round(l2_error, digits=6)), " *
            "peak at x = $(round(peak_x, digits=2)), " *
            "max|h| on [85,100] = $(round(edge_max, sigdigits=3)) " *
            "($(round(edge_max/peak, sigdigits=3)) of peak), " *
            "mass drift = $(round(mass_drift, sigdigits=3))")

    # ── (iv) the tiled path. 1 tile must reproduce single-process to round-off;
    #        2 and 3 tiles differ only through the order in which the 3-row halo
    #        overlaps are summed into the patch B-vector.
    tile_diffs = Float64[]
    for nw in (1, 2, 3)
        dist, _ = run_distributed_simulation(model, ic_spectral, nw, num_ts)
        @test all(isfinite, dist)
        d = maximum(abs.(dist[:, 1] .- single[:, 1]))
        push!(tile_diffs, d)
        # 1e-10 on an O(1) field after 600 steps. The B-assembly summation-order
        # argument predicts ~1e-8 at worst; the measurement is ~3e-14, so this
        # bound has four orders of headroom and would still catch a real
        # decomposition error (a mis-wrapped halo shows up at O(1), not O(1e-10)).
        @test d < 1.0e-10
    end
    println("  tile-vs-single max|Δh|: 1 tile $(round(tile_diffs[1], sigdigits=3)), " *
            "2 tiles $(round(tile_diffs[2], sigdigits=3)), " *
            "3 tiles $(round(tile_diffs[3], sigdigits=3))")

    # ── The seam itself, with the blob sitting ON it. At 300 steps the centre has
    #    moved 30 m and is exactly at x = 100 ≡ 0, so both endpoints carry the PEAK
    #    rather than a tail: the strongest available statement that the field wrapped
    #    rather than ran into something.
    half_ts = 300
    straddle, _ = run_single_process_simulation(model, ic_spectral, half_ts)
    sp = createGrid(model.grid_params)
    sp.physical[:, :, 1] .= straddle
    spectralTransform!(sp)
    k_pts = collect(LinRange(0.0, 10.0, 5))
    hL, hR, dhL, dhR = endpoint_pair(sp, 1, k_pts)
    @test maximum(hL) > 0.9                       # the peak is AT the seam
    @test maximum(abs.(hL .- hR)) < 1.0e-8 * maximum(abs.(hL))
    @test maximum(abs.(dhL .- dhR)) < 1.0e-8 * max(maximum(abs.(dhL)), 1.0)
    println("  seam at t = 300 steps: h(0) peak = $(round(maximum(hL), digits=4)), " *
            "|h(0) − h(L)| = $(maximum(abs.(hL .- hR))), " *
            "|∂ₓh(0) − ∂ₓh(L)| = $(maximum(abs.(dhL .- dhR)))")
end

# ════════════════════════════════════════════════════════════════════════════
# 2. The moist compressible model with periodic sides
# ════════════════════════════════════════════════════════════════════════════
@testset "moist_compressible_XZ with periodic sides" begin

    # Geometry copied from the "SI horizontal-acoustic ceiling removed (resting XZ,
    # RiRk)" configuration in test_moist_compressible.jl, with the side walls turned
    # periodic. ts is 0.25 s rather than that test's 1.0 s because that test runs with
    # options[:horizontal_semiimplicit] = true, which periodic sides are not allowed to
    # use (testset 3); at dx = 500 m a horizontal acoustic Courant of 340·1.0/500 = 0.68
    # sits on the EXPLICIT AB3 limit (~0.7) and this run blows up there. Measured: the
    # blow-up is identical with u = 0 and with walls instead of periodic sides, i.e. it
    # is the documented explicit horizontal-acoustic ceiling and has nothing to do with
    # the boundary condition. ts = 0.25 puts Co_h at 0.17.
    DX = 500.0
    NCX = 50
    LX = DX * NCX          # 25 km
    LZ = 10.0e3
    TS = 0.25
    NUM_TS = 200
    U0 = 10.0

    """The periodic moist config, its reference file written, plus a built ModelTile.

    The reference sounding is isothermal at 290 K with a CONSTANT vapour mixing ratio,
    so the moist gas constant is constant and `p = p₀ exp(-g z / (R_m T))` is exactly
    hydrostatic for `ρ_t = p/(R_m T)`, `ρ_v = q_v ρ_t`, `ρ_d = ρ_t − ρ_v`. RH works out
    near 67% at the surface, so nothing condenses and this stays a boundary-condition
    test rather than a microphysics one.

    `positivity` carries `:k` only: an `:i` bound is refused outright under periodic
    boundaries (testset 3), because the periodic Γ forms slaved coefficients from signed
    combinations of the free ones.
    """
    function build_mc(tmpdir; options = Dict{Symbol,Any}())
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
        periodic  = Dict(v => PeriodicBC() for v in keys(vars))
        scalar_bc = Dict(v => NeumannBC()  for v in keys(vars))
        wall_bc   = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(
            geometry = "RiRk",
            iMin = 0.0, iMax = LX, num_cells_i = NCX,
            kMin = 0.0, kMax = LZ, num_cells_k = 20,
            BCL = periodic, BCR = periodic, BCB = wall_bc, BCT = wall_bc,
            vars = vars,
            positivity = Dict("rho_c" => Dict(:k => 0.0),
                              "rho_r" => Dict(:k => 0.0)),
        )
        ref_file = joinpath(tmpdir, "periodic_rirk.ref")
        model = ModelParameters(
            ts = TS, integration_time = NUM_TS * TS, output_interval = NUM_TS * TS,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                   :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                   :tau_qss => 10.0, :alpha => 0.0,
                                   :z_damp => 30.0e3, :f => 0.0),
            options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                             :exact_reference_state => true,
                                             :precipitation => false), options))
        patch = createGrid(model.grid_params)
        kDim = patch.params.kDim
        z = getGridpoints(patch)[1:kDim, end]
        T0 = 290.0
        q_v = 0.008
        R_m = Scythe.Rd * (1.0 - q_v) + Scythe.Rv * q_v
        p_Pa  = @. 101325.0 * exp(-Scythe.gravity * z / (R_m * T0))
        rho_t = p_Pa ./ (R_m * T0)
        rho_v = q_v .* rho_t
        Scythe.write_exact_ref_mc(ref_file, z, p_Pa, rho_t .- rho_v, rho_v, zeros(kDim))
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return model, patch, mtile
    end

    """Seed `patch` and return its spectral IC."""
    function seed!(patch, model, mtile; bubble::Bool, u0::Float64)
        patch.physical .= 0.0
        if bubble
            Scythe.moist_temperature_bubble_mc!(patch, getGridpoints(patch),
                mtile.ref_state;
                xc = 0.75 * LX, xr = 4.0e3, zc = 2.0e3, zr = 2.0e3, dT_max = 2.0)
        end
        patch.physical[:, model.grid_params.vars["u"], 1] .= u0
        spectralTransform!(patch)
        gridTransform!(patch)
        return copy(patch.spectral)
    end

    mktempdir() do tmpdir
        model, patch, mtile = build_mc(tmpdir)
        vars = model.grid_params.vars
        W = rirk_mish_weights(patch.params)
        rho_t_i = vars["rho_t"]; p_i = vars["p"]; u_i = vars["u"]; w_i = vars["w"]

        # ── Control: the resting periodic base. If the periodic side walls injected
        #    anything at all — a seam mismatch, a Γ wraparound error — it would show
        #    up here first, because nothing else in this run is moving.
        rest_ic = seed!(patch, model, mtile; bubble = false, u0 = 0.0)
        rest_M0 = dot(W, patch.physical[:, rho_t_i, 1])
        resting, _ = run_single_process_simulation(model, rest_ic, NUM_TS)
        rest_drift = abs(dot(W, resting[:, rho_t_i]) - rest_M0)
        @test all(isfinite, resting)
        @test rest_drift == 0.0
        # Every prognostic except Q_ss stays at EXACTLY zero for the whole run — the
        # periodic base is a discrete fixed point, bit for bit. Q_ss carries a
        # 3.3e-8 residual because this config does not set
        # options[:consistent_qss_reference]; that is the known reference-fit crumb
        # that option exists to remove (test_moist_compressible.jl, "consistent_qss_
        # reference: the resting base is an EXACT fixed point"), not a seam effect.
        for (name, v) in vars
            name == "Q_ss" && continue
            @test maximum(abs.(view(resting, :, v))) == 0.0
        end
        qss_resid = maximum(abs.(view(resting, :, vars["Q_ss"])))
        @test qss_resid < 1.0e-6
        println("  resting periodic base after $NUM_TS steps: every prognostic but " *
                "Q_ss exactly 0, |Q_ss′| = $(round(qss_resid, sigdigits=3)), " *
                "∫ρ_t′ drift = $rest_drift")

        # ── The live run: warm bubble at 0.75·L riding a uniform u = 10 m/s.
        ic_spectral = seed!(patch, model, mtile; bubble = true, u0 = U0)
        mass_0  = dot(W, patch.physical[:, rho_t_i, 1])
        # ∫|ρ_t′| is the physical scale the drift is measured against — ∫ρ_t′ itself
        # is a near-cancelling sum, so a ratio against it would flatter the result.
        mass_scale = dot(W, abs.(patch.physical[:, rho_t_i, 1]))

        single, _ = run_single_process_simulation(model, ic_spectral, NUM_TS)
        @test all(isfinite, single)
        @test maximum(abs.(single[:, p_i])) < 1.0e4      # bounded, not blowing up
        @test maximum(abs.(single[:, w_i])) > 1.0e-3     # the bubble did something

        mass_drift = abs(dot(W, single[:, rho_t_i]) - mass_0) / mass_scale
        # The resting control's own drift is exactly 0, so the only tolerance that
        # means anything here is round-off on the quadrature sum itself.
        @test mass_drift < 1.0e-12
        println("  moist run: max|p′| = $(round(maximum(abs.(single[:, p_i])), digits=3)) Pa, " *
                "max|w| = $(round(maximum(abs.(single[:, w_i])), digits=4)) m/s, " *
                "∫ρ_t′ drift = $(round(mass_drift, sigdigits=3)) of ∫|ρ_t′|")

        # ── The seam. x = iMin and x = iMax are the SAME physical location under
        #    periodicity; the mish never lands on either, so the spline is evaluated
        #    there (see `endpoint_pair`). Both the value and ∂ₓ must match.
        final = createGrid(model.grid_params)
        final.physical[:, :, 1] .= single
        spectralTransform!(final)
        k_pts = collect(LinRange(0.0, LZ, 21))
        seam_worst = 0.0
        for name in ("p", "u", "w", "rho_t", "E_t")
            v = vars[name]
            fL, fR, dL, dR = endpoint_pair(final, v, k_pts)
            scale = max(maximum(abs.(fL)), maximum(abs.(fR)))
            rel = scale > 0.0 ? maximum(abs.(fL .- fR)) / scale : 0.0
            seam_worst = max(seam_worst, rel)
            @test rel < 1.0e-8
            dscale = max(maximum(abs.(dL)), maximum(abs.(dR)))
            @test (dscale > 0.0 ? maximum(abs.(dL .- dR)) / dscale : 0.0) < 1.0e-8
        end
        # p′ at the seam is not a trivial zero — the bubble's acoustic/buoyancy
        # response has reached the boundary, so the match above has content.
        fL, _, _, _ = endpoint_pair(final, p_i, k_pts)
        @test maximum(abs.(fL)) > 0.1
        println("  seam agreement (worst of p,u,w,ρ_t,E_t, relative): $seam_worst; " *
                "|p′| at the seam = $(round(maximum(abs.(fL)), digits=3)) Pa")

        # ── Tiles.
        for nw in (1, 2)
            dist, _ = run_distributed_simulation(model, ic_spectral, nw, NUM_TS)
            @test all(isfinite, dist)
            worst = 0.0
            worst_var = ""
            for (name, v) in vars
                scale = maximum(abs.(view(single, :, v)))
                scale == 0.0 && continue
                rel = maximum(abs.(dist[:, v] .- single[:, v])) / scale
                if rel > worst
                    worst = rel
                    worst_var = name
                end
            end
            # Same argument as testset 1: the only legitimate difference is the
            # order of the halo summation into the patch B-vector. 1e-10 relative
            # over 200 moist steps; measured ~2e-13.
            @test worst < 1.0e-10
            println("  $nw-tile vs single-process: worst relative diff = " *
                    "$(round(worst, sigdigits=3)) (on $worst_var)")
        end
    end
end

# ════════════════════════════════════════════════════════════════════════════
# 3. The refusals that must keep firing under periodic sides
# ════════════════════════════════════════════════════════════════════════════
@testset "periodic configurations that must be refused" begin

    # ── (a) A positivity bound on the i leg. The periodic Γ builds the slaved
    #        coefficients from SIGNED combinations of the free ones, so a box
    #        constraint in free space does not imply the bound on the reconstruction.
    #        This has to fail at grid construction, not silently produce a wrong fit.
    let
        vars = Dict("h" => 1, "u" => 2, "v" => 3, "w" => 4)
        periodic = Dict(v => PeriodicBC() for v in keys(vars))
        natural  = Dict(v => NaturalBC()  for v in keys(vars))
        base = (geometry = "RiRk", iMin = 0.0, iMax = 100.0, num_cells = 50,
                kMin = 0.0, kMax = 10.0, num_cells_k = 4,
                BCL = periodic, BCR = periodic, BCB = natural, BCT = natural,
                vars = vars)
        gp_i = GridParameters(; base..., positivity = Dict("h" => Dict(:i => 0.0)))
        err = try
            createGrid(gp_i)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin("PERIODIC", msg)
        @test occursin("positivity bound is not supported", msg)

        # ...while the k leg, whose boundaries are not periodic, is accepted.
        gp_k = GridParameters(; base..., positivity = Dict("h" => Dict(:k => 0.0)))
        @test createGrid(gp_k) isa Springsteel.AbstractGrid
    end

    # ── (b) and (c): the two opt-in horizontal solvers. Both are off by default;
    #        neither supports a periodic u side wall, and both must say so.
    mktempdir() do tmpdir
        function refusal_setup(tmpdir, options)
            vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
            periodic  = Dict(v => PeriodicBC() for v in keys(vars))
            scalar_bc = Dict(v => NeumannBC()  for v in keys(vars))
            wall_bc   = merge(scalar_bc, Dict("w" => DirichletBC()))
            gp = GridParameters(geometry = "RiRk",
                iMin = 0.0, iMax = 25.0e3, num_cells_i = 50,
                kMin = 0.0, kMax = 10.0e3, num_cells_k = 20,
                BCL = periodic, BCR = periodic, BCB = wall_bc, BCT = wall_bc,
                vars = vars)
            ref_file = joinpath(tmpdir, "refuse_$(join(sort(string.(keys(options))))).ref")
            model = ModelParameters(
                ts = 0.25, integration_time = 1.0, output_interval = 1.0,
                equation_set = "moist_compressible_XZ",
                ref_state_file = ref_file, grid_params = gp,
                physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                       :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                       :tau_qss => 10.0, :alpha => 0.0,
                                       :z_damp => 30.0e3, :f => 0.0),
                options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                                 :exact_reference_state => true,
                                                 :precipitation => false), options))
            patch = createGrid(model.grid_params)
            kDim = patch.params.kDim
            z = getGridpoints(patch)[1:kDim, end]
            T0 = 290.0; q_v = 0.008
            R_m = Scythe.Rd * (1.0 - q_v) + Scythe.Rv * q_v
            p_Pa  = @. 101325.0 * exp(-Scythe.gravity * z / (R_m * T0))
            rho_t = p_Pa ./ (R_m * T0); rho_v = q_v .* rho_t
            Scythe.write_exact_ref_mc(ref_file, z, p_Pa, rho_t .- rho_v, rho_v,
                                      zeros(kDim))
            haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                    size(patch.spectral, 1), size(patch.spectral, 2))
            mtile = createModelTile(patch, patch, model, haloReceiveMap)
            return model, patch, mtile
        end

        ref2(mtile) = (collect(view(Springsteel.ref_rho_t(mtile.ref_state), :, 1:2)),
                       collect(view(Springsteel.ref_rho_d(mtile.ref_state), :, 1:2)),
                       collect(view(Springsteel.ref_total_energy(mtile.ref_state), :, 1:2) .+
                               view(Springsteel.ref_pressure(mtile.ref_state), :, 1:2)))
        ref1(mtile) = (collect(view(Springsteel.ref_rho_t(mtile.ref_state), :, 1)),
                       collect(view(Springsteel.ref_rho_d(mtile.ref_state), :, 1)),
                       collect(view(Springsteel.ref_total_energy(mtile.ref_state), :, 1) .+
                               view(Springsteel.ref_pressure(mtile.ref_state), :, 1)))

        # (b) options[:exact_si] — src/exact_si.jl
        model, patch, mtile = refusal_setup(tmpdir, Dict{Symbol,Any}(:exact_si => true))
        rt, rd, etp = ref2(mtile)
        err = try
            Scythe.create_exact_si_data(patch, model, mtile.mc_ref_diag.Pxi_prof,
                                        rt, rd, etp)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("periodic", sprint(showerror, err))

        # (c) options[:horizontal_semiimplicit] — src/horizontal_si.jl
        model, patch, mtile = refusal_setup(tmpdir,
            Dict{Symbol,Any}(:horizontal_semiimplicit => true))
        rt, rd, etp = ref1(mtile)
        err = try
            Scythe.create_horizontal_solve_data(patch, model, mtile.mc_ref_diag.Pxi_prof,
                                                rt, rd, etp)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("Dirichlet or Neumann u side walls only", sprint(showerror, err))
    end
end

end
