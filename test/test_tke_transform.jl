# test/test_tke_transform.jl -- the MYNN TKE density's diagnostics and its CONTROL VARIABLE
# (`physics.boundary_layer.tke_transform`, `options[:tke_transform]`, `physical_params[:tke_mu]`;
# Stream D of the 2026-10-27 close-out, reference/session_2026-10_scratch/closeout_2026-10-27/
# rhoe_diag/DECISION.md).
#
# The measured defect: the transported TKE density rho_e rings negative at 70-90 % of the points
# above the mixing top (min -1.5 J/m3 at 3-4.5 km, lobes as large as the signal), where the
# closure reads max(e, 0) (K_m = eps = 0) while the transport, the netCDF writer and the
# sidecar read the signed slot. Under `:bhyp` the slot carries Ooyama's control variable
# nu_e = bhyp(rho_e; mu_e) (rho_e_bar == 0, so no reference), every reader sees the recovered
# ahyp(nu_e) >= 0, and every density-space tendency reaches the slot through J_e = dnu/drho.
#
# E0 (this file's first testset) is the diagnostics, bitwise on the state: the TKE lobe census on
# the energy-census line, the split of the MYNN clamp counter, the rho_e line of the state-minima
# trace, and the netCDF fields rho_e_deficit, rho_w (signed) and lobe_pool (the partition gap's
# Lambda per point).

using Test
using Scythe
using Springsteel
using SparseArrays
using NCDatasets

isdefined(Main, :MYNNTestFixtures) || include("mynn_test_fixtures.jl")
isdefined(Main, :VaporTransformFixtures) || include("vapor_transform_fixtures.jl")

"Capture everything `f()` prints to stdout (Julia-level `println` only)."
function _tke_capture(f)
    path, io = mktemp()
    redirect_stdout(io) do
        f()
    end
    flush(io); close(io)
    return read(path, String)
end

"Seed a negative TKE lobe aloft (3-4 km, depth `amp` J/m^3) into the slot, refit."
function _tke_seed_lobe!(p, re, z, kDim; amp = 0.5)
    for j in axes(p.physical, 1)
        k = mod1(j, kDim)
        p.physical[j, re, 1] -= amp * exp(-((z[k] - 3500.0) / 300.0)^2)
    end
    spectralTransform!(p)
    gridTransform!(p)
    return nothing
end

@testset "MYNN TKE density: diagnostics and control variable" begin
    M = Main.MYNNTestFixtures

    # ══ E0. diagnostics ═══════════════════════════════════════════════════════════════
    @testset "E0: TKE lobe census, clamp split, minima trace (bitwise on the state)" begin
        mktempdir() do tmp
            m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                extra_options = Dict{Symbol,Any}(:state_minima_trace => 1))
            kDim = gp.kDim
            M.set_state!(p, mo, kDim, z, col; W0 = 0.3)
            re = mo.grid_params.vars["rho_e"]
            _tke_seed_lobe!(p, re, z, kDim)
            slot = copy(p.physical[:, re, 1])
            nneg = count(<(0.0), slot)
            @test nneg > 0
            @test any(>(0.0), slot)

            # The clamp split: one step of every column counts each negative slot point once.
            M.step_all!(m, p, kDim, 1)
            @test sum(m.mynn.n_eneg_col) == nneg
            @test sum(m.mynn.n_ehigh_col) == 0
            @test sum(m.mynn.n_clamp_col) >= nneg          # every negative point is clamped
            line = Scythe.mynn_census_line(m.mynn)
            @test occursin("neg=$(nneg) high=0 small=$(sum(m.mynn.n_clamp_col) - nneg)", line)
            d = Scythe.mynn_diagnostics(m)
            @test d.attrs["n_clamp_e_neg"] == Float64(nneg)
            @test d.attrs["n_clamp_e_high"] == 0.0
            # ... and the slot was not touched by counting it
            @test p.physical[:, re, 1] == slot

            # The TKE lobe census on the energy-census line.
            out = _tke_capture(() -> Scythe.energy_census_trace(m, 11))
            @test occursin("rho_e<0 points=$(nneg) ", out)
            eb = m.ebudget
            expect = 0.0
            for g in axes(p.physical, 1)
                k = mod1(g, kDim)
                x = p.physical[g, re, 1]
                x < 0.0 && (expect += eb.wxn[div(g - 1, kDim) + 1] * (eb.wz[k] * x))
            end
            @test m.mynn.n_eneg_pts == nneg
            @test m.mynn.eneg_mass ≈ expect rtol = 1.0e-12
            @test m.mynn.eneg_mass < 0.0

            # The minima trace's rho_e line.
            tr = _tke_capture(() -> Scythe.state_minima_trace(m, 1))
            @test occursin("rho_e trace", tr)
            @test occursin("rho_e<0 points=$(nneg)", tr)
            @test occursin("min rho_e=$(round(minimum(slot); sigdigits = 4)) J/m^3", tr)

            # The netCDF writer: rho_e_deficit = min(rho_e, 0), rho_w signed, the lobe pool.
            ctx = Scythe.netcdf_output_context(p, mo)
            path = joinpath(tmp, "e0.nc")
            Scythe.write_netcdf_comprehensive(path, p, mo, 1.0, ctx)
            NCDataset(path, "r") do ds
                for nm in ("rho_e", "e", "rho_e_deficit", "rho_w", "rho_w_deficit",
                           "lobe_pool", "column_lobe_pool")
                    @test haskey(ds, nm)
                end
                @test Array(ds["rho_e_deficit"]) == min.(Array(ds["rho_e"]), 0.0)
                @test any(<(0.0), Array(ds["rho_e_deficit"]))
                @test Array(ds["rho_w_deficit"]) == min.(Array(ds["rho_w"]), 0.0)
                # no transform declared: the pool is an exact zero field
                @test all(x -> x === 0.0, Array(ds["lobe_pool"]))
            end
        end
        # A tile without the closure prints neither clause.
        mktempdir() do tmp
            m, p, mo, gp, z, col = M.make_mynn_tile(tmp; mynn = false, num_cells_k = 25,
                kMax = 5.0e3, extra_options = Dict{Symbol,Any}(:state_minima_trace => 1))
            M.set_state!(p, mo, gp.kDim, z, col)
            M.step_all!(m, p, gp.kDim, 1)
            out = _tke_capture(() -> Scythe.energy_census_trace(m, 11))
            @test occursin("energy census", out)
            @test !occursin("rho_e<0", out)
            tr = _tke_capture(() -> Scythe.state_minima_trace(m, 1))
            @test occursin("rho_w trace", tr)
            @test !occursin("rho_e trace", tr)
        end
    end

    @testset "E0: the lobe pool and the signed rho_w in the derived fields" begin
        z = collect(0.0:250.0:1000.0)
        nk = length(z)
        zp() = zeros(Float64, nk)
        ref = Scythe.RegularReferenceProfiles(z, zp(), zp(), zp(), zp(), zp(), zp(), zp(), zp())
        cfg(; ctrans = :none, rtrans = :none) =
            Scythe.DerivedConfig(ctrans, 1.0e-7, rtrans, 1.0e-7, :none, 1.0, :none,
                                 (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16), 1, false, false,
                                 8.0e6, 100.0, "test", false, :eos)
        m(f) = [f(i, z[k]) for i in 1:3, k in 1:nk]
        slots = (; p = m((i, zz) -> 1.0e5 * exp(-zz / 8000.0)),
                   rho_d = m((i, zz) -> 1.15 * exp(-zz / 8500.0)),
                   E_t = m((i, zz) -> 1.15 * exp(-zz / 8500.0) * (718.0 * 290.0 + 9.81 * zz)),
                   Q_ss = zeros(3, nk), rho_v = fill(1.0e-3, 3, nk), u = zeros(3, nk),
                   w = zeros(3, nk), rain = zeros(3, nk), cloud = zeros(3, nk))
        cl = zeros(3, nk); cl[1, 2] = -2.0e-4; cl[3, 3] = 4.0e-4
        rn = zeros(3, nk); rn[2, 4] = -3.0e-6
        re = fill(0.2, 3, nk); re[2, 3] = -0.7
        s2 = merge(slots, (; cloud = cl, rain = rn, rho_e = re))
        # untransformed: Lambda is an exact zero field, rho_w the plain sum
        dn = Scythe.mc_derived_fields(z, s2, ref, cfg())
        @test all(x -> x === 0.0, dn.lobe_pool)
        @test dn.rho_w == (slots.rho_v .+ cl) .+ rn
        @test dn.rho_e_deficit == min.(re, 0.0)
        @test count(<(0.0), dn.rho_e_deficit) == 1
        # transformed cloud and rain: Lambda = sum_k 2 min(nu_k, 0) over the transformed slots
        dt = Scythe.mc_derived_fields(z, s2, ref, cfg(ctrans = :bhyp, rtrans = :bhyp))
        @test dt.lobe_pool == 2.0 .* min.(cl, 0.0) .+ 2.0 .* min.(rn, 0.0)
        @test dt.rho_w == (slots.rho_v .+ 2.0 .* cl) .+ 2.0 .* rn
        @test dt.column_lobe_pool == Scythe._column_integral(dt.lobe_pool, z)
        # the partition gap's identity X = B + Lambda: X - Lambda = B >= 0 (the knee bias)
        X = dt.rho_w .- ((dt.rho_v .+ dt.rho_c) .+ dt.rho_r)
        @test all(X .- dt.lobe_pool .>= -1.0e-18)
    end

    # ══ E1. the option, the name, the accessors ══════════════════════════════════════
    @testset "E1: option, slot name, alias, width and refusals" begin
        E = Dict{Symbol,Any}
        @test Scythe.tke_transform_mode(E()) === :none
        @test Scythe.tke_transform_mode(E(:tke_transform => :none)) === :none
        @test Scythe.tke_transform_mode(E(:tke_transform => :bhyp)) === :bhyp
        @test_throws ErrorException Scythe.tke_transform_mode(E(:tke_transform => :bhyp_smooth))
        @test_throws ErrorException Scythe.tke_transform_mode(E(:tke_transform => :log))
        @test Scythe.TKE_MU_DEFAULT == 1.0e-7
        @test Scythe.tke_mu(E()) == 1.0e-7
        @test Scythe.tke_mu(E(:tke_mu => 3.0e-8)) == 3.0e-8
        @test Scythe.MC_NU_ALIAS["rho_e"] == "nu_e"
        @test Scythe.tke_var_name(E()) == "rho_e"
        @test Scythe.tke_var_name(E(:tke_transform => :bhyp)) == "nu_e"
        for cyl in (false, true)
            n0 = Scythe.mc_var_names(E(:mynn => true); cyl)
            n1 = Scythe.mc_var_names(E(:mynn => true, :tke_transform => :bhyp); cyl)
            @test n0[end] == "rho_e" && n1[end] == "nu_e"
            @test !("rho_e" in n1) && !("nu_e" in n0)
            @test n0[1:end-1] == n1[1:end-1]
            V = Dict(v => i for (i, v) in enumerate(n1))
            @test Scythe.mc_slot(V, "rho_e") == V["nu_e"]
            @test Scythe.mc_optional_slot(V, "rho_e") == V["nu_e"]
            # without the closure there is no slot to rename
            @test Scythe.mc_var_names(E(:tke_transform => :bhyp); cyl) ==
                  Scythe.mc_var_names(E(); cyl)
        end
        # the recovery: the slot itself under :none (bitwise, signed), ahyp under :bhyp
        mu = 1.0e-7
        @test Scythe.recover_rho_e(-0.3, :none, mu) === -0.3
        @test Scythe.recover_rho_e(0.25, :none, mu) === 0.25
        @test Scythe.recover_rho_e(-0.3, :bhyp, mu) === 0.0
        @test Scythe.recover_rho_e(0.0, :bhyp, mu) === 0.0
        for r in (0.0, 1.0e-9, 1.0e-7, 4.5e-5, 0.4, 6.0)
            @test Scythe.recover_rho_e(Scythe.bhyp(r, mu), :bhyp, mu) ≈ r rtol = 1.0e-14 atol = 1.0e-30
        end
        # the registry and the run configuration
        @test haskey(Scythe.KNOWN_OPTION_KEYS, :tke_transform)
        @test haskey(Scythe.KNOWN_PHYSICAL_KEYS, :tke_mu)
        @test Scythe.BoundaryLayer().tke_transform === :none
        @test Scythe.BoundaryLayer().tke_mu == 1.0e-7
        @test Scythe.OWNED_OPTION_KEYS[:tke_transform] == "physics.boundary_layer.tke_transform"
        @test Scythe.OWNED_PHYSICAL_KEYS[:tke_mu] == "physics.boundary_layer.tke_mu"
        o0, p0 = Scythe.build_physics(Scythe.load_config())
        @test !haskey(o0, :tke_transform) && !haskey(p0, :tke_mu)   # :none writes nothing
        o1, p1 = Scythe.build_physics(Scythe.load_config(
            set = ["physics.boundary_layer.tke_transform=bhyp",
                   "physics.boundary_layer.tke_mu=5.0e-8"]))
        @test o1[:tke_transform] === :bhyp
        @test p1[:tke_mu] == 5.0e-8
        @test_throws ErrorException Scythe.load_config(
            set = ["physics.boundary_layer.tke_transform=bhyp_smooth"])
        @test_throws ErrorException Scythe.validate(Scythe.load_config(
            set = ["physics.boundary_layer.tke_transform=bhyp",
                   "physics.boundary_layer.scheme=louis"]))
        @test_throws ErrorException Scythe.validate(Scythe.load_config(
            set = ["physics.boundary_layer.tke_transform=bhyp",
                   "physics.boundary_layer.tke_mu=0.0"]))
        # the tile: the slot is nu_e, resolved by role, and the MYNN state carries the mode
        M = Main.MYNNTestFixtures
        mktempdir() do tmp
            m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                extra_options = E(:tke_transform => :bhyp))
            V = mo.grid_params.vars
            @test haskey(V, "nu_e") && !haskey(V, "rho_e")
            @test m.mc_slots.rho_e == V["nu_e"]
            @test m.mynn.tke_transform === :bhyp
            @test m.mynn.tke_mu == 1.0e-7
            m0, _, _, _, _, _ = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3)
            @test m0.mynn.tke_transform === :none
            # refusals at setup: a knee above 1e-2 of the closure floor in the mixing layer,
            # a non-positive width, and the transform declared without the closure
            @test_throws ErrorException M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                extra_options = E(:tke_transform => :bhyp), extra_params = E(:tke_mu => 1.0e-5))
            @test_throws ErrorException M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                extra_options = E(:tke_transform => :bhyp), extra_params = E(:tke_mu => 0.0))
            @test_throws ErrorException M.make_mynn_tile(tmp; mynn = false, num_cells_k = 25,
                kMax = 5.0e3, extra_options = E(:tke_transform => :bhyp))
        end
    end

    # ══ E2. staging and tendencies ═════════════════════════════════════════════════════
    M = Main.MYNNTestFixtures
    F = Main.VaporTransformFixtures
    EMU = 1.0e-7
    E = Dict{Symbol,Any}
    "The TKE slot's recovered density (the reader's view) under either mode."
    rec(slot, etr) = etr === :none ? copy(slot) : Scythe.ahyp.(slot, EMU)

    @testset "E2: :none and the absent key reproduce the MYNN goldens bitwise" begin
        # the MYNN arm of test_vapor_transform.jl's :none capture (re-seeded bf577b2)
        golden = (UInt128(259941565336452810300720602), UInt128(9083517470135141505831894328))
        mktempdir() do tmp
            for etr in (nothing, :none)
                xo = E()
                etr === nothing || (xo[:tke_transform] = etr)
                m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                                                        ctrans = :none, rtrans = :none,
                                                        extra_options = xo)
                M.set_state!(p, mo, gp.kDim, z, col; W0 = 0.3)
                for t in 1:3
                    M.step_all!(m, p, gp.kDim, t); Scythe.calcTendency(m); gridTransform!(p)
                end
                V = mo.grid_params.vars
                @test F.s4_checksum(m.var_np1, V) == golden[1]
                @test F.s4_checksum(p.physical, V) == golden[2]
            end
        end
    end

    @testset "E2: rest (rho_e = 0, no wind, no fluxes): every TKE tendency is the :none one, zero" begin
        mktempdir() do tmp
            out = Dict{Symbol,Any}()
            for etr in (:none, :bhyp)
                m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                    init = :zero, fluxes = false, extra_options = E(:tke_transform => etr))
                kDim = gp.kDim
                M.set_state!(p, mo, kDim, z, col; V0 = 0.0, e0 = 0.0, qpert = 0.0)
                re = m.mc_slots.rho_e
                @test all(iszero, p.physical[:, re, 1])
                M.step_all!(m, p, kDim, 1)
                out[etr] = copy(m.expdot_n[:, re])
                out[Symbol(etr, :_all)] = copy(m.expdot_n)
            end
            @test all(iszero, out[:bhyp])
            @test all(out[:bhyp] .=== out[:none])          # sign of zero included
            # ... and the transform touches no other slot at rest
            @test out[:bhyp_all][:, 1:end-1] == out[:none_all][:, 1:end-1]
        end
    end

    @testset "E2: recovered rho_e under :bhyp tracks :none over 1000 steps" begin
        mktempdir() do tmp
            out = Dict{Symbol,Any}()
            for etr in (:none, :bhyp)
                m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                    extra_options = E(:tke_transform => etr))
                kDim = gp.kDim
                M.set_state!(p, mo, kDim, z, col; W0 = 0.3)
                for t in 1:1000
                    M.step_all!(m, p, kDim, t); Scythe.calcTendency(m); gridTransform!(p)
                end
                @test all(isfinite, p.physical)
                out[etr] = rec(p.physical[:, m.mc_slots.rho_e, 1], etr)
                out[Symbol(etr, :_K)] = copy(m.mynn.K_m)
            end
            rn = out[:none]; rb = out[:bhyp]
            # the :none column stayed positive (so the comparison is the transform's, not a
            # lobe's) and the closure moved it
            @test all(>(0.0), rn)
            sc = maximum(rn)
            live = rn .> 1.0e-4 * sc
            relerr = maximum(abs.(rb[live] .- rn[live]) ./ rn[live])
            abserr = maximum(abs.(rb .- rn)) / sc
            @info "TKE transform tracking over 1000 steps" relerr abserr
            @test relerr <= 1.0e-9
            @test abserr <= 1.0e-9
            @test maximum(abs.(out[:bhyp_K] .- out[:none_K])) <= 1.0e-8 * maximum(out[:none_K])
        end
    end

    @testset "E2: the D3 column identity closes on the RECOVERED TKE rate under :bhyp" begin
        # sum_k w_k (dE_t + drho_e) = bdry_E, with drho_e the slot's tendency read back
        # through J_e (the transport of the TKE slot is identically zero on this fixture)
        mktempdir() do tmp
            m_on, p_on, mo, gpo, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, ts = 0.5,
                extra_options = E(:tke_transform => :bhyp))
            m_off, p_off, mf, _, _, _ = M.make_mynn_tile(tmp; mynn = false, num_cells_k = 25,
                                                         ts = 0.5)
            kDim = gpo.kDim
            M.set_state!(p_on, mo, kDim, z, col)
            M.set_state!(p_off, mf, kDim, z, col)
            M.step_all!(m_on, p_on, kDim); M.step_all!(m_off, p_off, kDim)
            re = m_on.mc_slots.rho_e
            wq = M.gauss_weights(kDim, 25, gpo.kMax - gpo.kMin)
            c = 6
            rng = ((c - 1) * kDim + 1):(c * kDim)
            nu = p_on.physical[rng, re, 1]
            J = Scythe.dbhyp.(Scythe.ahyp.(nu, EMU), EMU)
            @test all(>(0.0), nu)
            dE = m_on.expdot_n[rng, 6] .- m_off.expdot_n[rng, 6]
            de = m_on.expdot_n[rng, re] ./ J
            r = sum(wq .* (dE .+ de)) - m_on.mynn.bdry_E[c]
            scale = sum(wq .* abs.(dE)) + sum(wq .* abs.(de))
            @test abs(r) <= 1.0e-12 * scale
        end
    end

    @testset "E2: every reader sees the recovered rho_e; a nu_e lobe is counted" begin
        mktempdir() do tmp
            for etr in (:none, :bhyp)
                m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                    extra_options = E(:tke_transform => etr, :state_minima_trace => 1))
                kDim = gp.kDim
                M.set_state!(p, mo, kDim, z, col; W0 = 0.3)
                re = m.mc_slots.rho_e
                _tke_seed_lobe!(p, re, z, kDim; amp = 0.5)
                slot = copy(p.physical[:, re, 1])
                nneg = count(<(0.0), slot)
                @test nneg > 0
                M.step_all!(m, p, kDim, 1)
                # the closure's transport column of the LAST column advanced: e = rho_e/rho_t
                # read from the recovered density
                MN = m.mynn.colscratch[Threads.threadid()]
                ncol = div(size(p.physical, 1), kDim)
                lastc = ((ncol - 1) * kDim + 1):(ncol * kDim)
                if etr === :bhyp
                    @test all(>=(0.0), MN.e[1:kDim])
                    @test m.species.rho[re, Threads.threadid()] == Scythe.ahyp.(slot[lastc], EMU)
                else
                    @test any(<(0.0), MN.e[1:kDim])
                end
                @test sum(m.mynn.n_eneg_col) == nneg            # the clamp split, either mode
                # the census: under :bhyp the lobe's energy is the 2 nu weighting
                Scythe.energy_census_trace(m, 11)
                eb = m.ebudget
                a = etr === :none ? 1.0 : 2.0
                expect = sum(eb.wxn[div(g - 1, kDim) + 1] * (eb.wz[mod1(g, kDim)] * (a * slot[g]))
                             for g in eachindex(slot) if slot[g] < 0.0)
                @test m.mynn.n_eneg_pts == nneg
                @test m.mynn.eneg_mass ≈ expect rtol = 1.0e-12
                # the sidecar's e: recovered / rho_t, never negative under :bhyp
                d = Scythe.mynn_diagnostics(m)
                etr === :bhyp ? (@test all(>=(0.0), d.e)) : (@test any(<(0.0), d.e))
                # the minima trace: the recovered minimum and the slot's negative points
                tr = _tke_capture(() -> Scythe.state_minima_trace(m, 1))
                if etr === :bhyp
                    @test occursin("min rho_e=0.0 J/m^3", tr)
                    @test occursin("nu_e<0 points=$(nneg)", tr)
                else
                    @test occursin("rho_e<0 points=$(nneg)", tr)
                end
                # the netCDF file: recovered rho_e and e >= 0, the raw nu_e beside them,
                # rho_e_deficit the lobe under ahyp_signed
                ctx = Scythe.netcdf_output_context(p, mo)
                path = joinpath(tmp, "rd_$(etr).nc")
                Scythe.write_netcdf_comprehensive(path, p, mo, 1.0, ctx)
                NCDataset(path, "r") do ds
                    re_nc = Array(ds["rho_e"])
                    if etr === :bhyp
                        @test haskey(ds, "nu_e")
                        nu_nc = Array(ds["nu_e"])
                        @test all(>=(0.0), re_nc)
                        @test all(>=(0.0), Array(ds["e"]))
                        @test re_nc == Scythe.ahyp.(nu_nc, EMU)
                        @test any(<(0.0), nu_nc)
                        @test Array(ds["rho_e_deficit"]) ==
                              map(x -> x < 0.0 ? Scythe.ahyp_signed(x, EMU) : 0.0, nu_nc)
                    else
                        @test !haskey(ds, "nu_e")
                        @test any(<(0.0), re_nc)
                        @test Array(ds["rho_e_deficit"]) == min.(re_nc, 0.0)
                    end
                end
            end
        end
    end

    @testset "E2: the transformed MYNN column step is allocation-free" begin
        mktempdir() do tmp
            m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                extra_options = E(:tke_transform => :bhyp))
            M.set_state!(p, mo, gp.kDim, z, col; W0 = 0.3)
            M.step_all!(m, p, gp.kDim, 1)
            kDim = gp.kDim
            # (the driver entry, as test_allocations.jl measures it: `advance_column`'s own
            # dispatch allocates one object under :none too)
            Scythe.moist_compressible_XZ(m, 1, kDim, 2)
            @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 3)) == 0
        end
    end
end
