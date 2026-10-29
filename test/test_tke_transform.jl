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
end
