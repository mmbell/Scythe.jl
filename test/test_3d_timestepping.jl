using Test
using Scythe
using CSV
using DataFrames

# Stream C1 of the 3-D warm-bubble plan: the RLR (spline-r, Fourier-lambda, spline-z) set
# TIME-STEPPED end to end through the run driver, not just one tendency evaluation
# (test_moist_compressible.jl, "RLR: WN0 tendencies match the axisymmetric set"). An
# axis-centred warm bubble with f = 0 is an exactly axisymmetric problem, so the RLR solution
# must stay WN0 (every ring constant to roundoff), carry v = 0, and its ring means must track
# the axisymmetric set on the matched RiRk grid (the radial and vertical mish nodes of the two
# grids are bitwise identical; the trajectories are NOT, since the ring FFT seeds roundoff).
# The 2-worker RLR run must reproduce the 1-worker run (tiled transform + shared-array
# exchange). Full default physics: ice (ISHMAEL) + MYNN + two-moment rain; radiation off.
#
# run_simulation launches a child Julia process per segment, which adds its own workers, so
# this file lives in the `:nesting` group. Three concurrent runs of 40 steps at ts = 0.5 s on a 24 km x
# 4 km, 6 x 8 cell grid (500 m vertical cells, the smoke's spacing). NOT 2 km x 8 cells: at
# 250 m cells ts = 0.5 s is above the vertical semi-implicit ceiling (acoustic Co_z 3.1) and
# the AXISYMMETRIC set itself blows up at the lowest level within 20 s (p' -24 kPa, w 13 m/s;
# BL and surface fluxes off changes nothing) -- measured 2026-10-28, see
# reference/session_2026-10_scratch/closeout_2026-10-27/r1_rlr/README.md.

const T3D_SOUNDING = "@scythe/benchmarks/reference_data/o01_rainfall/dunion_MT_hum90.ref"

t3d_toml(es, geom, outdir) = """
[run]
name             = "t3d_$(geom)"
equation_set     = "$es"
integration_time = 20.0
segment_length   = 0.0

[output]
dir      = "$outdir"
interval = 10.0
formats  = ["csv"]
sidecars = false

[grid]
geometry    = "$geom"
boundaries  = [0.0, 24.0e3]
num_cells   = [6]
z_top       = 4000.0
num_cells_k = 8

[boundary_conditions]
lateral = "walls"

[timestep]
ts = 0.5

[reference_state]
sounding = "$T3D_SOUNDING"

[initial_conditions]
method = "temperature_bubble"

[initial_conditions.temperature_bubble]
xc     = 0.0
xr     = 8.0e3
zc     = 500.0
zr     = 1000.0
dT_max = 3.0

[environment]
f_latitude = 0.0
sst        = 302.65

[physics.radiation]
scheme = "none"
"""

"Run one arm; return the final `<t>_physical.csv` as a DataFrame and the prognostic names."
function t3d_run(tmp, name, es, geom; workers = 1)
    out = joinpath(tmp, name)
    p = joinpath(tmp, "$(name).toml")
    write(p, t3d_toml(es, geom, out))
    cfg = load_config(p)
    vars = first.(sort(collect(build_model(cfg).grid_params.vars), by = last))
    exec = ExecutionEnv(workers_per_patch = [workers], worker_threads = 1)
    t = @elapsed run_simulation(cfg, exec; config_sources = [p])
    println("3D timestepping $(name): $(round(t; digits = 1)) s")
    seg = joinpath(out, "seg_01")
    files = filter(f -> endswith(f, "_physical.csv"), readdir(seg))
    tags = [parse(Float64, replace(f, "_physical.csv" => "")) for f in files]
    tfin = maximum(tags)
    df = CSV.read(joinpath(seg, files[argmax(tags)]), DataFrame)
    return (df = df, vars = vars, t = tfin)
end

"Per (r, z) ring: mean, and max - min, of column `v` of an RLR physical CSV."
function t3d_rings(df, v)
    g = Dict{Tuple{Float64,Float64},Vector{Float64}}()
    for (r, z, x) in zip(df.r, df.z, df[!, v])
        push!(get!(g, (r, z), Float64[]), x)
    end
    return Dict(k => (sum(x) / length(x), maximum(x) - minimum(x)) for (k, x) in g)
end

@testset "3D RLR time stepping (axis-centred warm bubble)" begin
    tmp = mktempdir()
    # The three arms run CONCURRENTLY. Each run_simulation launches a fresh child Julia
    # process (plus its own workers) into its own output dir, and that child's cost is almost
    # entirely compiling the driver (~110-160 s each); the 40 steps take seconds. Run one
    # after another they cost ~7 min; overlapped, the wall time is the slowest arm. The
    # master only waits on the child processes, so the @async tasks do not contend.
    # Results are collected after the @sync; a failing arm surfaces as its exception.
    arms = Dict{String,Any}()
    @sync begin
        @async arms["rlr_w1"] = t3d_run(tmp, "rlr_w1", "moist_compressible_RLR", "RLR"; workers = 1)
        @async arms["rlr_w2"] = t3d_run(tmp, "rlr_w2", "moist_compressible_RLR", "RLR"; workers = 2)
        @async arms["axisym"] = t3d_run(tmp, "axisym", "moist_compressible_axisym", "RiRk"; workers = 1)
    end
    rlr1, rlr2, ax = arms["rlr_w1"], arms["rlr_w2"], arms["axisym"]
    vars = rlr1.vars
    @test rlr1.t == 20.0 && rlr2.t == 20.0 && ax.t == 20.0
    @test vars == ax.vars == rlr2.vars
    @test any(startswith(v, "nu_i") for v in vars)           # ice is on (the default)

    # Every column finite, every slot (ice included) and every derivative slot.
    @test all(isfinite, Matrix(rlr1.df))
    @test all(isfinite, Matrix(rlr2.df))
    @test all(isfinite, Matrix(ax.df))

    # Per-slot scale: the field's max |.| over both geometries, floored so a slot that is
    # (near) zero everywhere is held to an absolute standard instead of a ratio of zeros.
    # v is identically 0 in the exact answer, so its own max is roundoff: v is measured
    # against the wind scale (max |u|, |w|), as the roundoff it carries comes from there.
    wind = max(maximum(abs, rlr1.df.u), maximum(abs, rlr1.df.w), 1.0e-10)
    scale(v) = v == "v" ? wind :
               max(maximum(abs, rlr1.df[!, v]), maximum(abs, ax.df[!, v]), 1.0e-10)

    # Azimuthal symmetry: every ring constant to roundoff.
    spread = Dict(v => maximum(last(x) for x in values(t3d_rings(rlr1.df, v))) / scale(v)
                  for v in vars)
    @info "3D RLR ring spread (scaled): " *
          join(("$v=$(round(spread[v]; sigdigits = 3))" for v in vars), " ")
    @test spread["w"] <= 1.0e-9
    @test spread["nu_r"] <= 1.0e-9
    @test spread["E_t"] <= 1.0e-9
    @test maximum(values(spread)) <= 1.0e-9                 # ice slots symmetric too

    # v = 0 is the exact answer (f = 0, axis-centred bubble).
    vmax = maximum(abs, rlr1.df.v)
    @info "3D RLR max|v| = $vmax (axisym: $(maximum(abs, ax.df.v)))"
    @test vmax <= 1.0e-12
    @test maximum(abs, ax.df.v) == 0.0

    # WN0 ring means vs the axisymmetric run on the matched RiRk grid, pointwise.
    axidx = Dict((r, z) => i for (i, (r, z)) in enumerate(zip(ax.df.r, ax.df.z)))
    wn0 = Dict{String,Float64}()
    for v in vars
        rings = t3d_rings(rlr1.df, v)
        @test length(rings) == nrow(ax.df)
        d = 0.0
        for (k, (m, _)) in rings
            @test haskey(axidx, k)
            d = max(d, abs(m - ax.df[axidx[k], v]))
        end
        wn0[v] = d / scale(v)
    end
    @info "3D RLR WN0 vs axisym (scaled): " *
          join(("$v=$(round(wn0[v]; sigdigits = 3))" for v in vars), " ")
    @test maximum(values(wn0)) <= 1.0e-9

    # 2 workers reproduce 1 worker.
    @test rlr1.df.r == rlr2.df.r && rlr1.df.l == rlr2.df.l && rlr1.df.z == rlr2.df.z
    w12 = maximum(maximum(abs, rlr1.df[!, v] .- rlr2.df[!, v]) / scale(v) for v in vars)
    @info "3D RLR 2-worker vs 1-worker (scaled max) = $w12"
    @test w12 <= 1.0e-12
end
