using Test
using Scythe
using CSV
using DataFrames

# Stream C2/R2e: the 3-D Cartesian box (moist_compressible_RRR: spline x, y, z) TIME-STEPPED
# end to end through the run driver -- the sibling of test_3d_timestepping.jl (the RLR
# cylinder, C1). Four run_simulation arms, launched concurrently (each is a fresh child whose
# cost is the driver compile), all 40 steps at ts 0.5 s under the default physics (ice +
# MYNN + two-moment rain; radiation none), f = 0:
#
#   xz      the XZ slice: 24 km x 4 km, 6 x 8 cells, a warm bubble at x = 12 km.
#   slab    the RRR box 24 x 12 x 4 km, 6 x 4 x 8 cells, started from the XZ initial state
#           copied to every y (initial_conditions.method = "file"; v = 0). The box's walls are
#           free-slip (w Neumann on all four faces); the slice holds w on its x walls, so the
#           slab overrides w to Dirichlet on its x faces. The exact answer is y-invariant and
#           equal to the slice.
#   sym1    the RRR square box 24 x 24 x 4 km, 6 x 6 x 8 cells, a bubble centred at (12, 12)
#           km on 1 worker, the box's default (free-slip) walls. The exact answer is
#           symmetric under x <-> y with u <-> v.
#   sym2    sym1 on 2 workers (two x tiles of 3 cells): the RRR tile exchange (stage R2c).
#
# Gates (close-out plan, stage R2e): slab == XZ <= 1e-9 scaled; x<->y symmetry <= 1e-10;
# 2 workers vs 1 <= 1e-12. 20 s of a 3 K bubble is well before glaciation, so the ice slots
# are held to the same gates (C1: ISHMAEL amplifies roundoff only after glaciation).

const R3D_SOUNDING = "@scythe/benchmarks/reference_data/o01_rainfall/dunion_MT_hum90.ref"

function r3d_toml(name, es, geom, outdir; jaxis = "", ic = "", bc_extra = "")
    return """
    [run]
    name             = "$name"
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
    $jaxis
    z_top       = 4000.0
    num_cells_k = 8

    [boundary_conditions]
    lateral = "walls"
    $bc_extra

    [timestep]
    ts = 0.5

    [reference_state]
    sounding = "$R3D_SOUNDING"

    $ic

    [environment]
    f_latitude = 0.0
    sst        = 302.65

    [physics.radiation]
    scheme = "none"
    """
end

const R3D_BUBBLE = """
[initial_conditions]
method = "temperature_bubble"

[initial_conditions.temperature_bubble]
xc     = 12.0e3
xr     = 8.0e3
zc     = 500.0
zr     = 1000.0
dT_max = 3.0
"""

"Run one arm; return the final `<t>_physical.csv`, the prognostic names and the final time."
function r3d_run(tmp, name, toml; workers = 1)
    out = joinpath(tmp, name)
    p = joinpath(tmp, "$(name).toml")
    write(p, toml(out))
    cfg = load_config(p)
    vars = first.(sort(collect(build_model(cfg).grid_params.vars), by = last))
    exec = ExecutionEnv(workers_per_patch = [workers], worker_threads = 1)
    t = @elapsed run_simulation(cfg, exec; config_sources = [p])
    println("3D RRR timestepping $(name): $(round(t; digits = 1)) s")
    seg = joinpath(out, "seg_01")
    files = filter(f -> endswith(f, "_physical.csv"), readdir(seg))
    tags = [parse(Float64, replace(f, "_physical.csv" => "")) for f in files]
    df = CSV.read(joinpath(seg, files[argmax(tags)]), DataFrame)
    return (df = df, vars = vars, t = maximum(tags))
end

@testset "3D RRR time stepping (slab == XZ, x<->y symmetry, 2 workers)" begin
    # SCYTHE_R3D_KEEP=<dir> keeps the four run trees there (for inspection) instead of a
    # temporary directory.
    keep = get(ENV, "SCYTHE_R3D_KEEP", "")
    tmp = isempty(keep) ? mktempdir() : mkpath(keep)
    t_wall = time()

    # The slab's initial state: the XZ arm's own initial condition (built in-process by
    # the same initialize_case! the XZ child runs) copied to every y of the box.
    xz_toml(out) = r3d_toml("r3d_xz", "moist_compressible_XZ", "RiRk", out; ic = R3D_BUBBLE)
    xzpre = joinpath(tmp, "xz_ics")
    write(joinpath(tmp, "xz_ics.toml"), xz_toml(xzpre))
    xcfg = load_config(joinpath(tmp, "xz_ics.toml"))
    xmodel = build_model(xcfg)
    redirect_stdout(devnull) do
        Scythe.initialize_case!(xcfg, xmodel)
    end
    xz0 = CSV.read(xmodel.initial_conditions, DataFrame)
    slab_ics = joinpath(tmp, "slab_ics.csv")
    slab_jaxis = "boundaries_j = [0.0, 12.0e3]\nnum_cells_j = [4]"
    # The box's walls are free-slip (w Neumann on all four faces); the slice's x walls hold
    # w, so the slab holds it on its x faces too and the two runs are the same problem.
    slab_bc = "[boundary_conditions.override]\nw = { inner = \"dirichlet\", outer = \"dirichlet\" }"
    slab_toml(out) = r3d_toml("r3d_slab", "moist_compressible_RRR", "RRR", out;
        jaxis = slab_jaxis, bc_extra = slab_bc,
        ic = "[initial_conditions]\nmethod = \"file\"\n\n[initial_conditions.file]\npaths = [\"$slab_ics\"]")
    # The box's own mish, from a throw-away build (the file is read in place, so the
    # RunConfig needs it to exist: write a placeholder first).
    write(slab_ics, "")
    write(joinpath(tmp, "slab_probe.toml"), slab_toml(joinpath(tmp, "slab_probe")))
    smodel = build_model(load_config(joinpath(tmp, "slab_probe.toml")))
    spatch = Scythe.createGrid(smodel.grid_params)
    sp = Scythe.getGridpoints(spatch)
    svars = first.(sort(collect(smodel.grid_params.vars), by = last))
    xrow = Dict((xz0.r[i], xz0.z[i]) => i for i in 1:nrow(xz0))
    open(slab_ics, "w") do io
        println(io, join(vcat(["r", "l", "z"], svars), ","))
        for i in axes(sp, 1)
            j = xrow[(sp[i, 1], sp[i, 3])]
            vals = [v == "v" ? 0.0 : xz0[j, v] for v in svars]
            println(io, join(string.([sp[i, 1], sp[i, 2], sp[i, 3], vals...]), ","))
        end
    end

    sym_jaxis = "boundaries_j = [0.0, 24.0e3]\nnum_cells_j = [6]"
    sym_ic = replace(R3D_BUBBLE, "xc     = 12.0e3" => "xc     = 12.0e3\nyc     = 12.0e3")
    sym_toml(name) = out -> r3d_toml(name, "moist_compressible_RRR", "RRR", out;
                                     jaxis = sym_jaxis, ic = sym_ic)

    arms = Dict{String,Any}()
    @sync begin
        @async arms["xz"] = r3d_run(tmp, "xz", xz_toml)
        @async arms["slab"] = r3d_run(tmp, "slab", slab_toml)
        @async arms["sym1"] = r3d_run(tmp, "sym1", sym_toml("r3d_sym1"); workers = 1)
        @async arms["sym2"] = r3d_run(tmp, "sym2", sym_toml("r3d_sym2"); workers = 2)
    end
    xz, slab, sym1, sym2 = arms["xz"], arms["slab"], arms["sym1"], arms["sym2"]
    vars = slab.vars
    @test xz.t == 20.0 && slab.t == 20.0 && sym1.t == 20.0 && sym2.t == 20.0
    @test vars == sym1.vars == sym2.vars && vars[10] == "v"
    @test any(startswith(v, "nu_i") for v in vars)            # ice is on (the default)
    @test setdiff(vars, xz.vars) == ["v"]
    for a in (xz, slab, sym1, sym2)
        @test all(isfinite, Matrix(a.df))
    end

    # ── slab == XZ, pointwise at every y
    wind = max(maximum(abs, xz.df.u), maximum(abs, xz.df.w), 1.0e-10)
    xscale(v) = max(maximum(abs, xz.df[!, v]), 1.0e-10)
    xi = Dict((xz.df.r[i], xz.df.z[i]) => i for i in 1:nrow(xz.df))
    slabd = Dict{String,Float64}()
    for v in xz.vars
        d = 0.0
        for i in 1:nrow(slab.df)
            d = max(d, abs(slab.df[i, v] - xz.df[xi[(slab.df.r[i], slab.df.z[i])], v]))
        end
        slabd[v] = d / xscale(v)
    end
    vslab = maximum(abs, slab.df.v) / wind
    @info "3D RRR slab vs XZ (scaled): " *
          join(("$v=$(round(slabd[v]; sigdigits = 3))" for v in xz.vars), " ") *
          "; max|v|/wind = $vslab; max w = $(maximum(xz.df.w)) m/s"
    @test maximum(xz.df.w) > 0.01                              # the bubble is rising
    @test maximum(values(slabd)) <= 1.0e-9
    @test vslab <= 1.0e-9

    # ── x <-> y symmetry of the centred bubble (u <-> v)
    df = sym1.df
    idx = Dict((df.r[i], df.l[i], df.z[i]) => i for i in 1:nrow(df))
    mir = [idx[(df.l[i], df.r[i], df.z[i])] for i in 1:nrow(df)]
    swap = Dict("u" => "v", "v" => "u")
    swind = max(maximum(abs, df.u), maximum(abs, df.w), 1.0e-10)
    sscale(v) = v in ("u", "v") ? swind : max(maximum(abs, df[!, v]), 1.0e-10)
    symd = Dict(v => maximum(abs.(df[!, v] .- df[mir, get(swap, v, v)])) / sscale(v) for v in vars)
    @info "3D RRR x<->y symmetry (scaled): " *
          join(("$v=$(round(symd[v]; sigdigits = 3))" for v in vars), " ")
    @test maximum(df.w) > 0.01
    @test maximum(values(symd)) <= 1.0e-10

    # ── 2 workers reproduce 1 worker
    @test sym1.df.r == sym2.df.r && sym1.df.l == sym2.df.l && sym1.df.z == sym2.df.z
    w12 = maximum(maximum(abs, sym1.df[!, v] .- sym2.df[!, v]) / sscale(v) for v in vars)
    @info "3D RRR 2-worker vs 1-worker (scaled max) = $w12"
    @test w12 <= 1.0e-12
    println("3D RRR timestepping: wall time $(round(time() - t_wall; digits = 1)) s")
end
