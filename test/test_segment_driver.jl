using Test
using Scythe

# Stage 3b of the configuration plan: the segment driver end to end. `run_simulation`
# launches one child Julia process per segment (`Scythe.segment_main`), each of which adds
# its own worker(s), so this file lives in the `:nesting` group (it spawns processes). Two
# 10 s segments on tiny grids, radiation off:
# (a) a single-patch XZ warm bubble, (b) a 2-patch axisymmetric nest.

const SIM_DIR = normpath(joinpath(@__DIR__, "..", "simulations"))

"Value of a `# key = value` line in a scythe_config.toml header, or `nothing`."
function header_value(path, key)
    for l in eachline(path)
        startswith(l, "#") || break
        m = match(Regex("^# " * key * " = (.*)\$"), l)
        m === nothing || return String(m.captures[1])
    end
    return nothing
end

"Common checks on a finished 2 x 10 s run in `root` with per-patch dirs under each segment."
function check_two_segment_run(root, npatches)
    rows = Scythe.read_segments_csv(root)
    @test sort(collect(keys(rows))) == [1, 2]
    @test all(rows[k]["status"] == "done" for k in 1:2)
    @test parse(Float64, rows[1]["t_offset_s"]) == 0.0
    @test parse(Float64, rows[2]["t_offset_s"]) == 10.0
    @test all(parse(Float64, rows[k]["integration_time_s"]) == 10.0 for k in 1:2)
    @test all(rows[k]["n_patches"] == string(npatches) for k in 1:2)
    @test all(length(split(rows[k]["ts_actual"], ';')) == npatches for k in 1:2)
    @test parse(Float64, rows[2]["start_hour"]) ≈ 10.0 / 3600.0
    for k in 1:2
        segdir = joinpath(root, Scythe.segment_dirname(k))
        @test isdir(segdir)
        @test isfile(joinpath(segdir, "segment_stdout.log"))
        @test isfile(joinpath(segdir, "scythe_config.toml"))
        for d in Scythe.segment_patch_dirs(root, k, npatches)
            @test isfile(checkpoint_path(d, 10.0))
        end
    end
    # segment 2 chained from each patch's segment-1 final checkpoint
    seg2cfg = joinpath(root, "seg_02", "scythe_config.toml")
    expected = [checkpoint_path(d, 10.0) for d in Scythe.segment_patch_dirs(root, 1, npatches)]
    @test header_value(seg2cfg, "initial_conditions") == join(expected, ";")
    @test parse(Float64, header_value(seg2cfg, "start_hour")) ≈ 10.0 / 3600.0
    @test parse(Float64, header_value(seg2cfg, "t_offset_s")) == 10.0
    @test header_value(joinpath(root, "seg_01", "scythe_config.toml"), "initial_conditions") ==
          join(Scythe.initial_ics_paths(root, npatches), ";")
    # the root config re-reads to the configuration that ran
    @test isfile(joinpath(root, "scythe_config.toml")) && isfile(joinpath(root, "reference.ref"))
    # the resolved boundary conditions: one [[patch]] per patch, every variable listed
    bcf = joinpath(root, "boundary_conditions.toml")
    @test isfile(bcf)
    bct = Scythe.TOML.parsefile(bcf)
    @test length(bct["patch"]) == npatches
    @test all(any(v -> v["name"] == "w" && v["bottom"] == "dirichlet", p["variable"])
              for p in bct["patch"])
    return rows
end

"Resume of a finished run re-runs nothing; a fresh start on the same root is refused."
function check_resume_and_refusal(root, cfg, exec)
    before = read(joinpath(root, "segments.csv"), String)
    mt = mtime(checkpoint_path(Scythe.segment_patch_dirs(root, 2, Scythe.npatches(cfg))[1], 10.0))
    @test run_simulation(cfg, exec; resume = true) === nothing
    @test read(joinpath(root, "segments.csv"), String) == before
    @test mtime(checkpoint_path(Scythe.segment_patch_dirs(root, 2, Scythe.npatches(cfg))[1],
                                10.0)) == mt
    err = try
        run_simulation(cfg, exec)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("refusing a fresh start", err)
end

tiny = ["run.integration_time=20", "run.segment_length=10", "output.interval=10",
        "output.restart_interval=10", "physics.radiation.scheme=none",
        "physics.microphysics.ice=none", "timestep.ts=0.5"]

@testset "Segment driver" begin

    @testset "single-patch XZ, 2 segments" begin
        root = mktempdir()
        set = [tiny; "grid.num_cells=[8]"; "output.dir=$root"]
        src = joinpath(SIM_DIR, "ocean_warm_bubble", "scythe_config.toml")
        cfg = load_config(src; set = set)
        exec = ExecutionEnv(workers_per_patch = [1], worker_threads = 1)
        t = @elapsed run_simulation(cfg, exec; config_sources = [src], set = set)
        println("segment driver XZ: $(round(t; digits = 1)) s")
        check_two_segment_run(root, 1)
        hdr = Scythe._read_header(joinpath(root, "scythe_config.toml"))
        @test occursin(src, hdr) && occursin("grid.num_cells=[8]", hdr)
        @test occursin("workers_per_patch = 1", hdr)
        @test load_config(joinpath(root, "scythe_config.toml")) == cfg
        check_resume_and_refusal(root, cfg, exec)
    end

    # The nest case costs ~6 min (two child processes each compiling the nested driver on
    # fresh workers for a 10 s segment). Gated behind SCYTHE_SLOW_TESTS=1 like the slow
    # benchmark smoke tests: the single-patch case above covers the driver's segment
    # bookkeeping every run, and the per-patch checkpoint chaining it adds is exercised
    # by the tc_axisym smoke run (simulations/tc_axisym/README.md).
    if get(ENV, "SCYTHE_SLOW_TESTS", "") == "1"
    @testset "2-patch axisymmetric nest, 2 segments" begin
        root = mktempdir()
        set = [tiny; "grid.boundaries=[0.0,40.0e3,80.0e3]"; "grid.num_cells=[6,6]";
               "physics.boundary_layer.scheme=none"; "physics.surface.fluxes=false";
               "initial_conditions.balanced_vortex.r_m=20.0e3";
               "initial_conditions.balanced_vortex.r_0=70.0e3";
               "output.dir=$root"]
        src = joinpath(SIM_DIR, "tc_axisym", "scythe_config.toml")
        cfg = load_config(src; set = set)
        exec = ExecutionEnv(workers_per_patch = [1, 1], worker_threads = 1)
        t = @elapsed run_simulation(cfg, exec; config_sources = [src], set = set)
        println("segment driver nest: $(round(t; digits = 1)) s")
        check_two_segment_run(root, 2)
        @test all(isfile(joinpath(root, "nest$(i)_ics.csv")) for i in 1:2)
        @test isdir(joinpath(root, "seg_02", "nest1")) && isdir(joinpath(root, "seg_02", "nest2"))
        check_resume_and_refusal(root, cfg, exec)
    end
    end # SCYTHE_SLOW_TESTS
end
