using Test
using Scythe
using Springsteel

# Stages 3a/3b of the configuration plan: the src-side helpers the segmented run driver calls —
# checkpoint naming, segment arithmetic, per-patch nest initial conditions, the strict
# whole-step guard in build_nest, and the ExecutionEnv defaults. No workers are spawned.

@testset "Run driver helpers" begin

    @testset "checkpoint_path" begin
        @test checkpoint_path("out", 3600.0) == joinpath("out", "3600.0.jld2")
        @test checkpoint_path("out", 3600) == joinpath("out", "3600.0.jld2")
        @test checkpoint_path("/a/b", 12.3456) == joinpath("/a/b", "12.35.jld2")
        @test checkpoint_path("x", 0.0) == joinpath("x", "0.0.jld2")
        # the name write_restart uses, character for character
        t = 86400.0 * 1.5
        @test basename(checkpoint_path("d", t)) == "$(string(round(t; digits=2))).jld2"
    end

    @testset "seg_length / seg_offset" begin
        # whole number of segments
        @test [Scythe.seg_length(j, 10.0, 2.0) for j in 1:5] == fill(2.0, 5)
        @test [Scythe.seg_offset(k, 10.0, 2.0) for k in 1:5] == [0.0, 2.0, 4.0, 6.0, 8.0]
        # non-integer number of segments: 10 / 4 → 4, 4, 2
        @test [Scythe.seg_length(j, 10.0, 4.0) for j in 1:3] == [4.0, 4.0, 2.0]
        @test [Scythe.seg_offset(k, 10.0, 4.0) for k in 1:3] == [0.0, 4.0, 8.0]
        @test Scythe.seg_offset(3, 10.0, 4.0) + Scythe.seg_length(3, 10.0, 4.0) == 10.0
        # one segment longer than the run
        @test Scythe.seg_length(1, 1.5, 2.0) == 1.5
        @test Scythe.seg_offset(1, 1.5, 2.0) == 0.0
        # integer arguments are accepted and return Float64
        @test Scythe.seg_length(2, 5, 3) === 2.0
        @test Scythe.seg_offset(2, 5, 3) === 3.0
    end

    @testset "whole_steps" begin
        @test Scythe.whole_steps(3600.0, 0.5)
        @test Scythe.whole_steps(86400.0, 0.15)          # 576000 steps despite 0.15 in binary
        @test Scythe.whole_steps(0.3, 0.1)               # 2.9999999999999996 → within 1e-9
        @test !Scythe.whole_steps(3600.0, 0.7)
        @test !Scythe.whole_steps(1.0, 0.3)
        @test Scythe.whole_steps(1.0 + 1e-12, 0.5)
        @test !Scythe.whole_steps(1.0 + 1e-6, 0.5)
    end

    @testset "ExecutionEnv defaults" begin
        e = ExecutionEnv()
        @test e.workers_per_patch == [1]
        @test e.worker_threads == 0
        @test e.blas_threads == 1
        @test e.slurm == false
        e2 = ExecutionEnv(workers_per_patch = [2, 1, 2], slurm = true)
        @test sum(e2.workers_per_patch) == 5
        @test e2.slurm
        @test_throws ArgumentError setup_workers!(ExecutionEnv(workers_per_patch = Int[]))
        @test_throws ArgumentError setup_workers!(ExecutionEnv(workers_per_patch = [1, 0]))
        # worker helpers compile (no spawn): argument guards fire before addprocs, and the
        # extension probe used by the worker loader is well-formed on this build
        @test_throws ArgumentError Scythe.add_workers(0)
        @test_throws ArgumentError Scythe.add_slurm_workers(1; worker_threads = -1)
        @test Base.get_extension(Scythe, :ScytheRRTMGPExt) === nothing ||
              Base.get_extension(Scythe, :ScytheRRTMGPExt) isa Module
        @test hasmethod(Scythe._pin_blas_and_load, Tuple{Vector{Int}, Int})
    end

    @testset "git helpers" begin
        root = normpath(joinpath(@__DIR__, ".."))
        sha, dirty = Scythe.git_info(root)
        @test sha isa String && dirty isa Bool
        @test Scythe.git_untracked_count(root) isa Int
    end

    base_1d(; output_interval = 80.0, restart_interval = 0.0, ics = "") = ModelParameters(
        ts = 0.1,
        integration_time = 80.0,
        output_interval = output_interval,
        restart_interval = restart_interval,
        equation_set = "LinearAdvection1D",
        initial_conditions = ics,
        output_dir = "./output_rundriver_test/",
        grid_params = GridParameters(
            geometry = "R",
            num_cells = 10, iMin = 0.0, iMax = 10.0,   # placeholder, ignored
            BCL = Dict("u" => DirichletBC()),
            BCR = Dict("u" => NaturalBC()),
            vars = Dict("u" => 1)),
        physical_params = Dict(:c_0 => 1.0, :K => 0.0),
    )
    nest_1d(base; ts = [0.1, 0.05, 0.1], ics = String[]) = NestedModelParameters(
        boundaries = [-60.0, -20.0, 20.0, 60.0],
        num_cells = [40, 80, 40],
        ts = ts,
        workers_per_patch = [1, 1, 1],
        base = base,
        initial_conditions = ics)

    @testset "NestedModelParameters initial_conditions" begin
        # default (empty): derived nest{i}_<basename> beside the base path
        models, _ = build_nest(nest_1d(base_1d(ics = "/tmp/ics/run_ics.csv")))
        @test [m.initial_conditions for m in models] ==
              [joinpath("/tmp/ics", "nest$(i)_run_ics.csv") for i in 1:3]
        # default with no base IC: every patch empty
        models, _ = build_nest(nest_1d(base_1d()))
        @test all(isempty(m.initial_conditions) for m in models)
        # keyword omitted entirely == empty vector
        @test NestedModelParameters(boundaries = [0.0, 1.0], num_cells = [1], ts = [1.0],
                                    workers_per_patch = [1],
                                    base = base_1d()).initial_conditions == String[]
        # explicit per-patch paths are used verbatim, overriding the base path
        paths = ["/r/seg_01/nest1/3600.0.jld2", "/r/seg_01/nest2/3600.0.jld2",
                 "/r/seg_01/nest3/3600.0.jld2"]
        models, _ = build_nest(nest_1d(base_1d(ics = "/tmp/ics/run_ics.csv"); ics = paths))
        @test [m.initial_conditions for m in models] == paths
        # wrong length is refused
        @test_throws ArgumentError build_nest(nest_1d(base_1d(); ics = paths[1:2]))
        @test_throws ArgumentError build_nest(nest_1d(base_1d(); ics = [paths; paths[1]]))
    end

    @testset "build_nest strict_intervals" begin
        # ts_actual = [0.1, 0.05, 0.1]: 80 s is whole on every patch
        @test build_nest(nest_1d(base_1d()); strict_intervals = true) isa Tuple
        @test build_nest(nest_1d(base_1d(restart_interval = 40.0));
                         strict_intervals = true) isa Tuple
        # 0.125 s output: whole on 0.05? 2.5 steps — no; on 0.1 — 1.25, no
        @test_throws ArgumentError build_nest(nest_1d(base_1d(output_interval = 0.125));
                                              strict_intervals = true)
        # restart interval off the child's step
        @test_throws ArgumentError build_nest(nest_1d(base_1d(restart_interval = 40.025));
                                              strict_intervals = true)
        # a child whose ts rounds down (0.1 / ceil(0.1/0.03) = 0.025): 80 s still whole
        @test build_nest(nest_1d(base_1d(); ts = [0.1, 0.03, 0.1]);
                         strict_intervals = true)[2].ts_actual[2] ≈ 0.025
        # default keeps the legacy warning, no error
        @test_logs (:warn, r"output_interval") match_mode=:any build_nest(
            nest_1d(base_1d(output_interval = 0.125)))
        @test_logs (:warn, r"restart_interval") match_mode=:any build_nest(
            nest_1d(base_1d(restart_interval = 40.025)))
    end

    # ── Stage 3b: the segment driver's pure pieces ────────────────────────────────────

    owb = joinpath(@__DIR__, "..", "simulations", "ocean_warm_bubble", "config.toml")
    tc = joinpath(@__DIR__, "..", "simulations", "tc_axisym", "config.toml")

    @testset "segment_plan" begin
        plan(N, D) = Scythe.segment_plan(load_config(owb; set = [
            "run.integration_time=$N", "run.segment_length=$D"]))
        p = plan(20.0, 10.0)
        @test length(p) == 2
        @test [s.index for s in p] == [1, 2]
        @test [s.t_offset for s in p] == [0.0, 10.0]
        @test [s.integration_time for s in p] == [10.0, 10.0]
        # non-integer count: 25 / 10 -> 10, 10, 5
        p = plan(25.0, 10.0)
        @test [s.integration_time for s in p] == [10.0, 10.0, 5.0]
        @test [s.t_offset for s in p] == [0.0, 10.0, 20.0]
        # a ratio within round-off of an integer adds no sliver segment
        p = plan(0.3, 0.1)
        @test length(p) == 3
        @test sum(s.integration_time for s in p) ≈ 0.3
        # unsegmented: one segment of the whole run
        p = plan(7200.0, 0.0)
        @test length(p) == 1 && p[1].integration_time == 7200.0 && p[1].t_offset == 0.0
        # segment longer than the run
        p = plan(100.0, 3600.0)
        @test length(p) == 1 && p[1].integration_time == 100.0
        # the TC case: five 1-day segments
        p = Scythe.segment_plan(load_config(tc))
        @test length(p) == 5 && p[5].t_offset == 4 * 86400.0
    end

    @testset "layout helpers" begin
        @test Scythe.segment_dirname(3) == "seg_03"
        @test Scythe.segment_dirname(12) == "seg_12"
        @test Scythe.segment_patch_dirs("/r", 2, 1) == ["/r/seg_02"]
        @test Scythe.segment_patch_dirs("/r", 1, 2) == ["/r/seg_01/nest1", "/r/seg_01/nest2"]
        @test Scythe.initial_ics_paths("/r", 1) == ["/r/ics.csv"]
        @test Scythe.initial_ics_paths("/r", 3) == ["/r/nest$(i)_ics.csv" for i in 1:3]
        @test Scythe.exec_flags(ExecutionEnv(workers_per_patch = [2, 1], worker_threads = 3,
                                             blas_threads = 0, slurm = true)) ==
              ["--workers-per-patch", "2,1", "--worker-threads", "3", "--blas-threads", "0",
               "--slurm"]
    end

    @testset "checkpoint discovery" begin
        mktempdir() do root
            dirs = Scythe.segment_patch_dirs(root, 1, 2)
            foreach(mkpath, dirs)
            for t in (5.0, 10.0), d in dirs     # 10.0 written last: the newest
                touch(checkpoint_path(d, t))
                sleep(0.05)
            end
            # the expected name
            @test Scythe.find_segment_checkpoints(dirs, 10.0) ==
                  [checkpoint_path(d, 10.0) for d in dirs]
            @test Scythe.find_segment_checkpoints(dirs, 5) ==
                  [checkpoint_path(d, 5.0) for d in dirs]
            # expected name missing -> the newest by mtime, with a warning per patch
            paths = @test_logs (:warn, r"not found") (:warn, r"not found") begin
                Scythe.find_segment_checkpoints(dirs, 12.0)
            end
            @test paths == [checkpoint_path(d, 10.0) for d in dirs]
            # the patches' newest checkpoints disagree on t -> error
            sleep(0.05)
            touch(checkpoint_path(dirs[1], 7.0))
            @test_throws ErrorException Scythe.find_segment_checkpoints(dirs, 12.0)
            # no checkpoint at all
            empty = joinpath(root, "seg_09")
            mkpath(empty)
            @test_throws ErrorException Scythe.find_segment_checkpoints([empty], 10.0)
            @test_throws ErrorException Scythe.find_segment_checkpoints(
                [joinpath(root, "missing")], 10.0)
        end
    end

    @testset "segments.csv" begin
        mktempdir() do root
            @test Scythe.read_segments_csv(root) == Dict{Int,Dict{String,String}}()
            @test Scythe.segment_status(root, 1) == ""
            Scythe.update_segment_row!(root, 2, Dict("status" => "running", "t_offset_s" => 10.0,
                                                     "ts_actual" => "0.5;0.25"))
            Scythe.update_segment_row!(root, 1, Dict("status" => "done", "dir" => "/x, y",
                                                     "note" => "a \"quoted\" note"))
            rows = Scythe.read_segments_csv(root)
            @test sort(collect(keys(rows))) == [1, 2]
            @test rows[1]["dir"] == "/x, y"
            @test rows[1]["note"] == "a \"quoted\" note"
            @test rows[2]["ts_actual"] == "0.5;0.25"
            @test rows[2]["t_offset_s"] == "10.0"
            @test rows[2]["finished"] == ""
            # upsert merges: unspecified columns keep their values
            Scythe.update_segment_row!(root, 2, Dict("status" => "done"))
            @test Scythe.segment_status(root, 2) == "done"
            @test Scythe.read_segments_csv(root)[2]["ts_actual"] == "0.5;0.25"
            # the header is exactly the documented column list, rows sorted
            lines = readlines(joinpath(root, "segments.csv"))
            @test split(lines[1], ',') == Scythe.SEGMENTS_CSV_COLS
            @test startswith(lines[2], "1,") && startswith(lines[3], "2,")
            @test_throws ArgumentError Scythe.update_segment_row!(root, 1, Dict("bogus" => 1))
        end
    end

    @testset "fresh-start refusal" begin
        mktempdir() do root
            @test Scythe.check_fresh_root(root) === nothing
            @test Scythe.check_fresh_root(joinpath(root, "not_yet")) === nothing
            touch(joinpath(root, "config.toml")); touch(joinpath(root, "ics.csv"))
            @test Scythe.check_fresh_root(root) === nothing      # inputs are not output
            mkpath(joinpath(root, "seg_01"))
            @test_throws ErrorException Scythe.check_fresh_root(root)
        end
        mktempdir() do root
            mkpath(joinpath(root, "old", "nest2"))
            touch(joinpath(root, "old", "nest2", "3600.0.nc"))
            err = try Scythe.check_fresh_root(root); "" catch e; sprint(showerror, e) end
            @test occursin("mv", err) && occursin("never rm", err) && occursin("3600.0.nc", err)
        end
        # run_simulation refuses before writing anything
        mktempdir() do root
            mkpath(joinpath(root, "seg_01"))
            cfg = load_config(owb; set = ["output.dir=$root", "physics.radiation.scheme=none"])
            @test_throws ErrorException run_simulation(cfg, ExecutionEnv())
            @test !isfile(joinpath(root, "config.toml"))
            # resume with a disallowed --set is refused
            err = try
                run_simulation(cfg, ExecutionEnv(); resume = true, set = ["grid.num_cells=[8]"])
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("only run.integration_time", err)
            # resume with nothing to resume
            @test_throws ErrorException run_simulation(cfg, ExecutionEnv(); resume = true)
        end
    end

    @testset "segment_model whole-step guards" begin
        mktempdir() do root
            base = ["output.dir=$root", "grid.num_cells=[8]",
                    "timestep.ts=0.5", "output.interval=10", "output.restart_interval=10",
                    "run.integration_time=20", "run.segment_length=10"]
            cfg = load_config(owb; set = base)
            seg = merge(Scythe.segment_plan(cfg)[1], (initial_conditions = ["/x/ics.csv"],))
            m, ts = Scythe.segment_model(cfg, seg, ExecutionEnv())
            @test m isa ModelParameters && ts == [0.5]
            @test m.initial_conditions == "/x/ics.csv"
            @test m.output_dir == joinpath(root, "seg_01")
            # a nest's worker groups come from the execution environment
            @test_throws ErrorException Scythe.segment_model(cfg, seg,
                                                             ExecutionEnv(workers_per_patch = [1, 1]))
            bad(s...) = Scythe.segment_model(load_config(owb; set = [base; s...]), seg, ExecutionEnv())
            @test_throws ErrorException bad("timestep.ts=0.3")          # 10 / 0.3
            @test_throws ErrorException bad("output.interval=4")        # 10 / 4 outputs
            @test_throws ErrorException bad("output.restart_interval=0.7")
            # surface fluxes with no boundary-layer closure die at build, not on a worker
            @test_throws ErrorException bad("physics.boundary_layer.scheme=none")
            @test bad("physics.boundary_layer.scheme=none", "physics.surface.fluxes=false")[2] == [0.5]
        end
    end

    @testset "parse_launcher_args" begin
        a = parse_launcher_args(["--config", "/c/config.toml", "fine.toml", "rlr.toml",
                                 "--set", "run.integration_time=20", "--set", "grid.num_cells=[8]",
                                 "--resume", "--workers-per-patch", "2,1,1",
                                 "--worker-threads", "4", "--blas-threads", "0", "--slurm"])
        @test a.config_paths == ["/c/config.toml", "fine.toml", "rlr.toml"]
        @test a.set == ["run.integration_time=20", "grid.num_cells=[8]"]
        @test a.resume
        @test a.exec.workers_per_patch == [2, 1, 1]
        @test a.exec.worker_threads == 4 && a.exec.blas_threads == 0 && a.exec.slurm
        @test a.segment == 0
        d = parse_launcher_args(String[])
        @test d.config_paths == [joinpath(dirname(abspath(PROGRAM_FILE)), "config.toml")]
        @test d.set == String[] && !d.resume && d.segment == 0
        @test d.exec.workers_per_patch == [1] && d.exec.worker_threads == 0 &&
              d.exec.blas_threads == 1 && !d.exec.slurm
        @test parse_launcher_args(["--workers", "4"]).exec.workers_per_patch == [4]
        @test parse_launcher_args(["--config", "x.toml", "--segment", "3"]).segment == 3
        @test_throws ArgumentError parse_launcher_args(["--workers", "2",
                                                        "--workers-per-patch", "1,1"])
        @test_throws ArgumentError parse_launcher_args(["--workers-per-patch", "1,0"])
        @test_throws Scythe.ArgParse.ArgParseError parse_launcher_args(["--bogus"])
        # the child's flags round-trip through the parser
        e = ExecutionEnv(workers_per_patch = [3, 1], worker_threads = 2, blas_threads = 0,
                         slurm = true)
        r = parse_launcher_args(["--config", "c.toml", "--segment", "2",
                                 Scythe.exec_flags(e)...]).exec
        @test (r.workers_per_patch, r.worker_threads, r.blas_threads, r.slurm) ==
              (e.workers_per_patch, e.worker_threads, e.blas_threads, e.slurm)
    end
end
