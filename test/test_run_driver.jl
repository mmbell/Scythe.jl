using Test
using Scythe
using Springsteel

# Stage 3a of the configuration plan: the src-side helpers the segmented run driver calls —
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
end
