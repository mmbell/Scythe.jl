# test/test_legacy_fingerprints.jl -- Stage 0 golden-capture regression (config-
# consolidation plan §7 "Equivalence"). For the three CHEAP arms (owb_default,
# rce_default_seg1, and tc_default's nest fingerprint only -- no `init!`, no ICs, just the
# ModelParameters/NestedModelParameters the legacy driver builds), rebuild the legacy model
# in-process (OWB, RCE) or via a scratch subprocess (TC -- see below) and assert the
# fingerprint equals the committed `test/reference_data/legacy_models/<arm>.toml`. A
# mismatch here means either a driver file changed behavior since the golden capture
# (tools/capture_legacy_models.jl) was taken, or `test/config_fingerprint.jl` itself
# changed -- both are meant to be caught immediately, well before the config-consolidation
# work (stage 2) tries to reproduce these same values from `build_model(cfg)`.
#
# TC is done via a subprocess (not in-process) for two reasons: tc/tc_params.jl's
# `OUTPUT_DIR` const defaults to `tc/output/tc_axisym` -- a REAL output tree -- unless
# `SCYTHE_TC_OUTDIR` is set before the file is included, and `make_base` unconditionally
# `mkpath`s it; running in the test's own process would leave no clean way to redirect that
# without touching real output. A subprocess with `SCYTHE_TC_OUTDIR` pointed at a scratch
# directory keeps the whole thing (including the harmless `mkpath`) off the repo's own
# output trees, mirroring exactly how `tools/capture_legacy_models.jl` captured it.

using Test
using Scythe
using Springsteel
using TOML

const LEGACY_DIR = joinpath(@__DIR__, "reference_data", "legacy_models")
const SCYTHE_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "config_fingerprint.jl"))

"""Load a committed golden fingerprint, TOML round-tripped to plain Dicts/Numbers/Strings."""
load_golden(arm::AbstractString) = TOML.parsefile(joinpath(LEGACY_DIR, "$(arm).toml"))

@testset "Legacy driver golden fingerprints (Stage 0)" begin

    @testset "owb_default" begin
        include(joinpath(SCYTHE_ROOT, "benchmarks", "ocean_warm_bubble.jl"))
        opts = parse_benchmark_args(["--mode", "quick", "--stage", "mc", "--grid", "rirk"])
        model = owb_model(opts)
        @test model_fingerprint(model) == load_golden("owb_default")
    end

    @testset "rce_default_seg1" begin
        include(joinpath(SCYTHE_ROOT, "benchmarks", "ocean_rce.jl"))
        rce = parse_rce_args(["--mode", "quick", "--stage", "mc", "--grid", "rirk",
                              "--days", "20", "--segment-days", "2"])
        opts = parse_benchmark_args(rce.rest)
        root = mktempdir(; prefix = "scythe_test_rce_")
        model = rce_model(rce, opts, 1; root = root)
        @test model_fingerprint(model) == load_golden("rce_default_seg1")
    end

    @testset "tc_default (nest fingerprint only, subprocess)" begin
        tmp = mktempdir(; prefix = "scythe_test_tc_")
        outfile = joinpath(tmp, "fingerprint.toml")
        scratch_outdir = joinpath(tmp, "out")
        script = """
        using Scythe, Springsteel, TOML
        include("$(joinpath(SCYTHE_ROOT, "test", "config_fingerprint.jl"))")
        include("$(joinpath(SCYTHE_ROOT, "tc", "tc_params.jl"))")
        include("$(joinpath(SCYTHE_ROOT, "tc", "tc_init.jl"))")
        base = make_base(1800.0)
        nest = make_nest(base)
        open("$(outfile)", "w") do io
            TOML.print(io, nest_fingerprint(nest); sorted = true)
        end
        """
        scriptfile = joinpath(tmp, "run.jl")
        write(scriptfile, script)
        child_env = merge(Dict{String,String}(string(k) => string(v) for (k, v) in ENV),
                          Dict("SCYTHE_TC_OUTDIR" => scratch_outdir))
        cmd = setenv(`$(Base.julia_cmd()) --project=$(SCYTHE_ROOT) $(scriptfile)`, child_env)
        out = IOBuffer()
        ok = try
            run(pipeline(cmd; stdout = out, stderr = out))
            true
        catch
            false
        end
        ok || println(String(take!(out)))
        @test ok
        @test isfile(outfile)
        if ok && isfile(outfile)
            @test TOML.parsefile(outfile) == load_golden("tc_default")
        end
    end

end
