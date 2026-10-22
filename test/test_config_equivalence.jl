# test/test_config_equivalence.jl -- `build_model`/`initialize_case!` (src/config_build.jl,
# src/config_init.jl) against the Stage-0 golden capture of the three legacy drivers
# (test/reference_data/legacy_models/, tools/capture_legacy_models.jl).
#
#   1. EQUIVALENCE. For every captured arm, the case config (simulations/<case>/scythe_config.toml)
#      plus a legacy-pin overlay (test/reference_data/legacy_models/<arm>.overlay.toml, which
#      states the legacy physics) must fingerprint key by key like the golden. The only
#      tolerated differences are
#        - keys present only in the new build whose value is the moist-compressible library
#          default (`IMPLICIT_DEFAULTS`; writing a library default changes nothing the model
#          reads), and
#        - the path basenames, compared through `legacy_basename` (the new layout names the
#          files reference.ref / ics.csv / nest{i}_ics.csv under <root>, output in seg_NN).
#   2. DELTAS. The PRODUCTION config (no overlay) minus the legacy default arm must be
#      exactly `EXPECTED_DELTAS` -- every physics change the consolidation makes is listed.
#   3. IC BYTES. `initialize_case!` with the legacy-pin overlay writes initial conditions
#      and a reference file whose SHA-256 match the golden `.stage4.hashes.toml` (the
#      `.hashes.toml` capture converted to the Stage-4 slot layout, see below).
#
# STAGE 4 of the rho_t retirement deleted slot 3 (the prognostic "rho_t") and moved the vapor
# into it. The golden fingerprints are the Stage-0 capture, in the old layout: `ceq_legacy`
# maps them to the Stage-4 layout by that rule alone (`ceq_stage4_layout`: every rho_t entry
# dropped, the vapor's slot 3, every slot after the vapor's old one moved one in) so the rest
# of the comparison is unchanged.
#   4. `validate` refusals, the segment override, the advisory keyword.
#
# Standalone:
#   julia --project=. -e 'using Test, Scythe; include("test/test_config_equivalence.jl")'

using Test
using TOML
using SHA
using Scythe
using Springsteel

include(joinpath(@__DIR__, "config_fingerprint.jl"))

const CEQ_ROOT = normpath(joinpath(@__DIR__, ".."))
const CEQ_LM = joinpath(@__DIR__, "reference_data", "legacy_models")
const CEQ_SIM = joinpath(CEQ_ROOT, "simulations")
const CEQ_TC = joinpath(CEQ_SIM, "tc_axisym", "scythe_config.toml")
const CEQ_RCE = joinpath(CEQ_SIM, "ocean_2D_rce", "scythe_config.toml")
const CEQ_OWB = joinpath(CEQ_SIM, "ocean_warm_bubble", "scythe_config.toml")
ceq_ov(name) = joinpath(CEQ_LM, "$(name).overlay.toml")

# arm => (config files in load order, segment, needs RRTMGP)
const CEQ_ARMS = [
    ("tc_default",       [CEQ_TC, ceq_ov("tc_default")], nothing, false),
    ("tc_mynn_rad_stab", [CEQ_TC, ceq_ov("tc_default"), ceq_ov("tc_mynn_rad_stab")], nothing, true),
    ("tc_r1t1x",         [CEQ_TC, ceq_ov("tc_default"), ceq_ov("tc_r1t1x")], nothing, false),
    ("tc_rlr",           [CEQ_TC, joinpath(CEQ_SIM, "tc_axisym", "examples", "rlr.toml"), ceq_ov("tc_default")],
                         nothing, false),
    ("rce_default_seg1", [CEQ_RCE, ceq_ov("rce_default")], nothing, true),
    # Segment 3 of 20 d in 2 d segments: t_offset 4 d, restarting from seg_02's checkpoint.
    ("rce_default_seg3", [CEQ_RCE, ceq_ov("rce_default")],
                         (index = 3, t_offset = 345600.0, integration_time = 172800.0,
                          initial_conditions = ["/scratch/seg_02/172800.0.jld2"]), true),
    ("rce_ice0_louis",   [CEQ_RCE, ceq_ov("rce_default"), ceq_ov("rce_ice0_louis")], nothing, false),
    ("owb_default",      [CEQ_OWB, ceq_ov("owb_default")], nothing, false),
    ("owb_mynn_ice_rad", [CEQ_OWB, ceq_ov("owb_default"), ceq_ov("owb_mynn_ice_rad")], nothing, true),
]

# Keys the new build writes that a legacy driver left to the library `get` default. Keyed by
# the fingerprint path below the model (`options.<k>` / `physical_params.<k>`), value = the
# fingerprint value (options are `repr` strings). Each value IS the library default at the
# cited read site, so dropping the key changes nothing the model reads.
const IMPLICIT_DEFAULTS = Dict{String,Any}(
    "options.consistent_qss_reference" => "false",   # src/semiimplicit.jl:379
    "options.hydrostatic_reference"    => "false",   # src/semiimplicit.jl:370
    "options.state_dependent_si"       => "false",   # src/moist_compressible.jl:7014
    "options.exact_si"                 => "false",   # src/moist_compressible.jl:7002
    "options.wall_bc_tau"              => "300.0",   # src/semiimplicit.jl:1301
    "options.ice_microphysics"         => ":none",   # src/moist_compressible.jl:2408
    "options.rain_moments"             => "1",       # src/moist_compressible.jl:2256
    "options.condensation"             => "true",    # src/moist_compressible.jl:7424
    "options.condensate_transform"     => ":none",   # src/moist_compressible.jl:2177
    "options.rain_transform"           => ":none",   # src/moist_compressible.jl rain_transform_mode (same form as :2177)
    "options.mynn_interval"            => "20.0",    # src/mynn_state.jl:840
    "options.mynn_edmf"                => "0",       # src/mynn_state.jl:826
    "options.mynn_fidelity"            => ":scythe", # src/mynn_state.jl:891
    "options.mynn_init"                => ":taper",  # src/mynn_state.jl:829
    "options.mynn_output"              => "false",   # src/mynn_state.jl:849
    "options.radiation_output"         => "false",   # src/radiation_state.jl:595
    # NOT the library default (that is :eos since 2026-10-14) but the legacy drivers' physics:
    # the key did not exist and every consumer read the total-energy retrieval, which is
    # exactly :energy (bitwise; test/test_eos_temperature.jl). The base overlays pin it.
    "options.temperature"              => ":energy", # src/moist_compressible.jl temperature_mode
    # The vapor control variable (Stage 1 of the rho_t retirement): the library default is
    # :none, the density vapor every legacy driver ran; the base overlays pin it.
    "options.vapor_transform"          => ":none",   # src/moist_compressible.jl vapor_transform_mode
    # The water-slot SI slaving (Stage 2 of the rho_t retirement): the library default is
    # :none, the explicit water acoustics every legacy driver ran; the base overlays pin it.
    "options.si_water_slots"           => ":none",   # src/si_water_slaving.jl si_water_slots_mode
    # (The derived water anchor, Stage 3, was pinned here as options.water_anchor; Stage 4
    # retired the option -- the derived anchor is the only behaviour -- so the key is gone.)
    "physical_params.Khdiff_water"     => 0.0,       # src/moist_compressible.jl:6773
    "physical_params.Pr_t"             => 1.0,       # src/moist_compressible.jl:6770
    "physical_params.Sc_t"             => 1.0)       # src/moist_compressible.jl:6774

"Flatten a nested fingerprint Dict to `\"a.b.c\" => leaf`."
function ceq_flat(d::AbstractDict, pre::String = "")
    out = Dict{String,Any}()
    for (k, v) in d
        p = isempty(pre) ? String(k) : string(pre, ".", k)
        v isa AbstractDict ? merge!(out, ceq_flat(v, p)) : (out[p] = v)
    end
    return out
end

"The model-relative part of a flattened key (drops a nest's `patches.patch_<i>.` prefix)."
ceq_local(k::String) = replace(k, r"^patches\.patch_\d+\." => "")

"""
Legacy basename → the new layout's basename: `<case>_exact.ref` → `reference.ref`,
`[nest{i}_]<case>_ics.csv` → `[nest{i}_]ics.csv`, and the OWB output directory (a path
with a trailing slash, basename "") → `seg_01` (the new layout is always segmented).
Restart checkpoints (`172800.0.jld2`) and nest output dirs (`nest{i}`) are unchanged.
"""
function legacy_basename(k::String, v)
    endswith(k, "ref_state_file_basename") && return replace(v, r"^(tc|rce|owb)_exact\.ref$" => "reference.ref")
    endswith(k, "initial_conditions_basename") && return replace(v, r"(tc|rce|owb)_ics\.csv$" => "ics.csv")
    endswith(k, "output_dir_basename") && return v == "" ? "seg_01" : v
    return v
end

"""
The Stage-4 slot layout of a legacy fingerprint: per model (a nest: per patch) every entry
keyed by the variable `rho_t` is dropped (slot 3, the prognostic total density, is deleted),
the vapor (`rho_v`, or `nu_v`) takes slot 3, and every slot after the vapor's old one moves one
in; the slots between are unchanged. Nothing else in the fingerprint depends on the layout.
"""
function ceq_stage4_layout(d::Dict)
    out = Dict{String,Any}()
    pre = unique(m.captures[1] for m in (match(r"^(.*grid_params\.)vars\.", k) for k in keys(d))
                 if m !== nothing)
    vold = Dict{String,Int}()
    for g in pre
        @assert d[g * "vars.rho_t"] == 3
        vn = haskey(d, g * "vars.rho_v") ? "rho_v" : "nu_v"
        vold[g] = d[g * "vars." * vn]
    end
    for (k, v) in d
        endswith(k, ".rho_t") && occursin("grid_params.", k) && continue
        m = match(r"^(.*grid_params\.)vars\.(.+)$", k)
        if m === nothing
            out[k] = v
        else
            sv = vold[m.captures[1]]
            out[k] = v == sv ? 3 : (v > sv ? v - 1 : v)
        end
    end
    return out
end

ceq_legacy(arm) = ceq_stage4_layout(Dict(k => legacy_basename(k, v)
                                         for (k, v) in ceq_flat(TOML.parsefile(joinpath(CEQ_LM, "$(arm).toml")))))

function ceq_fingerprint(m)
    fp = m isa NestedModelParameters ? nest_fingerprint(m) : model_fingerprint(m)
    return ceq_flat(fp)
end

"New-only keys whose value is the library default, removed."
function ceq_drop_implicit(new::Dict, old::Dict)
    out = copy(new)
    for (k, v) in new
        haskey(old, k) && continue
        l = ceq_local(k)
        haskey(IMPLICIT_DEFAULTS, l) && isequal(IMPLICIT_DEFAULTS[l], v) && delete!(out, k)
    end
    return out
end

"`key => (old, new)` for every key that differs (`missing` = absent)."
function ceq_delta(old::Dict, new::Dict)
    d = Dict{String,Any}()
    for k in union(keys(old), keys(new))
        a = get(old, k, missing); b = get(new, k, missing)
        isequal(a, b) || (d[k] = (a, b))
    end
    return d
end

ceq_load(files; dir, extra = String[]) = load_config(files...; set = ["output.dir=$(dir)"; extra])

# ── EXPECTED_DELTAS: production config − legacy default arm ─────────────────────────
#
# Per case: the explicit changes below the model level (`options.*`, `physical_params.*`,
# scalar fields, `grid_params.positivity.*`), explicit BC changes of variables both builds
# have, and the variable-list change: `renamed` (legacy name => new name, same slot, all
# four BCs carried unchanged), `added` (name => slot) whose BCs follow the stated template
# (`lateral` = the variable whose BCL/BCR entries it takes, `vertical` = the one whose
# BCB/BCT entries it takes, both in the NEW build). For a nest the same table applies to
# every patch.

const BC_D2 = "BoundaryConditions(nothing, nothing, 0.0, nothing, false)"      # SecondDerivativeBC
const BC_NATURAL = "BoundaryConditions(nothing, nothing, nothing, nothing, false)"
const ICE_SLOTS = ["nu_i1", "nu_ni1", "nu_ai1", "nu_ci1", "nu_i2", "nu_ni2", "nu_ai2", "nu_ci2",
                   "nu_i3", "nu_ni3", "nu_ai3", "nu_ci3"]
ceq_added(first_slot) = merge(Dict("nu_nr" => first_slot),
                              Dict(v => first_slot + i for (i, v) in enumerate(ICE_SLOTS)),
                              Dict("rho_e" => first_slot + 13))
# (`nu_v`: the production build's vapor slot, renamed under the vapor control variable)
ceq_templates(rain) = merge(Dict(v => (lateral = "nu_v", vertical = rain) for v in ["nu_nr"; ICE_SLOTS]),
                            Dict("rho_e" => (lateral = "nu_v", vertical = "nu_v")))

# The ground and the lid of the scalar family: d2 in every legacy build, natural now
# (`boundary_conditions.vertical_scalars` / `vertical_scalars_lid`).
ceq_lid(vars) = Dict{String,Any}("$b.$v" => (BC_D2, BC_NATURAL) for v in vars for b in ("BCB", "BCT"))
const LID_SCALARS = ["p", "rho_d", "E_t", "Q_ss", "rho_v", "u"]   # (rho_t: deleted, Stage 4)
# Under the production vapor control variable (`physics.microphysics.vapor_transform = "bhyp"`)
# the vapor slot is RENAMED rho_v -> nu_v (same slot): its lid/ground change rides on the new
# name, the old name's BCs leave with the rename.
const LID_SCALARS_NU = filter(!=("rho_v"), LID_SCALARS)
const VAPOR_RENAMED_BC = Dict{String,Any}("BCB.nu_v" => (missing, BC_NATURAL),
                                          "BCT.nu_v" => (missing, BC_NATURAL))

const RAD_OPTIONS_ADDED = Dict(
    "options.radiation" => (missing, ":rrtmgp"), "options.radiation_method" => (missing, ":allsky"),
    "options.solar" => (missing, ":diurnal"), "options.radiation_forcing" => (missing, ":full"),
    "options.radiation_interval" => (missing, "300.0"),
    "options.radiation_z_max" => (missing, "17000.0"),
    "physical_params.latitude" => (missing, 20.0), "physical_params.start_doy" => (missing, 240.0),
    "physical_params.start_hour" => (missing, 0.0), "physical_params.sfc_albedo" => (missing, 0.06),
    "physical_params.sfc_emissivity" => (missing, 0.98))

const EXPECTED_DELTAS = Dict(
    # TC: ice (two-moment rain + number/ice transforms), MYNN, surface stability, diurnal
    # RRTMGP, rain/ice free at the lid, wall_bc_tau at the library value (inert under d2),
    # and the run length = one 1-day segment (the legacy capture used make_base(1800)).
    "tc" => (config = CEQ_TC, arm = "tc_default",
        changes = merge(RAD_OPTIONS_ADDED, Dict(
            "integration_time" => (1800.0, 86400.0),
            "options.ice_microphysics" => (missing, ":ishmael"),
            "options.rain_moments" => (missing, "2"),
            "options.rain_number_transform" => (missing, ":bhyp"),
            "options.ice_transform" => (missing, ":bhyp"),
            "options.louis_bl" => ("true", missing),
            "options.mynn" => (missing, "true"),
            "options.sfc_stability" => (missing, "true"),
            "options.wall_bc_tau" => ("Inf", "300.0"),
            "options.temperature" => (missing, ":eos"),
            "options.vapor_transform" => (missing, ":bhyp"),
            "options.si_water_slots" => (missing, ":mass"))),
        bc_changed = merge(ceq_lid([LID_SCALARS_NU; "v"; "nu_c"]), VAPOR_RENAMED_BC,
                           Dict("BCT.nu_r" => (BC_D2, BC_NATURAL))),
        renamed = Dict("rho_v" => "nu_v"),
        added = ceq_added(11), templates = ceq_templates("nu_r")),
    # RCE: both reference-state fixes and the state-dependent SI (physics otherwise the
    # legacy default already: MYNN :scythe, ice, diurnal RRTMGP, stability).
    "rce" => (config = CEQ_RCE, arm = "rce_default_seg1",
        changes = Dict(
            "options.consistent_qss_reference" => (missing, "true"),
            "options.hydrostatic_reference" => (missing, "true"),
            "options.state_dependent_si" => (missing, "true"),
            "options.temperature" => (missing, ":eos"),
            "options.vapor_transform" => (missing, ":bhyp"),
            "options.si_water_slots" => (missing, ":mass")),
        bc_changed = merge(ceq_lid([LID_SCALARS_NU; "nu_c"; "rho_e"]), VAPOR_RENAMED_BC),
        renamed = Dict("rho_v" => "nu_v"),
        added = Dict{String,Int}(), templates = Dict{String,Any}()),
    # OWB: Louis -> MYNN (+ rho_e), ice (two-moment rain, bhyp on every water slot: rho_r/rho_c
    # become nu_r/nu_c and the rain positivity bound goes), NetCDF instead of CSV, a restart
    # cadence, both reference fixes, state-dependent SI, surface stability.
    "owb" => (config = CEQ_OWB, arm = "owb_default",
        changes = Dict(
            "restart_interval" => (0.0, 21600.0),
            "options.output_formats" => ("[:csv]", "[:netcdf]"),
            "options.consistent_qss_reference" => (missing, "true"),
            "options.hydrostatic_reference" => (missing, "true"),
            "options.state_dependent_si" => (missing, "true"),
            "options.ice_microphysics" => (missing, ":ishmael"),
            "options.rain_moments" => (missing, "2"),
            "options.condensate_transform" => (missing, ":bhyp"),
            "options.rain_transform" => (missing, ":bhyp"),
            "options.rain_number_transform" => (missing, ":bhyp"),
            "options.ice_transform" => (missing, ":bhyp"),
            "options.louis_bl" => ("true", missing),
            "options.mynn" => (missing, "true"),
            "options.sfc_stability" => (missing, "true"),
            "grid_params.positivity.rho_r.i" => (0.0, missing),
            "grid_params.positivity.rho_r.k" => (0.0, missing),
            "options.temperature" => (missing, ":eos"),
            "options.vapor_transform" => (missing, ":bhyp"),
            "options.si_water_slots" => (missing, ":mass")),
        # nu_c is the renamed rho_c: its lid goes natural with the rest of the family.
        bc_changed = merge(ceq_lid(LID_SCALARS_NU), VAPOR_RENAMED_BC,
                           Dict{String,Any}("BCB.nu_c" => (missing, BC_NATURAL), "BCT.nu_c" => (missing, BC_NATURAL))),
        renamed = Dict("rho_r" => "nu_r", "rho_c" => "nu_c", "rho_v" => "nu_v"),
        added = ceq_added(10), templates = ceq_templates("nu_r")))

"The expected `key => (old, new)` delta for one model (prefix = nest patch prefix or \"\")."
function ceq_expected(spec, old::Dict, new::Dict, prefix::String)
    exp = Dict{String,Any}()
    for (k, v) in spec.changes
        exp[prefix * k] = v
    end
    g = prefix * "grid_params."
    bcs = ("BCL", "BCR", "BCB", "BCT")
    for (o, n) in spec.renamed
        exp[g * "vars.$o"] = (old[g * "vars.$o"], missing)
        exp[g * "vars.$n"] = (missing, old[g * "vars.$o"])
        for b in bcs
            exp[g * "$b.$o"] = (old[g * "$b.$o"], missing)
            exp[g * "$b.$n"] = (missing, old[g * "$b.$o"])
        end
    end
    for (v, slot) in spec.added
        exp[g * "vars.$v"] = (missing, slot)
        t = spec.templates[v]
        for b in bcs
            src = b in ("BCL", "BCR") ? t.lateral : t.vertical
            exp[g * "$b.$v"] = (missing, new[g * "$b.$src"])
        end
    end
    # Last: an explicit BC change also overrides what a rename would have carried.
    for (k, v) in spec.bc_changed
        exp[g * k] = v
    end
    return exp
end

cfg_errmsg(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

@testset "config build + init" begin

@testset "equivalence: $arm" for (arm, files, seg, rad) in CEQ_ARMS
    mktempdir() do dir
        cfg = ceq_load(files; dir = dir)
        m = build_model(cfg; segment = seg)
        old = ceq_legacy(arm)
        new = ceq_drop_implicit(ceq_fingerprint(m), old)
        d = ceq_delta(old, new)
        isempty(d) || foreach(k -> println("  $arm $k : $(repr(d[k][1])) => $(repr(d[k][2]))"),
                              sort(collect(keys(d))))
        @test isempty(d)
        # No new-only key may hide a non-default value behind IMPLICIT_DEFAULTS' name.
        for (k, v) in ceq_fingerprint(m)
            haskey(old, k) || @test isequal(IMPLICIT_DEFAULTS[ceq_local(k)], v)
        end
    end
end

@testset "production deltas: $case" for (case, spec) in EXPECTED_DELTAS
    mktempdir() do dir
        m = build_model(ceq_load([spec.config]; dir = dir))
        old = ceq_legacy(spec.arm)
        new = ceq_drop_implicit(ceq_fingerprint(m), old)
        d = ceq_delta(old, new)
        prefixes = m isa NestedModelParameters ?
            ["patches.patch_$i." for i in 1:length(m.num_cells)] : [""]
        exp = Dict{String,Any}()
        for p in prefixes
            merge!(exp, ceq_expected(spec, old, new, p))
        end
        for k in sort(collect(setdiff(keys(d), keys(exp))))
            println("  $case UNEXPECTED $k : $(repr(d[k][1])) => $(repr(d[k][2]))")
        end
        for k in sort(collect(setdiff(keys(exp), keys(d))))
            println("  $case MISSING $k : $(repr(exp[k]))")
        end
        @test keys(d) == keys(exp)
        @test all(isequal(d[k], exp[k]) for k in intersect(keys(d), keys(exp)))
    end
end

@testset "IC bytes: $arm" for (arm, overlays) in [
        ("owb_default", [CEQ_OWB, ceq_ov("owb_default")]),
        ("rce_default_seg1", [CEQ_RCE, ceq_ov("rce_default")]),
        ("tc_default", [CEQ_TC, ceq_ov("tc_default")]),
        ("tc_mynn_rad_stab", [CEQ_TC, ceq_ov("tc_default"), ceq_ov("tc_mynn_rad_stab")])]
    # (the Stage-4 golden: the .hashes.toml ICs in the Stage-4 layout; the reference unchanged)
    golden = TOML.parsefile(joinpath(CEQ_LM, "$(arm).stage4.hashes.toml"))
    g0 = TOML.parsefile(joinpath(CEQ_LM, "$(arm).hashes.toml"))
    @test keys(golden) == keys(g0)
    @test all(golden[b] == g0[b] for b in keys(g0) if endswith(b, ".ref"))
    mktempdir() do dir
        cfg = ceq_load(overlays; dir = dir)
        m = build_model(cfg)
        initialize_case!(cfg, m)
        models = m isa NestedModelParameters ? first(Scythe.build_nest(m)) : [m]
        written = [[mm.initial_conditions for mm in models]; models[1].ref_state_file]
        @test length(written) == length(golden)
        for (b, h) in golden
            k = endswith(b, ".ref") ? "ref_state_file_basename" : "initial_conditions_basename"
            path = joinpath(dir, legacy_basename(k, b))
            @test path in written
            @test isfile(path) && bytes2hex(open(sha256, path)) == h
        end
    end
end

@testset "segment override" begin
    mktempdir() do dir
        cfg = ceq_load([CEQ_RCE]; dir = dir)
        m1 = build_model(cfg)
        @test m1.output_dir == joinpath(dir, "seg_01")
        @test m1.initial_conditions == joinpath(dir, "ics.csv")
        @test m1.ref_state_file == joinpath(dir, "reference.ref")
        @test m1.integration_time == 172800.0
        ck = joinpath(dir, "seg_02", "172800.0.jld2")
        m3 = build_model(cfg; segment = (index = 3, t_offset = 345600.0,
                                         integration_time = 86400.0, initial_conditions = [ck]))
        @test m3.output_dir == joinpath(dir, "seg_03")
        @test m3.initial_conditions == ck && m3.integration_time == 86400.0
        @test m3.physical_params[:start_hour] == 96.0
        # radiation.start_hour is the clock at t = 0 of the RUN; the segment adds its offset
        cfg6 = load_config(CEQ_RCE; set = ["output.dir=$(dir)", "physics.radiation.start_hour=6.0"])
        @test build_model(cfg6; segment = (index = 2, t_offset = 7200.0, integration_time = 7200.0,
                                           initial_conditions = [ck])).physical_params[:start_hour] == 8.0
        # nests: one IC path per patch, verbatim
        tcfg = ceq_load([CEQ_TC, ceq_ov("tc_default")]; dir = dir)
        paths = [joinpath(dir, "seg_01", "nest$i", "3600.0.jld2") for i in 1:3]
        nest = build_model(tcfg; segment = (index = 2, t_offset = 3600.0, integration_time = 3600.0,
                                            initial_conditions = paths))
        @test nest.initial_conditions == paths
        @test nest.base.output_dir == joinpath(dir, "seg_02")
        @test [mm.initial_conditions for mm in first(Scythe.build_nest(nest))] == paths
        @test occursin("2 path(s) for 3 patch", cfg_errmsg(() -> build_model(tcfg;
            segment = (index = 2, t_offset = 0.0, integration_time = 1.0, initial_conditions = paths[1:2]))))
    end
end

@testset "validate refusals" begin
    mktempdir() do dir
        s(xs...) = load_config(CEQ_OWB; set = ["output.dir=$(dir)", xs...])
        @test occursin("physics.boundary_layer.scheme",
                       cfg_errmsg(() -> build_model(s("physics.boundary_layer.scheme=louis"))))
        # Water mixing with ISHMAEL is no longer refused: the leg mixes every ice moment.
        @test build_model(s("physics.horizontal_mixing.water=true")) isa ModelParameters
        @test occursin("numerics.exact_si",
                       cfg_errmsg(() -> build_model(s("numerics.exact_si=true"))))
        # Smagorinsky now exists on the slice too; what is refused is a scheme with no length.
        @test build_model(s("physics.horizontal_mixing.scheme=smagorinsky")) isa ModelParameters
        @test occursin("no length scale",
                       cfg_errmsg(() -> build_model(s("physics.horizontal_mixing.scheme=smagorinsky",
                                                      "physics.horizontal_mixing.Ls=0",
                                                      "physics.horizontal_mixing.C_s=0"))))
        @test occursin("output.restart_interval",
                       cfg_errmsg(() -> build_model(s("run.segment_length=3600.0",
                                                      "output.restart_interval=0.0"))))
        @test occursin("mynn_interval",
                       cfg_errmsg(() -> build_model(s("physics.boundary_layer.mynn_interval=0.25"))))
        @test occursin("grid.boundaries",
                       cfg_errmsg(() -> build_model(s("grid.boundaries=[0.0, 1.0e5, 3.0e5]"))))
        @test occursin("ts_per_patch",
                       cfg_errmsg(() -> build_model(s("timestep.ts_per_patch=[0.5, 0.5]"))))
        @test occursin("water_budget_trace",
                       cfg_errmsg(() -> build_model(s("diagnostics.water_budget_trace=10"))))
        @test occursin("rain_moments",
                       cfg_errmsg(() -> build_model(s("physics.microphysics.rain_moments=1"))))
        @test occursin("physics.radiation.latitude",
                       cfg_errmsg(() -> build_model(s("physics.radiation.scheme=rrtmgp",
                                                      "physics.radiation.latitude=95.0"))))
        tcfg(xs...) = load_config(CEQ_TC; set = ["output.dir=$(dir)", xs...])
        @test occursin("boundary_conditions.lateral",
                       cfg_errmsg(() -> build_model(tcfg("boundary_conditions.lateral=periodic"))))
        # the production configs themselves validate and build
        @test build_model(tcfg()) isa NestedModelParameters
        @test build_model(s()) isa ModelParameters
    end
end

@testset "boundary_conditions.override changes exactly one entry" begin
    mktempdir() do dir
        base = build_model(ceq_load([CEQ_OWB]; dir = dir))
        ovr = build_model(ceq_load([CEQ_OWB]; dir = dir,
                                   extra = ["boundary_conditions.override.nu_r.top=dirichlet"]))
        old = ceq_fingerprint(base); new = ceq_fingerprint(ovr)
        d = ceq_delta(old, new)
        @test collect(keys(d)) == ["grid_params.BCT.nu_r"]
        @test d["grid_params.BCT.nu_r"] == (BC_NATURAL, repr(DirichletBC()))
        @test ovr.grid_params.BCT["nu_r"] == DirichletBC()
        @test all(ovr.grid_params.BCT[v] == base.grid_params.BCT[v]
                  for v in keys(base.grid_params.vars) if v != "nu_r")
        # On XZ the two lateral dicts are one object: an `inner` override must not leak to `outer`.
        lr = build_model(ceq_load([CEQ_OWB]; dir = dir,
                                  extra = ["boundary_conditions.override.nu_v.inner=natural"]))
        # (the vapor slot is `nu_v` under the production vapor control variable)
        @test lr.grid_params.BCL["nu_v"] == NaturalBC() && lr.grid_params.BCR["nu_v"] == NeumannBC()
        @test sort(collect(keys(ceq_delta(old, ceq_fingerprint(lr))))) == ["grid_params.BCL.nu_v"]
        # The table reports the override.
        cfgr = ceq_load([CEQ_OWB]; dir = dir, extra = ["boundary_conditions.override.nu_r.top=dirichlet"])
        r = only(filter(x -> x.variable == "nu_r", boundary_condition_table(ovr, cfgr)))
        @test r.top == "dirichlet" && occursin("override: top", r.note)
    end
end

@testset "boundary_condition_table" begin
    mktempdir() do dir
        cfg = ceq_load([CEQ_TC]; dir = dir)
        nest = build_model(cfg)
        t = boundary_condition_table(nest, cfg)
        vars = Scythe.mc_var_names(nest.base.options; cyl = true)
        @test length(vars) == 24          # (25 before Stage 4 deleted slot 3)
        @test length(t) == 3 * length(vars)
        @test [r.patch for r in t] == repeat(1:3; inner = length(vars))
        @test [r.variable for r in t if r.patch == 2] == vars
        models, _ = Scythe.build_nest(nest)
        # Every entry is the BC the built grid actually carries.
        for r in t
            gp = models[r.patch].grid_params
            @test r.inner == Scythe.bc_name(gp.BCL[r.variable])
            @test r.outer == Scythe.bc_name(gp.BCR[r.variable])
            @test r.bottom == Scythe.bc_name(gp.BCB[r.variable])
            @test r.top == Scythe.bc_name(gp.BCT[r.variable])
        end
        # build_nest: the TC's cells coarsen outward (10|20|40 km), so the OUTER patch of each
        # junction is the parent. The finer patch's outer side is R3X (child); the coarser
        # patch's inner side is its free collar termination (natural).
        for r in t
            p = r.patch
            @test (r.outer == "r3x") == (p < 3)
            @test occursin("outer = nest interface: R3X (child)", r.note) == (p < 3)
            @test occursin("inner = collar: natural (parent)", r.note) == (p > 1)
            p > 1 && @test r.inner == "natural"
        end
        # The domain's own walls: axis u, v Dirichlet, else Neumann; outer wall u Dirichlet only.
        byv(p) = Dict(r.variable => r for r in t if r.patch == p)
        ax, wall = byv(1), byv(3)
        @test all(ax[v].inner == (v in ("u", "v") ? "dirichlet" : "neumann") for v in vars)
        @test all(wall[v].outer == (v == "u" ? "dirichlet" : "neumann") for v in vars)
        # Sedimenting: rain, rain number and the twelve ice slots; natural at both ends.
        sed = ["nu_r"; "nu_nr"; ICE_SLOTS]
        for r in t
            s = r.variable in sed
            @test occursin("sedimenting: flux fit F=0 at lid", r.note) == s
            # The scalar family is natural at ground and lid; w stays Dirichlet.
            @test r.bottom == (r.variable == "w" ? "dirichlet" : "natural")
            @test r.top == r.bottom
        end
        # The file: deterministic, one [[patch]] per patch, the table's rows in order.
        f1 = write_boundary_conditions(joinpath(dir, "a.toml"), t; header = "h")
        f2 = write_boundary_conditions(joinpath(dir, "b.toml"), boundary_condition_table(nest, cfg); header = "h")
        @test read(f1, String) == read(f2, String)
        x = TOML.parsefile(f1)
        @test [p["index"] for p in x["patch"]] == [1, 2, 3]
        @test [v["name"] for v in x["patch"][3]["variable"]] == vars
        @test x["patch"][1]["variable"][1]["outer"] == "r3x"
        # The compact print: one line per variable plus the header and interface legend.
        out = sprint(print_boundary_conditions, t)
        @test count(==('\n'), out) == length(vars) + 1 + 1 + 3
        @test occursin("patch 2: inner = collar: natural (parent), outer = nest interface: R3X (child)", out)
    end
end

@testset "timestep advisory: convective_ceiling keyword" begin
    mktempdir() do dir
        m = build_model(ceq_load([CEQ_RCE]; dir = dir))
        names(r) = [l.name for l in r.limits]
        on = redirect_stdout(devnull) do
            Scythe.warn_timestep_stability(m.grid_params, m.ts; equation_set = m.equation_set,
                                           options = m.options)
        end
        off = redirect_stdout(devnull) do
            Scythe.warn_timestep_stability(m.grid_params, m.ts; equation_set = m.equation_set,
                                           options = m.options, convective_ceiling = false)
        end
        @test any(n -> occursin("convective SI ceiling", n), names(on))
        @test !any(n -> occursin("convective SI ceiling", n), names(off))
        @test length(names(off)) == length(names(on)) - 1
    end
end

@testset "initial_conditions.method = file reads checkpoints in place" begin
    # A .jld2 checkpoint must keep its extension: load_initial_conditions! dispatches on it,
    # and copying it under an ics.csv name once made the CSV reader parse binary (the TC
    # debug-restart recipe). build_model and the driver's segment-1 paths point at the
    # user's files; initialize_case! writes only the reference.
    mktempdir() do dir
        ck = [joinpath(dir, "nest$(i)_21600.0.jld2") for i in 1:3]
        foreach(p -> write(p, "x"), ck)
        cfg = ceq_load([CEQ_TC]; dir = dir,
                       extra = ["initial_conditions.method=file",
                                "initial_conditions.file.paths=[" *
                                join(("\"$p\"" for p in ck), ",") * "]"])
        nest = build_model(cfg)
        @test nest isa NestedModelParameters
        @test nest.initial_conditions == ck
        models, _ = Scythe.build_nest(nest)
        @test [m.initial_conditions for m in models] == ck
        @test Scythe.initial_ics_paths(cfg, dir, 3) == ck
        # A CSV-named single-patch file is used verbatim too.
        single = joinpath(dir, "state.csv"); write(single, "x")
        cfg1 = ceq_load([CEQ_OWB]; dir = dir,
                        extra = ["initial_conditions.method=file",
                                 "initial_conditions.file.paths=[\"$single\"]"])
        @test build_model(cfg1).initial_conditions == single
        # Wrong count is refused.
        bad = ceq_load([CEQ_TC]; dir = dir,
                       extra = ["initial_conditions.method=file",
                                "initial_conditions.file.paths=[\"$(ck[1])\"]"])
        @test_throws ErrorException build_model(bad)
    end
end

end # config build + init
