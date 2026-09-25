# test/config_fingerprint.jl -- golden-capture fingerprinting (Stage 0 of the config-
# consolidation plan). Turns a `ModelParameters`/`NestedModelParameters` built by one of the
# three legacy drivers (tc/tc_params.jl+tc_init.jl, benchmarks/ocean_rce.jl,
# benchmarks/ocean_warm_bubble.jl) into a plain, TOML-serializable, deterministic Dict, so the
# eventual `build_model(cfg)` replacement can be checked against it key by key (plan §7
# "Equivalence"). Every value is a String, Number, Bool or a nested Dict of those -- no Symbol,
# no custom struct -- and every Dict is built so that key order never matters (equality, not
# textual identity, is what later tests rely on; `TOML.print(...; sorted = true)` gives the
# committed file a deterministic on-disk order).
#
# `using Scythe, Springsteel` must already be in effect before this file is included (it does
# not `using` anything itself, matching the convention of the driver files it fingerprints).

"""
    grid_fingerprint(gp) -> Dict{String,Any}

Fingerprint one `Springsteel.SpringsteelGridParameters` (a.k.a. `GridParameters`): geometry,
axis ranges/cell counts/derived dimensions, the variable-name/slot map, the four boundary-
condition dicts (each `name => repr(bc)` -- a `BoundaryConditions` struct like `NeumannBC()`
and a legacy Dict-valued constant like `Springsteel.CubicBSpline.R1T1X` both stringify fine
under `repr`), the spline filter length `l_q`, the per-variable/per-axis `positivity` floor,
`max_wavenumber`, and the quadrature/mubar basis parameters.
"""
function grid_fingerprint(gp)::Dict{String,Any}
    d = Dict{String,Any}()
    d["geometry"] = String(gp.geometry)
    d["iMin"] = Float64(gp.iMin)
    d["iMax"] = Float64(gp.iMax)
    d["num_cells_i"] = Int(gp.num_cells_i)
    d["kMin"] = Float64(gp.kMin)
    d["kMax"] = Float64(gp.kMax)
    d["num_cells_k"] = Int(gp.num_cells_k)
    d["kDim"] = Int(gp.kDim)
    d["iDim"] = Int(gp.iDim)
    d["mubar"] = Int(gp.mubar)
    d["quadrature"] = String(gp.quadrature)
    d["vars"] = Dict{String,Any}(String(k) => Int(v) for (k, v) in gp.vars)
    d["l_q"] = Dict{String,Any}(String(k) => Float64(v) for (k, v) in gp.l_q)
    d["max_wavenumber"] = Dict{String,Any}(String(k) => Int(v) for (k, v) in gp.max_wavenumber)
    d["positivity"] = Dict{String,Any}(
        String(name) => Dict{String,Any}(String(axis) => Float64(val) for (axis, val) in axes)
        for (name, axes) in gp.positivity)
    for (label, bc) in (("BCL", gp.BCL), ("BCR", gp.BCR), ("BCB", gp.BCB), ("BCT", gp.BCT))
        d[label] = Dict{String,Any}(String(k) => repr(v) for (k, v) in bc)
    end
    return d
end

"""
    model_fingerprint(m::ModelParameters) -> Dict{String,Any}

Fingerprint one `ModelParameters`: the scalar run-control fields, the BASENAME (not the full,
run-specific path) of `initial_conditions`/`output_dir`/`ref_state_file`, `options` as a Dict
of `String(key) => repr(value)` (options are not homogeneously typed -- Bool, Float64, Int,
Symbol, `Vector{Symbol}` all appear -- so `repr` is the one representation that always
round-trips to a comparable string), `physical_params` as a Dict of `String(key) => Float64`,
and the grid fingerprint (`grid_fingerprint`).
"""
function model_fingerprint(m::ModelParameters)::Dict{String,Any}
    d = Dict{String,Any}()
    d["equation_set"] = String(m.equation_set)
    d["ts"] = Float64(m.ts)
    d["integration_time"] = Float64(m.integration_time)
    d["output_interval"] = Float64(m.output_interval)
    d["restart_interval"] = Float64(m.restart_interval)
    d["initial_conditions_basename"] = basename(m.initial_conditions)
    d["output_dir_basename"] = basename(m.output_dir)
    d["ref_state_file_basename"] = basename(m.ref_state_file)
    d["options"] = Dict{String,Any}(String(k) => repr(v) for (k, v) in m.options)
    d["physical_params"] = Dict{String,Any}(String(k) => Float64(v) for (k, v) in m.physical_params)
    d["grid_params"] = grid_fingerprint(m.grid_params)
    return d
end

"""
    nest_fingerprint(nest::NestedModelParameters) -> Dict{String,Any}

Fingerprint an N-patch nest: `Scythe.build_nest(nest)` is called (this is the one place a
`NestedModelParameters` becomes concrete per-patch `ModelParameters` and a `NestTopology`),
and the result records the requested `boundaries`/`num_cells`/`workers_per_patch`, the
DERIVED `topo.ts_actual`/`topo.n_sub` (post integer-subdivision rounding), and each patch's
own `model_fingerprint`, keyed `"patch_<i>"` (1-based, left to right).
"""
function nest_fingerprint(nest::NestedModelParameters)::Dict{String,Any}
    models, topo = Scythe.build_nest(nest)
    d = Dict{String,Any}()
    d["boundaries"] = Float64.(nest.boundaries)
    d["num_cells"] = Int.(nest.num_cells)
    d["ts_actual"] = Float64.(topo.ts_actual)
    d["n_sub"] = Int.(topo.n_sub)
    d["workers_per_patch"] = Int.(nest.workers_per_patch)
    d["patches"] = Dict{String,Any}("patch_$(i)" => model_fingerprint(m)
                                    for (i, m) in enumerate(models))
    return d
end
