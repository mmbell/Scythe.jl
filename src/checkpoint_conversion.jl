# ─────────────────────────────────────────────────────────────────────────────
# Restart and initial-condition conversion across the ρ_t retirement.
#
# Two layout changes have to be bridged by a checkpoint written before them:
#
#   * Stage 1 (the vapor control variable): a checkpoint written under the DENSITY vapor
#     (`"rho_v"`) restarting a model that carries `"nu_v"` (`vapor_transform = :bhyp`) — converted
#     ON LOAD (`vapor_checkpoint_conversion`, `convert_vapor_checkpoint!`), because only the one
#     slot's values change and the model's own reference is at hand there.
#   * Stage 4 (slot 3 deleted): a checkpoint written with slot 3 the prognostic total density
#     `"rho_t"` and the vapor APPENDED after slot 9 (10 on XZ, 11 on the cylinders). Its slot
#     LAYOUT differs from the model's, so it is NOT converted on load: `load_initial_conditions!`
#     refuses it, naming [`RHO_T_CONVERTER`](@ref), and the tool writes a Stage-4 file
#     ([`convert_rho_t_checkpoint`](@ref)):
#       - the common case (every checkpoint since Stage A carries a vapor slot): slot 3 is
#         DROPPED and the vapor slot moved into its position, every later slot one further in —
#         a pure permutation of the spectral and physical columns, exact;
#       - a checkpoint with NO vapor slot (pre-Stage-A): the vapor is DERIVED from slot 3 so
#         that the water sum equals the old ρ_t − ρ_d,
#             α_v ν_v = (ρ_t − ρ_d) − Σ_{k≠v} α_k ν_k     (totals),   ν_v' = ν_v − ν̄_v,
#         which needs the model's reference (`--config`). It preserves mass, buoyancy and T;
#         the old partition gap and lobe pool are folded into the vapor (design §6 Stage 4,
#         §8 item 8: converted restarts are not like-for-like).
#
# A density-vapor Stage-4 file restarting a `nu_v` model is then converted on load exactly as in
# Stage 1, so the two conversions compose and the tool never needs the model for the common case.
# ─────────────────────────────────────────────────────────────────────────────

"The restart converter for pre-Stage-4 checkpoints, relative to the Scythe repository root."
const RHO_T_CONVERTER = "tools/convert_rho_t_checkpoint.jl"

"How to run [`RHO_T_CONVERTER`](@ref), for the loader's refusal message."
rho_t_converter_invocation(path::AbstractString) =
    "julia --project=<Scythe> $(RHO_T_CONVERTER) $(path) <converted.jld2>" *
    " [--config <scythe_config.toml>]"

"""
    pre_stage4_layout(vars) -> Bool

Whether a variable map is the pre-Stage-4 moist-compressible layout: it declares the deleted
prognostic total-density slot `"rho_t"`.
"""
pre_stage4_layout(vars) = haskey(vars, "rho_t")

"""
    stage4_var_names(vars_old) -> (names, vapor)

The variable names of a pre-Stage-4 map in their STAGE-4 order — `"rho_t"` dropped, the vapor
(`"rho_v"` or `"nu_v"`) moved into slot 3, every other name in its old relative order — and the
vapor's name (`nothing` when the old map carries no vapor slot; the caller then inserts one).
"""
function stage4_var_names(vars_old::AbstractDict)
    names = first.(sort(collect(vars_old), by = last))
    vname = haskey(vars_old, "nu_v") ? "nu_v" : haskey(vars_old, "rho_v") ? "rho_v" : nothing
    rest = filter(n -> n != "rho_t" && n != vname, names)
    length(rest) >= 2 || error("stage4_var_names: $(sort(collect(keys(vars_old)))) is not a " *
                               "moist-compressible variable map")
    return vname === nothing ? (rest, nothing) : ([rest[1:2]; vname; rest[3:end]], vname)
end

# Every name-keyed dict of the grid parameters with the "rho_t" entries (and the per-direction
# "rho_t_k"/"rho_t_j" filter keys) removed.
function _drop_rho_t_keys(d::AbstractDict)
    out = copy(d)
    for k in ("rho_t", "rho_t_k", "rho_t_j")
        haskey(out, k) && delete!(out, k)
    end
    return out
end

"""
    stage4_grid_params(params_old) -> SpringsteelGridParameters

The grid parameters of a pre-Stage-4 checkpoint in the Stage-4 layout ([`stage4_var_names`](@ref)):
`vars` renumbered, and every name-keyed dict (the six BC sides, `l_q`, `max_wavenumber`, the
filters, `positivity`) without its `"rho_t"` entries. Every other field is the checkpoint's.
Requires the old map to carry a vapor slot.
"""
function stage4_grid_params(p)
    names, vname = stage4_var_names(p.vars)
    vname === nothing && error("stage4_grid_params: the checkpoint carries no vapor slot; the " *
                               "vapor has to be derived from slot 3 (pass the model)")
    vars = Dict{String,Int64}(n => i for (i, n) in enumerate(names))
    keyed = (:BCL, :BCR, :BCU, :BCD, :BCB, :BCT, :l_q, :max_wavenumber, :fourier_filter,
             :chebyshev_filter, :spline_filter, :positivity)
    kw = Dict{Symbol,Any}(n => getproperty(p, n) for n in propertynames(p))
    kw[:vars] = vars
    for f in keyed
        haskey(kw, f) && (kw[f] = _drop_rho_t_keys(kw[f]))
    end
    return Springsteel.SpringsteelGridParameters(; kw...)
end

"""
    convert_rho_t_grid(old; model = nothing, report = true) -> SpringsteelGrid

A pre-Stage-4 grid (`old`, e.g. `load_grid` of a checkpoint) in the Stage-4 layout.

With a vapor slot in `old` (the common case) it is a permutation: slot 3 is dropped, the vapor
moved into slot 3, and every surviving variable's spectral AND physical columns are copied bit
for bit; `model` is not needed (and not read). Without one, the vapor is DERIVED from slot 3,
`α_v ν_v = (ρ_t − ρ_d) − Σ_{k≠v} α_k ν_k` on the totals (the model's own reference: Springsteel's
`ρ̄_t`, `ρ̄_d`, `ρ̄_c` and the derived `ν̄_v`), its spectral column refit from those gridpoint
values with every other variable's coefficients kept; the new grid is then built from
`model.grid_params` (the vapor's boundary conditions are the model's). `report` prints one line
with what was done.
"""
function convert_rho_t_grid(old; model = nothing, report::Bool = true)
    vars_old = old.params.vars
    pre_stage4_layout(vars_old) ||
        error("convert_rho_t_grid: the grid is already in the Stage-4 layout (no \"rho_t\" slot)")
    names, vname = stage4_var_names(vars_old)
    if vname !== nothing
        new = createGrid(stage4_grid_params(old.params))
        for (n, i) in new.params.vars
            j = vars_old[n]
            new.spectral[:, i] .= view(old.spectral, :, j)
            new.physical[:, i, :] .= view(old.physical, :, j, :)
        end
        report && println("Scythe: converted the pre-Stage-4 checkpoint layout: slot 3 " *
                          "(rho_t) dropped, the vapor \"$(vname)\" moved from slot " *
                          "$(vars_old[vname]) to slot 3 (columns copied exactly).")
        return new
    end
    model === nothing && error("convert_rho_t_grid: the checkpoint carries no vapor slot " *
        "(pre-Stage-A), so the vapor has to be derived from slot 3 against the model's " *
        "reference: pass `model` (the tool's `--config`).")
    return _convert_rho_t_derive_vapor(old, model; report)
end

# The pre-Stage-A path: the vapor derived from slot 3 (see `convert_rho_t_grid`).
function _convert_rho_t_derive_vapor(old, model; report::Bool = true)
    gp = model.grid_params
    vnew = gp.vars
    pre_stage4_layout(vnew) && error("convert_rho_t_grid: the model's own `vars` still carry " *
                                     "\"rho_t\"; build them from `Scythe.mc_var_names(options)`")
    vars_old = old.params.vars
    rv_new = mc_slot(vnew, "rho_v")
    for (n, _) in vnew
        (mc_slot(vnew, "rho_v") == vnew[n] || haskey(vars_old, n)) || error(
            "convert_rho_t_grid: the model's variable \"$(n)\" is not in the checkpoint " *
            "($(sort(collect(keys(vars_old))))) and only the vapor can be derived")
    end
    new = createGrid(gp)
    size(new.spectral, 1) == size(old.spectral, 1) &&
        size(new.physical, 1) == size(old.physical, 1) ||
        error("convert_rho_t_grid: the model's grid ($(size(new.physical, 1)) points) is not " *
              "the checkpoint's ($(size(old.physical, 1)))")
    kDim = gp.kDim
    z = getGridpoints(new)[1:kDim, end]
    ref_state = build_reference_state(model, collect(Float64, z), reference_column(new, gp))
    diag = mc_reference_diagnostics(ref_state, collect(Float64, z); mc_ref_diag_kwargs(model)...)
    rtb = view(ref_rho_t(ref_state), :, 1)
    rdb = view(ref_rho_d(ref_state), :, 1)
    rcb = view(Springsteel.ref_rho_c(ref_state), :, 1)
    ws = water_anchor_set(model.options, model.physical_params, mc_slots(model))
    nvb = diag.nu_vbar
    av = ws.alpha[1]
    # old slot indices of the members other than the vapor, in the anchor's member order
    oldslot(s) = s == 0 ? 0 : vars_old[first(n for (n, i) in vnew if i == s)]
    for (n, i) in vnew
        i == rv_new && continue
        new.spectral[:, i] .= view(old.spectral, :, vars_old[n])
        new.physical[:, i, :] .= view(old.physical, :, vars_old[n], :)
    end
    vals = view(new.physical, :, rv_new, 1)
    irt = vars_old["rho_t"]; ird = vars_old["rho_d"]
    @inbounds for g in axes(old.physical, 1)
        k = mod1(g, kDim)
        W = (old.physical[g, irt, 1] + rtb[k]) - (old.physical[g, ird, 1] + rdb[k])
        for j in 2:6                        # cloud, rain, the three ice masses
            s = ws.slot[j]; a = ws.alpha[j]
            (s == 0 || a == 0.0) && continue
            v = old.physical[g, oldslot(s), 1]
            r = ws.ref[j]
            v = r == 2 ? (ws.ctrans ? v + bhyp(rcb[k], ws.cmu) : v + rcb[k]) : v
            W -= a * v
        end
        vals[g] = (W / av) - nvb[k]
    end
    saved = copy(new.spectral)
    spectralTransform!(new)
    for v in axes(new.spectral, 2)
        v == rv_new && continue
        new.spectral[:, v] .= view(saved, :, v)
    end
    gridTransform!(new)
    for (n, i) in vnew                      # the surviving columns, bit for bit
        i == rv_new && continue
        new.physical[:, i, :] .= view(old.physical, :, vars_old[n], :)
    end
    report && println("Scythe: converted the pre-Stage-A checkpoint (no vapor slot): the " *
                      "vapor derived from slot 3 so that rho_w equals the old rho_t - rho_d; " *
                      "the old partition gap and lobe pool are folded into the vapor.")
    return new
end

"""
    convert_rho_t_checkpoint(src, dst; model = nothing) -> dst

Convert the pre-Stage-4 `.jld2` checkpoint `src` to the Stage-4 layout and write it to `dst`
(Springsteel `save_grid`), through [`convert_rho_t_grid`](@ref). `dst` must not exist (model
output is never overwritten). The library half of [`RHO_T_CONVERTER`](@ref).
"""
function convert_rho_t_checkpoint(src::AbstractString, dst::AbstractString; model = nothing)
    isfile(src) || error("convert_rho_t_checkpoint: $(src) does not exist")
    ispath(dst) && error("convert_rho_t_checkpoint: $(dst) exists; choose a new path (model " *
                         "output is never overwritten)")
    new = convert_rho_t_grid(load_grid(String(src)); model)
    mkpath(dirname(abspath(dst)))
    save_grid(String(dst), new)
    return dst
end

"""
    convert_rho_t_csv(src, dst) -> dst

The initial-condition CSV form of the layout conversion (`write_ics_csv`'s format: coordinate
columns, then one column per variable in slot order): the `"rho_t"` column is dropped and the
vapor column moved into the third variable position, every value's TEXT copied unchanged, so a
CSV written before Stage 4 converts to the bytes the Stage-4 writer produces for the same state.
`dst` must not exist.
"""
function convert_rho_t_csv(src::AbstractString, dst::AbstractString)
    ispath(dst) && error("convert_rho_t_csv: $(dst) exists; choose a new path")
    lines = readlines(src)
    header = split(lines[1], ',')
    "rho_t" in header || error("convert_rho_t_csv: $(src) has no rho_t column")
    ncoord = count(h -> h in ("r", "l", "z"), header)
    vnames = header[(ncoord + 1):end]
    vars_old = Dict(String(n) => i for (i, n) in enumerate(vnames))
    names, vname = stage4_var_names(vars_old)
    vname === nothing && error("convert_rho_t_csv: $(src) carries no vapor column; a CSV " *
                               "initial condition cannot derive one (regenerate it)")
    order = [collect(1:ncoord); [ncoord + vars_old[n] for n in names]]
    mkpath(dirname(abspath(dst)))
    open(dst, "w") do f
        for l in lines
            fields = split(l, ',')
            println(f, join(fields[order], ","))
        end
    end
    return dst
end

"""
    to_pre_stage4_grid(new, model) -> SpringsteelGrid

The inverse layout map, for tests and forensics: a Stage-4 grid with the deleted slot 3
RE-INSERTED as the derived total `ρ_t' = (ρ_d + ρ_w) − ρ̄_t` (Springsteel's fitted reference)
and the vapor appended after slot 9 (10 on XZ, 11 on the cylinders) — a synthetic pre-Stage-4
checkpoint, every surviving column bitwise the Stage-4 one, slot 3's coefficients fitted from
its gridpoint values on `rho_d`'s boundary conditions and its value plane those exact values
(so the derive path of [`convert_rho_t_grid`](@ref) can be checked against the vapor it removes).
"""
function to_pre_stage4_grid(new, model)
    p = new.params
    pre_stage4_layout(p.vars) && error("to_pre_stage4_grid: already pre-Stage-4")
    names = first.(sort(collect(p.vars), by = last))
    vname = names[3]
    rest = names[[1, 2]]; tail = names[4:end]
    cyl = haskey(p.vars, "v")
    nfixed = cyl ? 10 : 9                   # slots 1-9 (+ v) before the old appended vapor
    body = [rest; "rho_t"; tail]            # p, rho_d, rho_t, u, ... (v)
    old_names = [body[1:nfixed]; vname; body[(nfixed + 1):end]]
    vars = Dict{String,Int64}(n => i for (i, n) in enumerate(old_names))
    keyed = (:BCL, :BCR, :BCU, :BCD, :BCB, :BCT, :l_q, :max_wavenumber, :fourier_filter,
             :chebyshev_filter, :spline_filter, :positivity)
    kw = Dict{Symbol,Any}(n => getproperty(p, n) for n in propertynames(p))
    kw[:vars] = vars
    for f in keyed
        haskey(kw, f) || continue
        d = copy(kw[f])
        for sfx in ("", "_k", "_j")
            haskey(d, "rho_d" * sfx) && (d["rho_t" * sfx] = d["rho_d" * sfx])
        end
        kw[f] = d
    end
    old = createGrid(Springsteel.SpringsteelGridParameters(; kw...))
    for (n, i) in p.vars
        old.spectral[:, vars[n]] .= view(new.spectral, :, i)
        old.physical[:, vars[n], :] .= view(new.physical, :, i, :)
    end
    gp = model.grid_params
    kDim = gp.kDim
    z = getGridpoints(new)[1:kDim, end]
    ref_state = build_reference_state(model, collect(Float64, z), reference_column(new, gp))
    diag = mc_reference_diagnostics(ref_state, collect(Float64, z); mc_ref_diag_kwargs(model)...)
    ws = water_anchor_set(model.options, model.physical_params, mc_slots(model))
    rtb = view(ref_rho_t(ref_state), :, 1)
    rcb = view(Springsteel.ref_rho_c(ref_state), :, 1)
    irt = vars["rho_t"]
    ird = p.vars["rho_d"]
    rdb = view(ref_rho_d(ref_state), :, 1)
    phys1 = view(new.physical, :, :, 1)
    rtp = zeros(size(new.physical, 1))
    @inbounds for g in axes(new.physical, 1)
        k = mod1(g, kDim)
        W = anchor_W_value(phys1, g, k, ws, diag.nu_vbar, rcb)
        rtp[g] = ((new.physical[g, ird, 1] + rdb[k]) + W) - rtb[k]
    end
    old.physical[:, irt, 1] .= rtp
    saved = copy(old.spectral)
    spectralTransform!(old)
    for v in axes(old.spectral, 2)
        v == irt && continue
        old.spectral[:, v] .= view(saved, :, v)
    end
    gridTransform!(old)
    for (n, i) in p.vars                    # the surviving columns, bit for bit
        old.physical[:, vars[n], :] .= view(new.physical, :, i, :)
    end
    old.physical[:, irt, 1] .= rtp
    return old
end

# ── The Stage 1 vapor conversion (on load), on the Stage-4 layout ──────────────

"""
    vapor_slot_from_density!(slot, rho_vbar, nu_vbar, kDim, mu) -> Int

The pointwise map of [`convert_vapor_checkpoint!`](@ref), in place on one slot column (z
fastest, `kDim` levels): `slot ← vapor_control(rho_vbar + slot) − nu_vbar`, i.e.
`bhyp(ρ) − ν̄_v` for `ρ ≥ 0` and `−bhyp(−ρ) − ν̄_v` below. Returns the number of negative
densities met.
"""
function vapor_slot_from_density!(slot::AbstractVector{Float64}, rho_vbar::Vector{Float64},
                                  nu_vbar::Vector{Float64}, kDim::Int, mu::Float64)
    nneg = 0
    @inbounds for g in eachindex(slot)
        k = mod1(g, kDim)
        rho = slot[g] + rho_vbar[k]
        rho < 0.0 && (nneg += 1)
        slot[g] = vapor_control(rho, mu) - nu_vbar[k]
    end
    return nneg
end

"""
    vapor_checkpoint_conversion(vars_file, vars_model, path) -> Int

The vapor slot index to convert when a checkpoint's variable map differs from the model's ONLY
by the vapor's name — `"rho_v"` (density, written under `vapor_transform = :none`) in the file
where the model carries `"nu_v"` (`:bhyp`) — and `0` when the maps agree or differ otherwise
(the caller's compatibility check then decides). Refused here, by name: a `"nu_v"` checkpoint
read as a density vapor (the clipped `ahyp` has no inverse where the recovered vapor is 0), and
a PRE-STAGE-4 checkpoint (a `"rho_t"` slot, the vapor appended) into a Stage-4 model, whose
refusal names [`RHO_T_CONVERTER`](@ref) and its invocation.
"""
function vapor_checkpoint_conversion(vars_file, vars_model, path)
    if pre_stage4_layout(vars_file) && !pre_stage4_layout(vars_model)
        error("restart archive $(path) is in the PRE-STAGE-4 layout: slot 3 is the prognostic " *
              "total density \"rho_t\" and the vapor is appended after slot 9, while this model " *
              "has no rho_t slot (Stage 4 of the rho_t retirement: rho_t is the derived " *
              "rho_d + sum_k alpha_k nu_k and slot 3 is the vapor). Convert it first with " *
              "$(RHO_T_CONVERTER):\n    " * rho_t_converter_invocation(path))
    end
    vars_file == vars_model && return 0
    if haskey(vars_file, "nu_v") && haskey(vars_model, "rho_v")
        error("restart archive $(path) carries the vapor CONTROL variable \"nu_v\" " *
              "(written under options[:vapor_transform] = :bhyp), but this model runs the " *
              "density vapor \"rho_v\" (vapor_transform = :none). There is no conversion in " *
              "that direction (ahyp is clipped at 0); run with vapor_transform = :bhyp.")
    end
    (haskey(vars_file, "rho_v") && haskey(vars_model, "nu_v") &&
     vars_file["rho_v"] == vars_model["nu_v"]) || return 0
    rest_f = Dict(k => v for (k, v) in vars_file if k != "rho_v")
    rest_m = Dict(k => v for (k, v) in vars_model if k != "nu_v")
    rest_f == rest_m || return 0
    return vars_model["nu_v"]
end

"""
    convert_vapor_checkpoint!(patch, model, iv)

Convert the density-vapor slot `iv` of a just-loaded checkpoint (`patch.physical` regenerated,
`patch.spectral` the archive's) to the vapor control variable of `model`
(`options[:vapor_transform] = :bhyp`):

    ν_v' = B(ρ̄_v + ρ_v') − ν̄_v ,   B(ρ) = bhyp(ρ; μ_v) for ρ ≥ 0,  −bhyp(−ρ; μ_v) for ρ < 0,

with `ρ̄_v`, `ν̄_v` the model's OWN derived reference (`build_reference_state` +
`mc_reference_diagnostics`, the objects every tile builds). A negative checkpoint vapor maps to
the odd continuation (the inverse of `ahyp_signed`): `2ν ≈ ρ` there, so the water the old run's
linear ρ_t carried stays in the slot as a ν lobe (recovered 0, counted by `MC_VNEG`) rather than
being dropped. Only the vapor slot is refit (its spectral column from the converted values);
every other slot keeps the archive's coefficients bit for bit. Converted restarts are not
like-for-like (design §8 item 8): validate from 24 h only.
"""
function convert_vapor_checkpoint!(patch::AbstractGrid, model::ModelParameters, iv::Int)
    gp = model.grid_params
    kDim = gp.kDim
    z = getGridpoints(patch)[1:kDim, end]
    ref_state = build_reference_state(model, collect(Float64, z), reference_column(patch, gp))
    diag = mc_reference_diagnostics(ref_state, collect(Float64, z); mc_ref_diag_kwargs(model)...)
    mu = vapor_mu(model.physical_params)
    saved = copy(patch.spectral)
    nneg = vapor_slot_from_density!(view(patch.physical, :, iv, 1), diag.rho_vbar,
                                    diag.nu_vbar, kDim, mu)
    spectralTransform!(patch)
    for v in axes(patch.spectral, 2)
        v == iv && continue
        patch.spectral[:, v] .= view(saved, :, v)
    end
    gridTransform!(patch)
    println("Scythe: converted the density-vapor checkpoint $(model.initial_conditions) to " *
            "the vapor control variable nu_v (mu_v = $(mu) kg/m^3): $(nneg) of " *
            "$(size(patch.physical, 1)) points carried negative vapor and enter as nu_v < 0.")
    return patch
end

"""
    _check_csv_layout(path, vars)

An initial-condition CSV written before Stage 4 carries a `"rho_t"` column. Springsteel's reader
reads every model variable BY NAME, so with the vapor column present such a file loads into the
Stage-4 layout as written — the `rho_t` column is simply not read (rho_t is the derived
`ρ_d + Σ_k α_k ν_k`), and a note says so. Without a vapor column (a pre-Stage-A file) there is
nothing to read the vapor from: refused, naming [`RHO_T_CONVERTER`](@ref).
"""
function _check_csv_layout(path, vars)
    isfile(path) || return nothing
    header = split(strip(first(eachline(path))), ',')
    "rho_t" in header || return nothing
    pre_stage4_layout(vars) && return nothing
    if !("rho_v" in header || "nu_v" in header)
        error("initial-condition CSV $(path) is in the PRE-STAGE-4 layout with no vapor column " *
              "(slot 3 the prognostic rho_t): there is no vapor to read. Regenerate it, or " *
              "convert a checkpoint of the same state with $(RHO_T_CONVERTER).")
    end
    println("Scythe: the initial-condition CSV $(path) carries a pre-Stage-4 \"rho_t\" column; " *
            "it is not read (rho_t is the derived rho_d + sum_k alpha_k nu_k).")
    return nothing
end

"""
    _check_csv_vapor_name(path, vars)

A CSV initial condition written under the density vapor (a `rho_v` column) cannot initialize a
vapor-control-variable model (`nu_v`): Springsteel's reader would only say "Grid is missing
variable nu_v". Refuse with the actual reason and the way to read it. There is no conversion
for a CSV (its fields are the raw mish values, written by an initializer for one convention);
run with `vapor_transform = :none`, or regenerate the IC under `:bhyp`.
"""
function _check_csv_vapor_name(path, vars)
    (haskey(vars, "nu_v") && isfile(path)) || return nothing
    header = split(strip(first(eachline(path))), ',')
    ("rho_v" in header && !("nu_v" in header)) || return nothing
    error("initial-condition CSV $(path) carries the DENSITY vapor column \"rho_v\", but " *
          "this model carries the vapor control variable \"nu_v\" " *
          "(options[:vapor_transform] = :bhyp, physics.microphysics.vapor_transform). A CSV " *
          "is not converted: run with vapor_transform = :none (\"none\" in the config) to " *
          "read it as written, or regenerate the initial condition under :bhyp.")
end
