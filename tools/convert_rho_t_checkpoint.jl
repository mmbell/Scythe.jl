#!/usr/bin/env julia
#
# Convert a PRE-STAGE-4 restart checkpoint (or initial-condition CSV) of the moist-compressible
# equation set to the Stage-4 slot layout of the rho_t retirement: slot 3, the prognostic total
# density "rho_t", is deleted (rho_t is the derived rho_d + sum_k alpha_k nu_k) and the vapor
# ("rho_v", or "nu_v" under vapor_transform = :bhyp) moves from its appended position (10 on the
# XZ slice, 11 on the cylinders) into slot 3; every later slot moves one in.
#
#   julia --project=<Scythe> tools/convert_rho_t_checkpoint.jl IN.jld2 OUT.jld2
#   julia --project=<Scythe> tools/convert_rho_t_checkpoint.jl IN.jld2 OUT.jld2 --config scythe_config.toml
#   julia --project=<Scythe> tools/convert_rho_t_checkpoint.jl IN_ics.csv OUT_ics.csv
#
# * A checkpoint that carries a vapor slot (every checkpoint written since Stage A): the
#   conversion is a pure permutation of the spectral and physical columns, exact; no config is
#   needed. What it DROPS is the old run's partition gap rho_t'(slot) - (rho_d' + W'): the
#   restarted total (the derived rho_d + sum_k alpha_k nu_k) differs from the old run's
#   prognostic rho_t by exactly that. With --config the tool prints its maximum and column
#   integral; without, it says the gap was not measured. A density-vapor checkpoint ("rho_v")
#   restarting a vapor-control-variable model ("nu_v") is then converted ON LOAD by the model,
#   as before Stage 4.
# * A checkpoint with NO vapor slot (pre-Stage A): the vapor is derived from slot 3 so that the
#   water sum equals the old rho_t - rho_d (nu_v' = (rho_t - rho_d - sum_{k!=v} alpha_k nu_k)/alpha_v
#   - nu_vbar), against the reference of the model the config describes; `--config` is required.
#   That folds the old partition gap and lobe pool into the vapor: a converted restart is not
#   like-for-like (validate from a quiet state).
# * A CSV initial condition: the rho_t column is dropped and the vapor column moved, every value's
#   text unchanged (a CSV with a rho_t column also LOADS as written: the column is not read).
#
# OUT must not exist. The library half is src/checkpoint_conversion.jl
# (`Scythe.convert_rho_t_checkpoint`, `Scythe.convert_rho_t_grid`, `Scythe.convert_rho_t_csv`).

using Scythe

function main(args)
    pos = String[]
    config = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--config"
            i < length(args) || error("--config needs a path")
            config = args[i + 1]
            i += 2
        elseif a in ("-h", "--help")
            println("usage: julia --project=<Scythe> tools/convert_rho_t_checkpoint.jl IN OUT " *
                    "[--config scythe_config.toml]")
            return 0
        else
            push!(pos, a)
            i += 1
        end
    end
    length(pos) == 2 || error("usage: convert_rho_t_checkpoint.jl IN OUT [--config cfg.toml]")
    src, dst = pos
    if endswith(src, ".csv")
        Scythe.convert_rho_t_csv(src, dst)
        println("wrote $(dst) (Stage-4 layout; the rho_t column dropped, the vapor column moved)")
        return 0
    end
    model = nothing
    if config !== nothing
        cfg = load_config(config)
        m = build_model(cfg)
        m isa ModelParameters || error("--config: a nested configuration has one model per " *
                                       "patch; convert each patch's checkpoint with a " *
                                       "single-patch config of that patch. The converted " *
                                       "segment checkpoints are then fed back to the nested " *
                                       "run through initial_conditions.method = \"file\" " *
                                       "(one converted file per patch, in initial_conditions.file.paths).")
        model = m
    end
    Scythe.convert_rho_t_checkpoint(src, dst; model)
    println("wrote $(dst) (Stage-4 layout)")
    return 0
end

main(ARGS)
