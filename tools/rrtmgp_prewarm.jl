#!/usr/bin/env julia
#
# Login-node prewarm for RRTMGP: triggers the lookup-table (lazy) artifact
# download while network access is available, and prints where the artifacts
# landed so a cluster preflight check can assert they are already local.
# Read-only with respect to Scythe: no Pkg.Registry.update(), no src/ changes.
#
# Run once per machine/user before a compute-node job that uses radiation:
#
#   julia --project=/Users/mmbell/Development/Scythe.jl tools/rrtmgp_prewarm.jl

using RRTMGP, NCDatasets, ClimaComms
ClimaComms.@import_required_backends

println("RRTMGP prewarm: pkgversion(RRTMGP) = ", pkgversion(RRTMGP))

const CTX = ClimaComms.context(ClimaComms.CPUMultiThreaded())
const FT = Float64

grid_params = RRTMGP.RRTMGPGridParams(
    FT; context = CTX, domain_nlay = 60, ncol = 1, isothermal_boundary_layer = true,
)

println("Building clear-sky lookup bundle (triggers rrtmgp-data artifact download if absent)...")
lookups_cs = RRTMGP.lookup_tables(grid_params, RRTMGP.ClearSkyRadiation(false))
println("  clear-sky lookup bundle built: nbnd_lw=", lookups_cs.nbnd_lw,
        " nbnd_sw=", lookups_cs.nbnd_sw, " ngas_lw=", lookups_cs.ngas_lw)

println("Building all-sky (cloud) lookup bundle...")
lookups_as = RRTMGP.lookup_tables(grid_params, RRTMGP.AllSkyRadiation(false, false))
println("  all-sky lookup bundle built: nbnd_lw=", lookups_as.nbnd_lw,
        " nbnd_sw=", lookups_as.nbnd_sw)

println()
println("Artifact file paths (RRTMGP.ArtifactPaths.get_lookup_filename):")
for optics_type in (:gas, :cloud), λ in (:lw, :sw)
    path = RRTMGP.ArtifactPaths.get_lookup_filename(optics_type, λ)
    println("  ", rpad(string(optics_type, "/", λ), 10), " -> ", path,
            "  (exists: ", isfile(path), ")")
end

# Gray smoke: exercises the non-artifact path too, as a minimal end-to-end
# sanity check that RRTMGP itself is usable (no NetCDF needed).
out = RRTMGP.solve_gray(FT; nlay = 60, ncol = 1)
println()
println("Gray smoke solve_gray OK: OLR = ", round(Array(out.lw_flux_up)[end, 1]; digits = 2), " W/m^2")

println()
println("RRTMGP_PREWARM_OK")
