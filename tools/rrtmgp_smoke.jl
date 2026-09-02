#!/usr/bin/env julia
#
# S0 API-verification smoke test for RRTMGP.jl v1.0, run against the local
# install to confirm every name/signature the radiation design (D1-D9 in
# we-have-now-completed-hidden-pie.md) assumes. Read-only: touches nothing
# under src/, test/, benchmarks/, tc/. Run with:
#
#   julia -t 6 --project=/Users/mmbell/Development/Scythe.jl tools/rrtmgp_smoke.jl
#
# Every section is clearly labelled so the transcript can be quoted directly
# in the S0 report.

function section(title)
    println()
    println("="^78)
    println(title)
    println("="^78)
end

# ---------------------------------------------------------------------------
section("(a) Imports and versions")
# ---------------------------------------------------------------------------

import ClimaComms
ClimaComms.@import_required_backends
using RRTMGP, NCDatasets

println("pkgversion(RRTMGP)      = ", pkgversion(RRTMGP))
println("pkgversion(ClimaComms)  = ", pkgversion(ClimaComms))
println("pkgversion(NCDatasets)  = ", pkgversion(NCDatasets))
println("Threads.nthreads()      = ", Threads.nthreads())
println("VERSION (Julia)         = ", VERSION)

const CTX = ClimaComms.context(ClimaComms.CPUMultiThreaded())
println("context                 = ", CTX)
println("device                  = ", ClimaComms.device(CTX))

# ---------------------------------------------------------------------------
section("(h) check_values toggle")
# ---------------------------------------------------------------------------
if isdefined(RRTMGP, :check_values)
    RRTMGP.check_values[] = true
    println("RRTMGP.check_values[] set to true for the remainder of this script")
else
    println("RRTMGP.check_values does not exist in this version")
end

# ---------------------------------------------------------------------------
section("(b) Gray smoke: RRTMGP.solve_gray")
# ---------------------------------------------------------------------------

println("methods(RRTMGP.solve_gray):")
for m in methods(RRTMGP.solve_gray)
    println("  ", m)
end

out_gray = RRTMGP.solve_gray(Float64; nlay = 60, ncol = 1)
println()
println("solve_gray(Float64; nlay=60, ncol=1) succeeded")
println("  net_flux[1,1]      (surface) = ", Array(out_gray.net_flux)[1, 1], " W/m^2")
println("  net_flux[end,1]    (TOA)     = ", Array(out_gray.net_flux)[end, 1], " W/m^2")
println("  lw_flux_up[end,1]  (OLR)     = ", Array(out_gray.lw_flux_up)[end, 1], " W/m^2")
println("  heating_rate[1,1]  (K/day)   = ", 86400 * Array(out_gray.heating_rate)[1, 1])
println("  heating_rate[end,1](K/day)   = ", 86400 * Array(out_gray.heating_rate)[end, 1])

# ---------------------------------------------------------------------------
section("(c) Public names, field layouts, method lists, getter inventory")
# ---------------------------------------------------------------------------

if isdefined(RRTMGP, :PUBLIC_NAMES)
    println("RRTMGP.PUBLIC_NAMES (", length(RRTMGP.PUBLIC_NAMES), " names):")
    for n in RRTMGP.PUBLIC_NAMES
        println("  ", n)
    end
else
    println("RRTMGP.PUBLIC_NAMES absent; falling back to names(RRTMGP)")
    for n in names(RRTMGP)
        println("  ", n)
    end
end

println()
println("fieldnames(RRTMGP.BCs.SwBCs)  = ", fieldnames(RRTMGP.BCs.SwBCs))
println("fieldnames(RRTMGP.BCs.LwBCs)  = ", fieldnames(RRTMGP.BCs.LwBCs))
println("fieldnames(RRTMGP.RRTMGPSolver) = ", fieldnames(RRTMGP.RRTMGPSolver))
println("fieldnames(RRTMGP.LookupBundle) = ", fieldnames(RRTMGP.LookupBundle))

println()
println("methods(RRTMGP.RRTMGPGridParams):")
for m in methods(RRTMGP.RRTMGPGridParams)
    println("  ", m)
end
println()
println("methods(RRTMGP.RRTMGPSolver):")
for m in methods(RRTMGP.RRTMGPSolver)
    println("  ", m)
end
println()
println("methods(RRTMGP.standard_atmosphere):")
for m in methods(RRTMGP.standard_atmosphere)
    println("  ", m)
end
println()
println("methods(RRTMGP.lookup_tables):")
for m in methods(RRTMGP.lookup_tables)
    println("  ", m)
end

println()
println("Getter functions in RRTMGP whose names match flux/temperature/pressure/",
        "mixing_ratio/cloud/zenith/albedo/emissivity/heating/col_dry/relative_humidity:")
kw = ["flux", "temperature", "pressure", "mixing_ratio", "cloud", "zenith",
      "albedo", "emissivity", "heating", "col_dry", "relative_humidity"]
all_names = names(RRTMGP; all = true)
matches = Symbol[]
for n in all_names
    s = String(n)
    startswith(s, "#") && continue
    if any(k -> occursin(k, lowercase(s)), kw)
        try
            v = getfield(RRTMGP, n)
            if v isa Function
                push!(matches, n)
            end
        catch
        end
    end
end
for n in sort(unique(matches); by = String)
    println("  ", n)
end

println()
println("Searching names(RRTMGP; all=true) for col_dry / compute_col_gas (settability probe):")
for n in all_names
    s = String(n)
    if occursin("col_dry", lowercase(s)) || occursin("compute_col_gas", lowercase(s))
        println("  ", n, "  (exported: ", Base.isexported(RRTMGP, n), ")")
    end
end
println("  RRTMGP.AtmosphericStates.getview_col_dry exists: ",
        isdefined(RRTMGP.AtmosphericStates, :getview_col_dry))
println("  RRTMGP.Optics.compute_col_gas! exists:           ",
        isdefined(RRTMGP.Optics, :compute_col_gas!))

# ---------------------------------------------------------------------------
section("(d) Standard atmosphere structure")
# ---------------------------------------------------------------------------

profile = RRTMGP.standard_atmosphere(Float64; kind = :tropical, nlay = 60)
println("typeof(profile) = ", typeof(profile))
println("fieldnames      = ", fieldnames(typeof(profile)))
println("size(p_lay)  = ", size(profile.p_lay), "   size(p_lev)  = ", size(profile.p_lev))
println("size(t_lay)  = ", size(profile.t_lay), "   size(t_lev)  = ", size(profile.t_lev))
println("size(z_lev)  = ", size(profile.z_lev))
println("size(t_sfc)  = ", size(profile.t_sfc), "   size(lat) = ", size(profile.lat))
println("size(vmr_h2o) = ", size(profile.vmr_h2o), "  size(vmr_o3) = ", size(profile.vmr_o3))
println("well_mixed_vmr = ", profile.well_mixed_vmr)
println()
println("z_lev[1]   = ", profile.z_lev[1, 1], " m   (level 1 -- expect surface, z=0)")
println("z_lev[end] = ", profile.z_lev[end, 1], " m  (level end -- expect top)")
println("p_lev[1]   = ", profile.p_lev[1, 1], " Pa  (expect ~surface pressure)")
println("p_lev[end] = ", profile.p_lev[end, 1], " Pa  (expect low pressure, top)")
println("t_lay[1]   = ", profile.t_lay[1, 1], " K   (near-surface layer T)")
println("t_lay[end] = ", profile.t_lay[end, 1], " K   (top layer T)")
println("t_sfc[1]   = ", profile.t_sfc[1], " K")
println("=> vertical ordering: index 1 = surface, index end = top ",
        "(", profile.z_lev[1,1] < profile.z_lev[end,1] ? "CONFIRMED ascending" : "DESCENDING, unexpected", ")")

# ---------------------------------------------------------------------------
section("(e) Layer-2 clear-sky tropical run (hand-built RRTMGPGridParams/BCs/AtmosphericState)")
# ---------------------------------------------------------------------------

# Helper: pad a (m, ncol) host array to n rows by repeating the top row --
# mirrors RRTMGP's own `_pad_top` (src/api/standalone.jl) used for the
# isothermal-boundary-layer padding. Kept local so this script has no src/
# dependency.
function pad_top(x::AbstractMatrix{FT}, n::Int) where {FT}
    m, ncol = size(x)
    m == n && return Array(x)
    y = Array{FT}(undef, n, ncol)
    @views y[1:m, :] .= x
    for k in (m + 1):n
        @views y[k, :] .= x[m, :]
    end
    return y
end

# Build an AtmosphericState (+ optional CloudState) directly from a
# standard_atmosphere profile, following the same construction RRTMGP.solve()
# uses internally (src/api/standalone.jl) plus the CloudState pattern from
# RRTMGP's own test helper (test/read_cloudy_sky.jl setup_cloudy_sky_as).
function build_state(::Type{FT}, grid_params, lookups, profile; with_cloud::Bool) where {FT}
    (domain_nlay, ncol) = size(profile.p_lay)
    nlay = grid_params.nlay   # includes the isothermal boundary layer if requested
    nlev = nlay + 1

    idx_gases = lookups.idx_gases_lw
    vmr_wm = zeros(FT, lookups.ngas_lw)
    for (gas, val) in profile.well_mixed_vmr
        vmr_wm[idx_gases[gas]] = FT(val)
    end
    vmr = RRTMGP.VolumeMixingRatios.VmrGM(
        pad_top(profile.vmr_h2o, nlay),
        pad_top(profile.vmr_o3, nlay),
        Array{FT}(vmr_wm),
    )

    # NOTE (S0 finding): RRTMGP.solve()'s own internal construction (standalone.jl)
    # leaves the isothermal-boundary-layer row of p_lay/t_lay at zero -- it is filled
    # in later by add_isothermal_boundary_layer! inside prepare_atmosphere!. But
    # check_values[]=true validates BEFORE prepare_atmosphere! runs (see
    # update_fluxes!), so `RRTMGP.solve(profile; isothermal_boundary_layer=true)`
    # with check_values on FAILS on its very first call (verified below). We avoid
    # that here by pre-seeding the boundary row with the top-domain value, exactly
    # as RRTMGPSolver's constructor already does for center_z/face_z/
    # deep_atmosphere_inverse_scaling (src/api/solver.jl) but NOT for p_lay/t_lay.
    layerdata = zeros(FT, 4, nlay, ncol)  # (col_dry, p_lay, t_lay, rel_hum)
    layerdata[2, :, :] .= pad_top(profile.p_lay, nlay)
    layerdata[3, :, :] .= pad_top(profile.t_lay, nlay)

    cloud_state = if with_cloud
        RRTMGP.AtmosphericStates.CloudState(
            zeros(FT, nlay, ncol),  # cld_r_eff_liq
            zeros(FT, nlay, ncol),  # cld_r_eff_ice
            zeros(FT, nlay, ncol),  # cld_path_liq
            zeros(FT, nlay, ncol),  # cld_path_ice
            zeros(FT, nlay, ncol),  # cld_frac
            zeros(Bool, nlay, ncol), # mask_lw
            zeros(Bool, nlay, ncol), # mask_sw
            RRTMGP.AtmosphericStates.MaxRandomOverlap(),
            2, # ice_rgh = medium
        )
    else
        nothing
    end

    as = RRTMGP.AtmosphericStates.AtmosphericState(
        zeros(FT, ncol),          # lon (unused, must be an array like lat)
        FT.(profile.lat),         # lat
        layerdata,
        pad_top(profile.p_lev, nlev),
        pad_top(profile.t_lev, nlev),
        FT.(profile.t_sfc),
        vmr,
        cloud_state,
        nothing,                  # aerosol_state
    )
    return as
end

function build_bcs(::Type{FT}, lookups, ncol;
                    sfc_emis = FT(0.98), cos_zenith = FT(0.2588),
                    toa_flux = FT(551.58), albedo = FT(0.06)) where {FT}
    bcs_lw = RRTMGP.BCs.LwBCs(fill(FT(sfc_emis), lookups.nbnd_lw, ncol), nothing)
    bcs_sw = RRTMGP.BCs.SwBCs(
        fill(FT(cos_zenith), ncol),
        fill(FT(toa_flux), ncol),
        fill(FT(albedo), lookups.nbnd_sw, ncol),
        nothing,
        fill(FT(albedo), lookups.nbnd_sw, ncol),
    )
    return bcs_lw, bcs_sw
end

FT = Float64
domain_nlay = size(profile.p_lay, 1)  # 60
ncol = 1

grid_params_cs = RRTMGP.RRTMGPGridParams(
    FT; context = CTX, domain_nlay, ncol, isothermal_boundary_layer = true,
)
println("grid_params_cs.nlay (domain+boundary) = ", grid_params_cs.nlay,
        "  (domain_nlay=", domain_nlay, ")")

lookups_cs = RRTMGP.lookup_tables(grid_params_cs, RRTMGP.ClearSkyRadiation(false))
println("lookup_tables built: nbnd_lw=", lookups_cs.nbnd_lw, " nbnd_sw=", lookups_cs.nbnd_sw,
        " ngas_lw=", lookups_cs.ngas_lw, " ngas_sw=", lookups_cs.ngas_sw)

as_cs = build_state(FT, grid_params_cs, lookups_cs, profile; with_cloud = false)
bcs_lw_cs, bcs_sw_cs = build_bcs(FT, lookups_cs, ncol)

params = RRTMGP.default_parameters(FT)
sol_cs = RRTMGP.RRTMGPSolver(
    grid_params_cs, RRTMGP.ClearSkyRadiation(false), params, bcs_lw_cs, bcs_sw_cs, as_cs;
    lookups = lookups_cs,
)

RRTMGP.update_fluxes!(sol_cs)
println("RRTMGP.update_fluxes!(sol_cs) succeeded")

lw_up   = Array(RRTMGP.lw_flux_up(sol_cs))
lw_dn   = Array(RRTMGP.lw_flux_dn(sol_cs))
lw_net  = Array(RRTMGP.lw_flux_net(sol_cs))
sw_dn   = Array(RRTMGP.sw_flux_dn(sol_cs))
sw_up   = Array(RRTMGP.sw_flux_up(sol_cs))
sw_net  = Array(RRTMGP.sw_flux_net(sol_cs))
hr      = Array(RRTMGP.heating_rate(sol_cs))  # K/s, (nlay, ncol) domain-masked
p_lay   = Array(RRTMGP.layer_pressure(sol_cs))
p_lev   = Array(RRTMGP.level_pressure(sol_cs))
t_lay   = Array(RRTMGP.layer_temperature(sol_cs))
z_lay_approx = nothing

println()
println("Getter shapes (domain-masked, boundary layer excluded):")
println("  lw_flux_up  ", size(lw_up),  "  lw_flux_dn ", size(lw_dn), "  lw_flux_net ", size(lw_net))
println("  sw_flux_up  ", size(sw_up),  "  sw_flux_dn ", size(sw_dn), "  sw_flux_net ", size(sw_net))
println("  heating_rate ", size(hr), " (K/s)")
println("  layer_pressure ", size(p_lay), "  level_pressure ", size(p_lev))
println("  layer_temperature ", size(t_lay))

println()
println("OLR (lw_flux_up at level end = TOA)         = ", lw_up[end, 1], " W/m^2")
println("LW down at surface (lw_flux_dn at level 1)  = ", lw_dn[1, 1], " W/m^2")
println("SW down at TOA   (sw_flux_dn at level end)  = ", sw_dn[end, 1], " W/m^2")
println("SW down at surface (sw_flux_dn at level 1)  = ", sw_dn[1, 1], " W/m^2")
println()
println("flux_net = up - down check (LW, level 1, surface):")
println("  lw_net[1,1] computed        = ", lw_net[1, 1])
println("  lw_up[1,1] - lw_dn[1,1]     = ", lw_up[1, 1] - lw_dn[1, 1])
println("  (LW down at surface should exceed LW up if the surface is net-cooled by 'greenhouse' emission)")
println()
println("Heating-rate profile (K/day) sampled near 1, 5, 10, 15 km (domain z from profile.z_lev midpoints):")
z_lay_mid = (profile.z_lev[1:end-1, 1] .+ profile.z_lev[2:end, 1]) ./ 2
for ztarget in (1e3, 5e3, 10e3, 15e3)
    k = argmin(abs.(z_lay_mid .- ztarget))
    println("  z~", round(z_lay_mid[k]/1e3, digits=2), " km (layer ", k, "): ",
            round(86400*hr[k,1], digits=4), " K/day")
end
println()
println("Level 1 = surface check: p_lev[1,1] = ", p_lev[1,1], " Pa (expect ~ surface pressure, largest)")
println("           p_lev[end,1] = ", p_lev[end,1], " Pa (expect smallest, domain top before boundary pad)")

println()
println("FINDING: RRTMGP.solve(profile; isothermal_boundary_layer=true) + check_values[]=true")
println("reproduces a validation false-positive because validate_inputs runs BEFORE")
println("prepare_atmosphere! seeds the boundary-layer row of p_lay/t_lay:")
try
    RRTMGP.solve(
        profile; method = RRTMGP.ClearSkyRadiation(false), isothermal_boundary_layer = true,
        surface_emissivity = 0.98, cos_zenith = 0.2588, toa_flux = 551.58, surface_albedo = 0.06,
    )
    println("  (unexpectedly succeeded -- re-check this finding)")
catch e
    println("  CONFIRMED -- solve() raised: ", sprint(showerror, e))
end

# ---------------------------------------------------------------------------
section("(f) toa_flux convention: is sw_flux_dn(TOA) == toa_flux or toa_flux*cos_zenith?")
# ---------------------------------------------------------------------------

toa_flux_bc = Array(RRTMGP.toa_sw_flux_dn(sol_cs))[1]
cosz_bc = Array(RRTMGP.cos_zenith(sol_cs))[1]
println("toa_sw_flux_dn(sol) (the BC input)   = ", toa_flux_bc, " W/m^2")
println("cos_zenith(sol)                       = ", cosz_bc)
println("sw_flux_dn at TOA (level end, getter) = ", sw_dn[end, 1], " W/m^2")
println("toa_flux_bc                            = ", toa_flux_bc)
println("toa_flux_bc * cos_zenith                = ", toa_flux_bc * cosz_bc)
println("=> sw_flux_dn(TOA) matches ",
        isapprox(sw_dn[end,1], toa_flux_bc; rtol=1e-6) ? "toa_flux directly (unscaled by cos_zenith)" :
        isapprox(sw_dn[end,1], toa_flux_bc*cosz_bc; rtol=1e-6) ? "toa_flux * cos_zenith" :
        "NEITHER exactly -- inspect manually")

# ---------------------------------------------------------------------------
section("(g) Memory + timing: clear-sky and all-sky at several sizes")
# ---------------------------------------------------------------------------

function median_time(f, n = 5)
    f() # warm-up (also triggers any remaining compilation)
    ts = Float64[]
    for _ in 1:n
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e6) # ms
    end
    sort!(ts)
    return ts[(n + 1) ÷ 2], ts
end

function build_and_time(FT, domain_nlay, ncol, method; isothermal_boundary_layer = true)
    grid_params = RRTMGP.RRTMGPGridParams(
        FT; context = CTX, domain_nlay, ncol, isothermal_boundary_layer,
    )
    prof = RRTMGP.standard_atmosphere(FT; kind = :tropical, nlay = domain_nlay, ncol = ncol)
    lookups = RRTMGP.lookup_tables(grid_params, method)
    with_cloud = method isa RRTMGP.AllSkyRadiation ||
                 method isa RRTMGP.AllSkyRadiationWithClearSkyDiagnostics
    as = build_state(FT, grid_params, lookups, prof; with_cloud)
    bcs_lw, bcs_sw = build_bcs(FT, lookups, ncol)
    params_l = RRTMGP.default_parameters(FT)
    solver = RRTMGP.RRTMGPSolver(
        grid_params, method, params_l, bcs_lw, bcs_sw, as; lookups,
    )
    memMB = Base.summarysize(solver) / 1e6
    med, ts = median_time(() -> RRTMGP.update_fluxes!(solver))
    return solver, memMB, med, ts
end

sizes = [
    (:a_165x45,  165, 45),
    (:b_315x225, 315, 225),
    (:c_115x225_stride3, 115, 225),
]

results = Dict{Symbol, Any}()
for (tag, nlay, ncol_) in sizes
    for (method_name, method) in (
        (:clearsky, RRTMGP.ClearSkyRadiation(false)),
        (:allsky,   RRTMGP.AllSkyRadiation(false, false)),
    )
        key = Symbol(tag, :_, method_name)
        println("Building + timing: domain_nlay=$nlay ncol=$ncol_ method=$method_name ...")
        solver, memMB, med, ts = build_and_time(FT, nlay, ncol_, method)
        results[key] = (; memMB, med, ts, nlay, ncol_, method_name)
        println("  Base.summarysize(solver) = ", round(memMB, digits = 3), " MB")
        println("  update_fluxes! median of 5 (after 1 warm-up) = ", round(med, digits = 3), " ms",
                "  (all: ", round.(ts, digits = 3), ")")
        println("  per-column cost = ", round(1000 * med / ncol_, digits = 4), " core-us; ",
                "core-ms per column = ", round(med / ncol_, digits = 5))
    end
end

println()
println("Summary table (nlay, ncol, method -> memory MB, median update_fluxes! ms, ms/column):")
println(rpad("size", 24), rpad("method", 12), rpad("mem [MB]", 12), rpad("median [ms]", 14), "ms/column")
for (tag, nlay, ncol_) in sizes
    for method_name in (:clearsky, :allsky)
        key = Symbol(tag, :_, method_name)
        r = results[key]
        println(
            rpad("$(nlay)x$(ncol_)", 24),
            rpad(String(method_name), 12),
            rpad(string(round(r.memMB, digits=2)), 12),
            rpad(string(round(r.med, digits=3)), 14),
            round(r.med / ncol_, digits = 5),
        )
    end
end

println()
println("Done. Threads.nthreads() = ", Threads.nthreads())
