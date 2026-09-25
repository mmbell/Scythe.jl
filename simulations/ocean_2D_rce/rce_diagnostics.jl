#!/usr/bin/env julia
# RCE diagnostics: time series, mean profiles, energy budget and equilibrium check for an
# ocean_2D_rce (simulations/ocean_2D_rce/run.jl) output tree. Reads ONLY the comprehensive
# `<t>.nc` files the model itself writes (src/netcdf_output.jl, src/radiation_io.jl) -- it
# never runs the model, and it never writes into the run's own output tree except the
# three files named below (or into `--out` if given, which defaults to `--dir` itself).
#
#   julia --project=. simulations/ocean_2D_rce/rce_diagnostics.jl --dir <root> \
#       [--avg-days M] [--out <root>]
#
#   --dir <root>       a run_simulation (src/run_driver.jl) output root (holds
#                      config.toml, segments.csv, seg_01/, ...)
#   --avg-days M       length [days] of the trailing averaging/equilibrium window
#                       (default 5.0). Clamped to whatever the run actually has, with a
#                       warning, when the run is shorter.
#   --out <root>       where to write the three output files (default: same as --dir)
#
# Writes:
#   <out>/rce_timeseries.csv   one row per output time, domain means over x
#   <out>/rce_profiles.csv     one row per z, mean over the trailing averaging window
#   <out>/RCE_SUMMARY.md       equilibrium check, energy budget table, CAPE sanity check
#
# ── Reading conventions (see benchmarks/common/rce_io.jl for the shared helpers) ──
#
# Regular-grid prognostics (p, rho_d, rho_t, E_t, Q_ss, rho_v, rho_c, rho_r, u, w, T,
# theta, theta_e, RH, q_v, q_c, q_r, rho_ice/q_i, rho_e/e) are true spline evaluations on
# the file's regular (x, z) grid. Physics-group fields (K_m, K_h, e, mynn_*, radiation
# q_lw/q_sw/q_sw_applied/dT_*/olr/.../surface F_sh/F_q/tau_u/.../ust/U10/...) are linearly
# regridded from each scheme's own mish (src/netcdf_output.jl's `PHYSICS_REGRID_NOTE`).
# `x` is `LinRange(iMin, iMax, n)` INCLUDING both ends, so under the case's periodic
# lateral BCs the last column duplicates the first physically -- every domain mean here
# drops it (`benchmarks/common/rce_io.jl`'s `xmean`/`xmean_profile`, `interior` range).
#
# `q_sw_applied` and `dT_net` are ALREADY multiplied by the file's `radiation_sw_scale`
# (the call-time-zenith diurnal taper between RRTMGP calls); `q_sw`/`dT_sw` are NOT.
# `sw_sfc_dn/up` and `sw_toa_dn/up` are also NOT scaled (src/radiation_io.jl's
# `radiation_diagnostics`: only `q_sw_applied` and `dT_net` carry the multiplication) --
# every boundary SW flux this script reports is multiplied by `radiation_sw_scale` here.

using Printf
using Statistics
using Dates
using CSV
using DataFrames
using Springsteel.Thermodynamics: Rd, Rv, Eps, Cpd, Cpv, gravity, L_v0, p_0, L_v,
    q_sat_liquid

include(joinpath(@__DIR__, "..", "..", "benchmarks", "common", "rce_io.jl"))

const t0 = time()

# ── CLI ──────────────────────────────────────────────────────────────────────
root = nothing
avg_days_req = 5.0
outdir = nothing
let i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--dir"; global root = ARGS[i + 1]; i += 2
        elseif a == "--avg-days"; global avg_days_req = parse(Float64, ARGS[i + 1]); i += 2
        elseif a == "--out"; global outdir = ARGS[i + 1]; i += 2
        else
            error("rce_diagnostics.jl: unrecognized argument '$a'. Usage: --dir <root> " *
                  "[--avg-days M] [--out <root>]")
        end
    end
end
root === nothing && error("rce_diagnostics.jl: --dir <root> is required")
isdir(root) || error("rce_diagnostics.jl: --dir $root is not a directory")
outdir === nothing && (outdir = root)
mkpath(outdir)

# ── Parcel thermodynamics: dry lift + Bolton (1980) pseudo-adiabat, NOT the file's own
# (reversible) theta_e -- see the docstrings below for why and how. ──────────────────

"""Bolton (1980) eq. 43 pseudo-equivalent potential temperature [K]. `r_gkg` is the
mixing ratio [g/kg] AT this level (the parcel's own conserved value below the LCL, the
local saturation value above it); `T_lcl` [K] is the temperature at the lifting
condensation level of the parcel this point sits on -- for the SATURATED/local
equivalent theta of an already-saturated environmental point, pass `T` itself (a
saturated point is its own LCL)."""
function bolton_theta_ep(T::Float64, p_hpa::Float64, r_gkg::Float64, T_lcl::Float64)
    a = 0.2854 * (1.0 - 0.28e-3 * r_gkg)
    return T * (1000.0 / p_hpa)^a *
           exp((3.376 / T_lcl - 0.00254) * r_gkg * (1.0 + 0.81e-3 * r_gkg))
end

"Virtual temperature [K] for an unloaded (no liquid/ice) parcel or environment."
tvirtual(T::Float64, qv::Float64) = T * (1.0 + qv / Eps) / (1.0 + qv)

"""
    parcel_cape(z, p, T, qv; origin = :surface) -> (; cape, cin, z_lcl, z_lfc, z_el)

CAPE/CIN [J/kg] of a parcel lifted from the origin level (`z[1]`, `:surface` only --
that is all this benchmark needs). Dry-adiabatic lift (theta, q_v conserved) to the LCL,
found by per-level saturation-deficit search (bisected between the bracketing levels);
pseudo-adiabatic ascent above it by conserving the Bolton (1980) equivalent potential
temperature `theta_ep`, solved for the parcel temperature at every level by bisection
(implicit in T because `q_sat_liquid` is). The pseudo-adiabat is UNLOADED throughout:
condensate is assumed to fall out immediately, so the parcel's own water above the LCL
is exactly `q_sat_liquid(T_parcel, p)` with no liquid-water loading term in the virtual
temperature -- the standard "pseudo-adiabatic CAPE" convention (not reversible CAPE,
which the file's own `theta_e` corresponds to and which this function deliberately does
NOT use). Buoyancy is `B = g(Tv_parcel - Tv_env)/Tv_env`; CAPE integrates `max(B, 0)`
from the LFC (first level at/above the LCL with B > 0) to the EL (the next level above
the LFC where B turns back to <= 0); CIN integrates `min(B, 0)` from the origin to the
LFC (so CIN comes out <= 0 by construction, a deficit, not its magnitude).

`z, p, T, qv` must all be increasing-height, origin-first (`z[1]` is the surface) and
the same length; `p` in Pa, `T` in K, `qv` in kg/kg. Returns `cape = cin = 0.0,
z_lcl = z_lfc = z_el = NaN` for a column that never reaches saturation (a dry column
under the parcel's own dry-adiabatic lift) or never reaches a level of free convection.
"""
function parcel_cape(z::Vector{Float64}, p::Vector{Float64}, T::Vector{Float64},
                      qv::Vector{Float64}; origin::Symbol = :surface)
    origin === :surface || error("parcel_cape: only origin = :surface is implemented")
    n = length(z)
    (length(p) == n && length(T) == n && length(qv) == n) ||
        error("parcel_cape: z, p, T, qv must all be the same length")
    NO_LCL = (cape = 0.0, cin = 0.0, z_lcl = NaN, z_lfc = NaN, z_el = NaN)
    T0 = T[1]; p0 = p[1]; qv0 = qv[1]
    theta0 = T0 * (p_0 / (p0 / 100.0))^(Rd / Cpd)
    Tdry(pk) = theta0 * (pk / p0)^(Rd / Cpd)

    # ── Dry ascent to the LCL ──
    qs1 = q_sat_liquid(T0, p0 / 100.0)
    local z_lcl, p_lcl, T_lcl
    if qs1 - qv0 <= 0.0                       # already saturated at the origin
        z_lcl = z[1]; p_lcl = p0; T_lcl = T0
    else
        prev_margin = qs1 - qv0
        k_lcl = 0; frac = 0.0
        for k in 2:n
            Td = Tdry(p[k])
            margin = q_sat_liquid(Td, p[k] / 100.0) - qv0
            if margin <= 0.0
                frac = prev_margin / (prev_margin - margin)
                k_lcl = k - 1
                break
            end
            prev_margin = margin
        end
        k_lcl == 0 && return NO_LCL           # never saturates: a dry column, no CAPE
        z_lcl = z[k_lcl] + frac * (z[k_lcl + 1] - z[k_lcl])
        p_lcl = p[k_lcl] + frac * (p[k_lcl + 1] - p[k_lcl])
        T_lcl = Tdry(p_lcl)
    end
    r_gkg = qv0 * 1000.0
    theta_ep_target = bolton_theta_ep(T_lcl, p_lcl / 100.0, r_gkg, T_lcl)

    # ── Parcel temperature at every level: dry below the LCL, pseudo-adiabatic (bisected
    # on theta_ep) above it. ──
    Tparcel = Vector{Float64}(undef, n)
    qvparcel = Vector{Float64}(undef, n)
    for k in 1:n
        if z[k] <= z_lcl
            Tparcel[k] = Tdry(p[k])
            qvparcel[k] = qv0
        else
            phpa = p[k] / 100.0
            lo, hi = 150.0, 340.0
            for _ in 1:60
                mid = 0.5 * (lo + hi)
                rs_gkg = 1000.0 * q_sat_liquid(mid, phpa)
                f = bolton_theta_ep(mid, phpa, rs_gkg, T_lcl) - theta_ep_target
                f < 0.0 ? (lo = mid) : (hi = mid)
            end
            Tparcel[k] = 0.5 * (lo + hi)
            qvparcel[k] = q_sat_liquid(Tparcel[k], phpa)
        end
    end

    # ── Buoyancy ──
    B = Vector{Float64}(undef, n)
    for k in 1:n
        B[k] = gravity * (tvirtual(Tparcel[k], qvparcel[k]) - tvirtual(T[k], qv[k])) /
               tvirtual(T[k], qv[k])
    end
    k_lcl_idx = findfirst(k -> z[k] >= z_lcl, 1:n)
    k_lcl_idx === nothing && return NO_LCL
    k_lfc = findfirst(k -> k >= k_lcl_idx && B[k] > 0.0, 1:n)
    k_lfc === nothing &&
        return (cape = 0.0, cin = trapz(z[1:k_lcl_idx], min.(view(B, 1:k_lcl_idx), 0.0)),
                z_lcl = z_lcl, z_lfc = NaN, z_el = NaN)
    k_el = n
    for k in (k_lfc + 1):n
        if B[k] <= 0.0
            k_el = k
            break
        end
    end
    cape = trapz(z[k_lfc:k_el], max.(view(B, k_lfc:k_el), 0.0))
    cin = trapz(z[1:k_lfc], min.(view(B, 1:k_lfc), 0.0))
    return (cape = cape, cin = cin, z_lcl = z_lcl, z_lfc = z[k_lfc], z_el = z[k_el])
end

# ── Catalog and time span ───────────────────────────────────────────────────
snaps = catalog_snapshots(root)
t_first = snaps[1].t_abs
t_last = snaps[end].t_abs
span_days = (t_last - t_first) / 86400.0
println("rce_diagnostics: $(length(snaps)) snapshots under $root, span = " *
        @sprintf("%.3f", span_days) * " day(s)")

avg_days = avg_days_req
clamped = false
if avg_days > span_days
    avg_days = span_days
    clamped = true
end
window_start = t_last - avg_days * 86400.0

# ── Main pass: one row of time-series diagnostics per snapshot, and window-mean profile
# accumulation for every snapshot with t_abs >= window_start. ──────────────────────────
rows = Vector{Dict{String,Any}}(undef, length(snaps))
prof_names = ("theta", "theta_e", "T", "q_v", "RH", "q_c", "q_r", "q_i", "u", "K_h",
             "dT_net", "q_lw", "q_sw_applied", "e")
prof_sum = Dict{String,Vector{Float64}}()
prof_count = Dict{String,Int}(nm => 0 for nm in prof_names)
cloud_frac_sum = nothing
theta_es_sum = nothing
n_window = 0
z_grid = Float64[]

for (i, snap) in enumerate(snaps)
    NCDataset(snap.path, "r") do ds
        x, z, interior = xz_coords(ds)
        isempty(z_grid) && (global z_grid = z)

        has_mynn = has_group(ds, "mynn"); has_rad = has_group(ds, "radiation")
        has_sfc = has_group(ds, "surface")
        sw_scale = has_rad ? Float64(get(ds.attrib, "radiation_sw_scale", 0.0)) : 0.0

        T_xz = field_xz(ds, "T"); p_xz = field_xz(ds, "p"); qv_xz = field_xz(ds, "q_v")
        u_xz = field_xz(ds, "u"); w_xz = field_xz(ds, "w"); E_t_xz = field_xz(ds, "E_t")
        rho_e_xz = has_var(ds, "rho_e") ? field_xz(ds, "rho_e") : nothing

        T1 = T_xz[:, 1]                                   # lowest level (z = 0), (x,)
        u1 = u_xz[:, 1]; w1 = w_xz[:, 1]
        ke1 = 0.5 .* (u1 .^ 2 .+ w1 .^ 2)
        e_v = ((Cpv - Rv) .* T1) .+ ke1                    # + gravity*z1, z1 = 0 here

        precip = field_x(ds, "precip_rate")                # mm/hr
        PW = field_x(ds, "PW")

        F_sh = has_sfc ? field_x(ds, "F_sh") : fill(NaN, length(x))
        F_q = has_sfc ? field_x(ds, "F_q") : fill(NaN, length(x))
        tau_u = has_sfc ? field_x(ds, "tau_u") : fill(NaN, length(x))
        U10 = has_sfc ? field_x(ds, "U10") : fill(NaN, length(x))
        mynn_pblh = has_mynn ? field_x(ds, "mynn_pblh") : fill(NaN, length(x))

        olr = has_rad ? field_x(ds, "olr") : fill(NaN, length(x))
        olr_top = has_rad ? field_x(ds, "olr_model_top") : fill(NaN, length(x))
        lw_sfc_net = has_rad ? field_x(ds, "lw_sfc_up") .- field_x(ds, "lw_sfc_dn") :
                                fill(NaN, length(x))
        sw_sfc_net = has_rad ?
            sw_scale .* (field_x(ds, "sw_sfc_dn") .- field_x(ds, "sw_sfc_up")) :
            fill(NaN, length(x))
        sw_toa_net = has_rad ?
            sw_scale .* (field_x(ds, "sw_toa_dn") .- field_x(ds, "sw_toa_up")) :
            fill(NaN, length(x))
        lwp = has_rad ? field_x(ds, "lwp") : fill(NaN, length(x))
        iwp = has_rad ? field_x(ds, "iwp") : fill(NaN, length(x))
        cloudy = has_rad ? field_x(ds, "cloudy") : fill(NaN, length(x))

        # ── Column integrals (trapezoid, uniform z). colQrad is NaN (honestly undefined,
        # not a fabricated 0) when the radiation group is absent, e.g. the t=0 file. ──
        colE_col = [trapz(z, (rho_e_xz === nothing ? E_t_xz[c, :] :
                              E_t_xz[c, :] .+ rho_e_xz[c, :])) for c in 1:length(x)]
        colQrad_col = if has_rad
            q_lw = field_xz(ds, "q_lw"); q_sw_applied = field_xz(ds, "q_sw_applied")
            [trapz(z, q_lw[c, :] .+ q_sw_applied[c, :]) for c in 1:length(x)]
        else
            fill(NaN, length(x))
        end

        # ── Energy-budget forcing terms, per column then domain-averaged ──
        Fq_col = F_q
        e_l_col = (Cpv .* T1) .- L_v.(T1)   # J/kg, on the vapor-energy basis (~+g z = 0)
        precip_flux_col = precip ./ 3600.0                 # mm/hr -> kg m^-2 s^-1
        precip_sink_col = precip_flux_col .* (-e_l_col)    # >= 0 W/m^2 (a loss)
        LHF_col = L_v.(T1) .* Fq_col                        # diagnostic-only L_v F_q
        drag_col = tau_u .* u1

        # ── CAPE/CIN: domain-mean-profile parcel, and the x-mean of per-column parcels ──
        Tp = xmean_profile(T_xz, interior); pp = xmean_profile(p_xz, interior)
        qp = xmean_profile(qv_xz, interior)
        capemean = parcel_cape(z, pp, Tp, qp)
        cape_c = Float64[]; cin_c = Float64[]
        for c in interior
            r = parcel_cape(z, p_xz[c, :], T_xz[c, :], qv_xz[c, :])
            push!(cape_c, r.cape); push!(cin_c, r.cin)
        end

        sfc_sensible = xmean(F_sh, interior)
        sfc_latent_energy = xmean(e_v .* Fq_col, interior)
        drag_work = xmean(drag_col, interior)
        precip_sink = xmean(precip_sink_col, interior)
        rad_int = xmean(colQrad_col, interior)
        predicted = sfc_sensible + sfc_latent_energy - drag_work + rad_int - precip_sink

        rows[i] = Dict{String,Any}(
            "t_abs_s" => snap.t_abs, "t_abs_h" => snap.t_abs / 3600.0,
            "t_abs_days" => snap.t_abs / 86400.0, "segment" => snap.segment,
            "precip_rate_mm_hr" => xmean(precip, interior),
            "precip_rate_mm_day" => xmean(precip, interior) * 24.0,
            "PW_kgm2" => xmean(PW, interior),
            "F_sh_Wm2" => sfc_sensible, "F_q_kgm2s" => xmean(Fq_col, interior),
            "LHF_Wm2" => xmean(LHF_col, interior),
            "U10_ms" => xmean(U10, interior), "mynn_pblh_m" => xmean(mynn_pblh, interior),
            "olr_Wm2" => xmean(olr, interior), "olr_model_top_Wm2" => xmean(olr_top, interior),
            "lw_sfc_net_Wm2" => xmean(lw_sfc_net, interior),
            "sw_sfc_net_Wm2" => xmean(sw_sfc_net, interior),
            "sw_toa_net_Wm2" => xmean(sw_toa_net, interior),
            "lwp_gm2" => xmean(lwp, interior), "iwp_gm2" => xmean(iwp, interior),
            "cloud_frac" => xmean(cloudy, interior),
            "max_abs_w_ms" => maximum(abs.(w_xz[interior, :])),
            "colE_Jm2" => xmean(colE_col, interior), "colQrad_Wm2" => rad_int,
            "sfc_sensible_Wm2" => sfc_sensible, "sfc_latent_energy_Wm2" => sfc_latent_energy,
            "drag_work_Wm2" => drag_work, "precip_energy_sink_Wm2" => precip_sink,
            "predicted_dEdt_Wm2" => predicted,
            "cape_meanprofile_Jkg" => capemean.cape, "cin_meanprofile_Jkg" => capemean.cin,
            "cape_xmean_Jkg" => mean(cape_c), "cin_xmean_Jkg" => mean(cin_c),
        )

        # ── Window-mean profile accumulation. A physics-group field (K_h, dT_net, q_lw,
        # q_sw_applied) absent from THIS snapshot (t=0: "physics_groups = none (schemes
        # not yet run)") is skipped rather than added as NaN, which would otherwise NaN
        # out the whole window average for that field even though every OTHER snapshot
        # has it -- `prof_count[nm]` (not the shared `n_window`) is the divisor below. ──
        if snap.t_abs >= window_start
            for nm in prof_names
                has_var(ds, nm) || continue
                v = xmean_profile(field_xz(ds, nm), interior)
                acc = get!(prof_sum, nm, zeros(length(z)))
                acc .+= v
                prof_count[nm] += 1
            end
            q_c_v = has_var(ds, "q_c") ? field_xz(ds, "q_c") : zeros(length(x), length(z))
            q_i_v = has_var(ds, "q_i") ? field_xz(ds, "q_i") : zeros(length(x), length(z))
            cf = [mean((q_c_v[interior, k] .+ q_i_v[interior, k]) .> 1.0e-5)
                  for k in 1:length(z)]
            global cloud_frac_sum = cloud_frac_sum === nothing ? cf : cloud_frac_sum .+ cf
            # Saturated equivalent potential temperature of the (window-mean-to-be) local
            # profile, from THIS snapshot's domain-mean p/T -- summed here and divided by
            # n_window below like every other profile field.
            tes = [bolton_theta_ep(Tp[k], pp[k] / 100.0, 1000.0 * q_sat_liquid(Tp[k], pp[k] / 100.0),
                                   Tp[k]) for k in 1:length(z)]
            global theta_es_sum = theta_es_sum === nothing ? tes : theta_es_sum .+ tes
            global n_window += 1
        end
    end
end

# ── Actual dE/dt (centered finite difference of the domain-mean column energy) and the
# residual against the predicted forcing, added as two more time-series columns. ──────
t_series = [r["t_abs_s"] for r in rows]
colE_series = [r["colE_Jm2"] for r in rows]
n = length(rows)
for i in 1:n
    if n == 1
        dEdt = NaN
    elseif i == 1
        dEdt = (colE_series[2] - colE_series[1]) / (t_series[2] - t_series[1])
    elseif i == n
        dEdt = (colE_series[n] - colE_series[n - 1]) / (t_series[n] - t_series[n - 1])
    else
        dEdt = (colE_series[i + 1] - colE_series[i - 1]) / (t_series[i + 1] - t_series[i - 1])
    end
    rows[i]["actual_dEdt_Wm2"] = dEdt
    resid = dEdt - rows[i]["predicted_dEdt_Wm2"]
    rows[i]["residual_Wm2"] = resid
    denom = max(abs(rows[i]["predicted_dEdt_Wm2"]), 1.0)
    rows[i]["residual_frac"] = resid / denom
end

# ── Write rce_timeseries.csv ────────────────────────────────────────────────
ts_cols = ["t_abs_s", "t_abs_h", "t_abs_days", "segment", "precip_rate_mm_hr",
          "precip_rate_mm_day", "PW_kgm2", "F_sh_Wm2", "F_q_kgm2s", "LHF_Wm2", "U10_ms",
          "mynn_pblh_m", "olr_Wm2", "olr_model_top_Wm2", "lw_sfc_net_Wm2",
          "sw_sfc_net_Wm2", "sw_toa_net_Wm2", "lwp_gm2", "iwp_gm2", "cloud_frac",
          "max_abs_w_ms", "colE_Jm2", "colQrad_Wm2", "sfc_sensible_Wm2",
          "sfc_latent_energy_Wm2", "drag_work_Wm2", "precip_energy_sink_Wm2",
          "predicted_dEdt_Wm2", "actual_dEdt_Wm2", "residual_Wm2", "residual_frac",
          "cape_meanprofile_Jkg", "cin_meanprofile_Jkg", "cape_xmean_Jkg", "cin_xmean_Jkg"]
ts_df = DataFrame(; (Symbol(c) => [r[c] for r in rows] for c in ts_cols)...)
CSV.write(joinpath(outdir, "rce_timeseries.csv"), ts_df)

# ── Write rce_profiles.csv ──────────────────────────────────────────────────
n_window == 0 && error("no snapshot fell inside the averaging window -- this should " *
                       "be unreachable (window_start <= t_last always)")
prof_cols = Dict{String,Any}("z_m" => z_grid)
for nm in prof_names
    cnt = prof_count[nm]
    prof_cols[nm] = cnt == 0 ? fill(NaN, length(z_grid)) : prof_sum[nm] ./ cnt
end
prof_cols["cloud_fraction"] = cloud_frac_sum ./ n_window
prof_cols["theta_es"] = theta_es_sum ./ n_window
prof_col_order = vcat("z_m", collect(prof_names), "cloud_fraction", "theta_es")
prof_df = DataFrame(; (Symbol(c) => prof_cols[c] for c in prof_col_order)...)
CSV.write(joinpath(outdir, "rce_profiles.csv"), prof_df)

# ── Reference-profile CAPE sanity check (pbar/rho_vbar/rho_dbar/Tbar off the LAST
# snapshot -- these are the exact, time-invariant reference state, identical in every
# file: `exact_reference_state = 1`). ──────────────────────────────────────────────
ref_cape = NCDataset(snaps[end].path, "r") do ds
    z = field_z(ds, "z")
    pbar = field_z(ds, "pbar"); Tbar = field_z(ds, "Tbar")
    rho_dbar = field_z(ds, "rho_dbar"); rho_vbar = field_z(ds, "rho_vbar")
    qvbar = rho_vbar ./ rho_dbar
    parcel_cape(z, pbar, Tbar, qvbar)
end

# ── Equilibrium check over the trailing window ──────────────────────────────
win_idx = findall(r -> r["t_abs_s"] >= window_start, rows)
win_rows = rows[win_idx]
# nanmean, not mean: a t=0 snapshot's physics-group fields (F_sh, olr, ...) are NaN
# (physics_groups = "none (schemes not yet run)"), and a short run's window can include
# it (exactly the case for the quick smoke tree, whose whole 0.02-day span is the window).
mean_col(key) = nanmean(r[key] for r in win_rows)
precip_mmday = mean_col("precip_rate_mm_day")
colQrad_mean = mean_col("colQrad_Wm2")            # W/m^2, net radiative heating of column
rho_w = 1000.0
# (-<int q_rad dz>)/(rho_w * L_v0) has units (W/m^2)/((kg/m^3)*(J/kg)) = m/s of liquid
# water equivalent; * 86400 -> m/day; * 1000 -> mm/day.
rad_cooling_mmday = (-colQrad_mean / (rho_w * L_v0)) * 86400.0 * 1000.0
ratio_LvP_rad = (L_v0 * (precip_mmday / 1000.0 / 86400.0 * rho_w) + mean_col("sfc_sensible_Wm2")) /
                max(abs(-colQrad_mean), 1.0e-9)
slope_energy_Wm2 = ols_slope([r["t_abs_s"] for r in win_rows], [r["colE_Jm2"] for r in win_rows])
slope_PW_mmday = ols_slope([r["t_abs_days"] for r in win_rows], [r["PW_kgm2"] for r in win_rows])

# Diurnal composite of precip: hour-of-day bins from absolute time (t = 0 is hour 0).
diurnal_precip = fill(NaN, 24)
diurnal_n = zeros(Int, 24)
for r in win_rows
    hr = mod(floor(Int, r["t_abs_s"] / 3600.0), 24) + 1
    v = r["precip_rate_mm_day"]
    diurnal_n[hr] += 1
    diurnal_precip[hr] = isnan(diurnal_precip[hr]) ? v : diurnal_precip[hr] + v
end
for h in 1:24
    diurnal_n[h] > 0 && (diurnal_precip[h] /= diurnal_n[h])
end

equilibrated = abs(slope_energy_Wm2) < 5.0 && abs(slope_PW_mmday) < 0.5 &&
              0.9 <= ratio_LvP_rad <= 1.1

# ── Energy-budget table over the last window (last min(10, n_window) rows) ─────────
last_k = min(10, length(win_rows))
budget_rows = win_rows[(end - last_k + 1):end]
mean_residual = nanmean(r["residual_Wm2"] for r in win_rows)
mean_predicted = nanmean(r["predicted_dEdt_Wm2"] for r in win_rows)
mean_residual_frac = mean_residual / max(abs(mean_predicted), 1.0)
mean_residual_all = nanmean(r["residual_Wm2"] for r in rows)

# Taper loss: what radiation_z_max's stratospheric-extension taper removes from the
# column's own heating-rate integral relative to its own top/bottom boundary fluxes.
# toa_net_down uses olr_model_top (the flux AT the model's own top face) because the
# column integral ∫q_rad dz only spans the MODEL z grid (0..z[end]), not the full
# column (incl. extension) that `olr` is defined on.
taper_loss = nanmean(begin
    sfc_net_down = -r["lw_sfc_net_Wm2"] + r["sw_sfc_net_Wm2"]
    toa_net_down = r["sw_toa_net_Wm2"] - r["olr_model_top_Wm2"]
    r["colQrad_Wm2"] - (toa_net_down - sfc_net_down)
end for r in win_rows)

# ── RCE_SUMMARY.md ──────────────────────────────────────────────────────────
open(joinpath(outdir, "RCE_SUMMARY.md"), "w") do io
    println(io, "# RCE diagnostics summary")
    println(io)
    println(io, "Root: `$root`")
    println(io, @sprintf("Snapshots: %d, span %.3f day(s) (t = %.1f .. %.1f s)",
                         length(snaps), span_days, t_first, t_last))
    println(io, @sprintf("Averaging/equilibrium window: %.3f day(s) requested, %.3f day(s) used%s",
                         avg_days_req, avg_days,
                         clamped ? " -- **window shorter than requested** (run not that long yet)" : ""))
    println(io, @sprintf("Generated %s", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")))
    println(io)

    println(io, "## Equilibrium check (trailing window)")
    println(io)
    println(io, @sprintf("- precip: %.3f mm/day", precip_mmday))
    println(io, @sprintf("- radiative cooling equivalent: %.3f mm/day  (= -<int q_rad dz> / (rho_w L_v0) * 86400 * 1000)",
                         rad_cooling_mmday))
    println(io, @sprintf("- ratio (L_v*P + SH) / (-<int q_rad dz>): %.3f  (1.0 = surface+precip forcing balances radiative cooling)",
                         ratio_LvP_rad))
    println(io, @sprintf("- OLS slope of column energy vs time: %.3f W/m^2  (a TREND, not the instantaneous imbalance)",
                         slope_energy_Wm2))
    println(io, @sprintf("- OLS slope of PW vs time: %.3f mm/day", slope_PW_mmday))
    println(io)
    println(io, @sprintf("**EQUILIBRATED? %s**  (provisional thresholds: |energy slope| < 5 W/m^2, |PW slope| < 0.5 mm/day, ratio in 0.9..1.1)",
                         equilibrated ? "YES" : "NO"))
    println(io)

    println(io, "### Diurnal composite of precip (hour of absolute time, t=0 -> hour 0)")
    println(io)
    println(io, "| hour | mean precip (mm/day) | n |")
    println(io, "|---:|---:|---:|")
    for h in 1:24
        diurnal_n[h] > 0 &&
            println(io, @sprintf("| %d | %.3f | %d |", h - 1, diurnal_precip[h], diurnal_n[h]))
    end
    println(io)

    println(io, "## Column energy budget")
    println(io)
    println(io, "Forcing = <F_sh> + <(C_pv-R_v)*T1 + ke1 + g*z1) * F_q> - <tau_u u1> + " *
               "<int(q_lw+q_sw_applied)dz> - <precip energy sink>, compared against the " *
               "centered-finite-difference d/dt of the domain-mean column energy " *
               "<int(E_t + rho_e) dz>. This is a snapshot-forcing-vs-finite-difference-trend " *
               "comparison, not an exact discrete energy-conservation check (that would need " *
               "the model's own per-step accounting).")
    println(io)
    println(io, "| t (h) | SH | LatentE | Drag | RadInt | PrecipSink | Predicted | Actual dE/dt | Residual | Residual/|Pred| |")
    println(io, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in budget_rows
        println(io, @sprintf("| %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.3f |",
                             r["t_abs_h"], r["sfc_sensible_Wm2"], r["sfc_latent_energy_Wm2"],
                             r["drag_work_Wm2"], r["colQrad_Wm2"], r["precip_energy_sink_Wm2"],
                             r["predicted_dEdt_Wm2"], r["actual_dEdt_Wm2"], r["residual_Wm2"],
                             r["residual_frac"]))
    end
    println(io)
    println(io, @sprintf("Window-mean residual: %.4f W/m^2 (%.2f%% of |mean predicted forcing| = %.3f W/m^2)",
                         mean_residual, 100.0 * mean_residual_frac, abs(mean_predicted)))
    println(io, @sprintf("Whole-run-mean residual: %.4f W/m^2", mean_residual_all))
    println(io, @sprintf("Taper loss (<int q_rad dz> - (TOA net down - surface net down), model-top boundary): %.4f W/m^2",
                         taper_loss))
    println(io)

    println(io, "## Reference-profile CAPE sanity check")
    println(io)
    println(io, "Parcel lifted from the reference-state (`pbar`/`Tbar`/`rho_vbar`/`rho_dbar`) " *
               "surface, the exact steady sounding this RCE case starts from:")
    println(io, @sprintf("- CAPE = %.1f J/kg, CIN = %.1f J/kg, z_LCL = %.0f m, z_LFC = %s m, z_EL = %s m",
                         ref_cape.cape, ref_cape.cin, ref_cape.z_lcl,
                         isnan(ref_cape.z_lfc) ? "n/a" : @sprintf("%.0f", ref_cape.z_lfc),
                         isnan(ref_cape.z_el) ? "n/a" : @sprintf("%.0f", ref_cape.z_el)))
    println(io, "(published RCE-like tropical soundings give roughly 1000-2000 J/kg -- a " *
               "sanity range, not a target this run is checked against.)")
    println(io)

    println(io, "## Mean profiles over the window")
    println(io)
    println(io, "See `rce_profiles.csv`. Moist-neutrality check: mean surface `theta_e` " *
               "(the file's own REVERSIBLE equivalent potential temperature, " *
               "`netcdf_output.jl`'s definition) vs the column-minimum of the saturated " *
               "equivalent potential temperature `theta_es` computed here (Bolton, local " *
               "T/p, T_lcl = T -- NOT the same formula as the file's theta_e):")
    # Standard convective/potential-instability check (e.g. Betts 1986): theta_es(z) =
    # T*(1000/p)^kappa*(...) grows monotonically at great height purely from the pressure
    # factor, so a MINIMUM (not a maximum) is the physically meaningful comparison point
    # -- it marks the level a lifted, still-unsaturated surface parcel would find LEAST
    # resistance to further ascent once saturated. surface theta_e > min(theta_es aloft)
    # indicates the layer up to that level is potentially/convectively unstable.
    theta_e_sfc = prof_cols["theta_e"][1]
    theta_es_min, k_min = findmin(prof_cols["theta_es"])
    println(io, @sprintf("- surface theta_e = %.2f K; min theta_es aloft = %.2f K at z = %.0f m (surface theta_e - min theta_es = %.2f K; > 0 => potentially/convectively unstable up to that level)",
                         theta_e_sfc, theta_es_min, z_grid[k_min], theta_e_sfc - theta_es_min))
    println(io)
end

elapsed = time() - t0
println(@sprintf("rce_diagnostics.jl: wrote %s, %s, %s in %.1f s",
                 joinpath(outdir, "rce_timeseries.csv"), joinpath(outdir, "rce_profiles.csv"),
                 joinpath(outdir, "RCE_SUMMARY.md"), elapsed))
