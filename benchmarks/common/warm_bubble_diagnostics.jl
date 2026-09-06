# Shared diagnostics for the moist-compressible warm-bubble benchmarks (o01_rainfall.jl and
# ocean_warm_bubble.jl): output-snapshot enumeration, water path, the rain / ice / radiation
# diagnostic sets. MOVED VERBATIM out of benchmarks/o01_rainfall.jl (2026-09-05, MYNN-EDMF S1)
# so the second benchmark reuses them rather than copying them; every function keeps its
# o01_* name and its behaviour. Requires benchmarks/common/diagnostics.jl (mc_state,
# gauss_cell_weights, domain_integral) and the caller's `using CSV, DataFrames, Springsteel,
# Scythe`.

"""Sorted (time, path) pairs for every physical output snapshot of THIS run
(bounded by the integration time as a second guard against stale files)."""
function output_snapshots(model)
    files = filter(f -> endswith(f, "_physical.csv"), readdir(model.output_dir))
    pairs = [(parse(Float64, replace(f, "_physical.csv" => "")),
              joinpath(model.output_dir, f)) for f in files]
    filter!(p -> p[1] <= model.integration_time + 1.0e-6, pairs)
    return sort(pairs, by = first)
end

"""Domain-mean total water path [mm] of a snapshot (1 kg/m² of water ≡ 1 mm)."""
function water_path_mm(path, model, ref, kDim)
    df = CSV.read(path, DataFrame)
    ncols = div(nrow(df), kDim)
    rho_wbar = Springsteel.ref_rho_t(ref)[:, 1] .- Springsteel.ref_rho_d(ref)[:, 1]
    rho_w = (df.rho_t .- df.rho_d) .+ repeat(rho_wbar, ncols)
    W = domain_integral(reshape(rho_w, kDim, ncols), model)   # kg/m per unit y
    return W / (model.grid_params.iMax - model.grid_params.iMin)
end

"""
Rain diagnostics over every output snapshot: peak surface rain rate, onset time,
maximum rain density, and the time-integrated surface mass and energy fluxes
("surface" = the lowest mish level; Gauss nodes exclude z = 0 itself).
"""
function o01_rain_diagnostics(model, ref, kDim)
    # Slots 8 and 9 hold control variables when their transforms are on; `mc_state` needs
    # to know, and returns the recovered densities so nothing here reads a raw column.
    ctf = Scythe.condensate_transform_mode(model.options)
    cmu = get(model.physical_params, :condensate_mu, 1.0e-7)
    rtf = Scythe.rain_transform_mode(model.options)
    rmu = get(model.physical_params, :rain_mu, 1.0e-7)
    # Rain NUMBER (two-moment arm only): min over the run of the recovered n_r, the
    # negative-ringing monitor for the S9 positivity-vs-transform arm. The slot is a total
    # (zero reference), so recovery is `recover_n_r` alone; the column is "n_r" or, under
    # options[:rain_number_transform], its control-variable name "nu_nr". NaN when the run
    # carries no rain number, so the row stays informational and never gates a 1-moment run.
    nrtf = Scythe.rain_number_transform_mode(model.options)
    nrmu = get(model.physical_params, :mu_rain_n, 1.0)
    min_nr = NaN
    gp = model.grid_params
    snaps = output_snapshots(model)
    peak_rate = 0.0
    onset = NaN
    max_rr = 0.0
    min_rr = 0.0
    # The cloud undershoot was invisible in this CSV — only rain was reported — so every
    # rho_c experiment had to be read out of scythe_err.log. min_rho_v goes with it: it is
    # read off the PROGNOSTIC vapor slot now, so it reports the transported field's own
    # undershoot rather than the accumulated error of four others.
    max_rc = 0.0
    min_rc = 0.0
    min_rv = Inf
    # Total water rho_w = rho_t - rho_d, still worth reporting beside the vapor even though
    # the vapor is no longer assembled from it. It was the quantity that separated the two
    # ways a RESIDUAL vapor could fail (rho_w < 0 was the shadow of a negative rho_c, which
    # cancels in the residual; rho_v < 0 at healthy rho_w meant the condensate had claimed
    # more water than the column held — see the STAGE 5 RESOLUTION of
    # reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md). Now the two are independent fields,
    # and rho_w - (rho_v + rho_c + rho_r) is the reconciliation gap instead.
    min_rw = Inf
    # Dry-density headroom, as a FRACTION of the reference (rho_d spans two decades over the
    # column, so an absolute minimum would only ever report the model top). Diagnostic only:
    # rho_d has no rate sinks at all, so it cannot be depleted the way rho_c is — a fraction
    # that departs from 1 by more than rounding means the dynamics are moving mass, and one
    # that approaches 0 is a blow-up, not a positive-definiteness problem. Positivity bounds
    # for rho_d/rho_t exist and are tested but stay OFF (benchmarks/FUTURE_WORK.md); this is
    # the number that would say when to reconsider.
    min_rd_frac = Inf
    times = Float64[]
    rate_int = Float64[]    # domain-integrated surface rain flux [kg/(m s) per unit y]
    eflux_int = Float64[]   # domain-integrated surface energy flux [J/(m s) per unit y]
    Wh = Float64[]
    for (t, path) in snaps
        df = CSV.read(path, DataFrame)
        ncols = div(nrow(df), kDim)
        surf = 1:kDim:nrow(df)
        Tk, _, rho_d, rho_v, rho_c, rho_t, rho_r =
            mc_state(df, ref, kDim, ncols; transform = ctf, mu = cmu,
                     rain_transform = rtf, rain_mu = rmu)
        max_rr = max(max_rr, maximum(rho_r))
        min_rr = min(min_rr, minimum(rho_r))
        max_rc = max(max_rc, maximum(rho_c))
        min_rc = min(min_rc, minimum(rho_c))
        min_rv = min(min_rv, minimum(rho_v))
        nr_col = "nu_nr" in names(df) ? "nu_nr" : ("n_r" in names(df) ? "n_r" : "")
        if !isempty(nr_col)
            n_r = Scythe.recover_n_r.(df[!, nr_col], nrtf, nrmu)
            m = minimum(n_r)
            min_nr = isnan(min_nr) ? m : min(min_nr, m)
        end
        min_rw = min(min_rw, minimum(rho_t .- rho_d))
        min_rd_frac = min(min_rd_frac,
                          minimum(rho_d ./ repeat(Springsteel.ref_rho_d(ref)[:, 1], ncols)))
        rr_s = max.(rho_r[surf], 0.0)
        Vt = Scythe.rain_terminal_velocity.(rr_s, rho_d[surf], Tk[surf])
        R = -rr_s .* Vt                                       # kg/m²/s, >= 0
        pk = maximum(R)
        peak_rate = max(peak_rate, pk)
        if isnan(onset) && pk > 1.0e-3                        # 1 g m^-2 s^-1
            onset = t
        end
        if isempty(Wh)
            Wh = gauss_cell_weights(ncols, gp.num_cells, gp.iMax - gp.iMin,
                                    ncols ÷ gp.num_cells, gp.quadrature)
        end
        e_l = (Scythe.Cpv .* Tk[surf]) .- Scythe.L_v.(Tk[surf])   # + g*z ≈ 0 at the surface
        push!(times, t)
        push!(rate_int, sum(Wh .* R))
        push!(eflux_int, sum(Wh .* (rr_s .* Vt .* e_l)))      # F_E(z≈0), > 0 for e_l < 0
    end
    # Trapezoid over the 60 s output cadence (few-% cross-check, not the budget)
    accum_flux = 0.0
    accum_E = 0.0
    for i in 1:(length(times) - 1)
        dt = times[i+1] - times[i]
        accum_flux += 0.5 * (rate_int[i] + rate_int[i+1]) * dt
        accum_E += 0.5 * (eflux_int[i] + eflux_int[i+1]) * dt
    end
    width = gp.iMax - gp.iMin
    return Dict(
        "peak_rain_rate_gm2s" => 1000.0 * peak_rate,
        "rain_onset_min" => onset / 60.0,
        "max_rho_r_gm3" => 1000.0 * max_rr,
        "min_rho_r_gm3" => 1000.0 * min_rr,             # spline undershoot monitor
        "max_rho_c_gm3" => 1000.0 * max_rc,
        "min_rho_c_gm3" => 1000.0 * min_rc,             # ditto, for the condensate
        "min_rho_v_gm3" => 1000.0 * min_rv,             # the prognostic vapor's headroom
        "min_n_r_perm3" => min_nr,                      # rain-number ringing monitor (NaN if 1-moment)
        "min_rho_w_gm3" => 1000.0 * min_rw,             # total water; separates the two failures
        "min_rho_d_frac" => min_rd_frac,                # dry density / reference; 1 = untouched
        "accum_rainfall_flux_mm" => accum_flux / width, # cross-check of the exact budget
        "precip_energy_gain_Jm2" => accum_E / width,    # predicted domain E_t gain
    )
end

"""
    o01_ice_diagnostics(model, ref, kDim) -> Dict

Ice diagnostics over every output snapshot, for `options[:ice_microphysics] = :ishmael`.
Returns an empty `Dict` when ice is off, so the caller can `merge` unconditionally.

Per species (1 = planar-nucleated, 2 = columnar-nucleated, 3 = aggregates): the extreme mass
density and the extreme number density over the whole run, the peak ice water path, and the
HEIGHTS between which ice ever appears. The last is the physical gate — ice above the freezing
level and only falling through below it — and it is reported as a height rather than a mask
because that is what a Dunion-sounding run can be checked against by eye (the 0 °C level sits
near 4.7 km).

`min_rho_i*` is the spline-undershoot monitor, exactly as `min_rho_r_gm3` is for the rain, and
the ice transform (`options[:ice_transform]`) is undone here when it is on — the CSV column is
the control variable, not the density.
"""
function o01_ice_diagnostics(model, ref, kDim)
    Scythe.ice_microphysics(model.options) === :ishmael && return _o01_ice_diags(model, ref, kDim)
    return Dict{String,Float64}()
end

# See the Z cap comment inside _o01_ice_diags. 17 km (the O01 sponge base) + 3 km.
const Z_ICEDIAG_CAP = 20.0e3

function _o01_ice_diags(model, ref, kDim)
    itf = Scythe.ice_transform_mode(model.options)
    names = Scythe.ice_var_names(model.options)
    gp = model.grid_params
    patch = createGrid(gp)
    z = Scythe.getGridpoints(patch)[1:kDim, end]
    d = Dict{String,Float64}()
    maxq = zeros(3); minq = zeros(3); maxn = zeros(3)
    iwp = 0.0
    ztop = -Inf
    zbot = Inf
    for (_, path) in output_snapshots(model)
        df = CSV.read(path, DataFrame)
        ncols = div(nrow(df), kDim)
        tot = zeros(nrow(df))
        for k in 1:3
            mu_q = Scythe.ice_mu(model.physical_params, 4 * (k - 1) + 1)
            mu_n = Scythe.ice_mu(model.physical_params, 4 * (k - 1) + 2)
            q = Scythe.recover_total.(df[!, names[4 * (k - 1) + 1]], itf, mu_q)
            n = Scythe.recover_total.(df[!, names[4 * (k - 1) + 2]], itf, mu_n)
            maxq[k] = max(maxq[k], maximum(q))
            minq[k] = min(minq[k], minimum(q))
            maxn[k] = max(maxn[k], maximum(n))
            tot .+= max.(q, 0.0)
        end
        iwp = max(iwp, domain_integral(reshape(tot, kDim, ncols), model) /
                       (gp.iMax - gp.iMin))
        # Where the ice IS. The threshold is the LARGER of a fixed floor and 1% of this
        # snapshot's own peak, so it reports the anvil at any loading: a fixed 1 mg/m^3 gate
        # returns NaN for an early snapshot whose whole ice field is 3e-5 g/m^3, which says
        # nothing about where the ice is.
        thr = max(1.0e-14, 0.01 * maximum(tot))
        # The scan is capped at Z_ICEDIAG_CAP: the DK83 sponge occupies z > 17 km on the
        # O01 case, and the accepted full-resolution run showed ~0.1 mg/m^3 of number-less
        # trace mass accumulating against the model lid inside it (fit ringing at the
        # boundary, inert to every rate: FINDINGS 5m), which put ice_top at the lid while
        # the physical overshoot topped out at ~20 km. The cap sits 3 km above the sponge
        # base so the overshoot stays readable and only the lid trace is excluded.
        live = findall(>(thr), reshape(tot, kDim, ncols))
        if !isempty(live)
            ztop = max(ztop, maximum(z[c.I[1]] for c in live if z[c.I[1]] <= Z_ICEDIAG_CAP;
                                     init = -Inf))
            zbot = min(zbot, minimum(z[c.I[1]] for c in live))
        end
    end
    for k in 1:3
        d["max_rho_i$(k)_gm3"] = 1000.0 * maxq[k]
        d["min_rho_i$(k)_gm3"] = 1000.0 * minq[k]
        d["max_n_i$(k)_perL"] = maxn[k] / 1000.0
    end
    d["max_ice_water_path_mm"] = iwp
    d["ice_top_km"] = isfinite(ztop) ? ztop / 1000.0 : NaN
    d["ice_base_km"] = isfinite(zbot) ? zbot / 1000.0 : NaN
    return d
end

"""
    o01_radiation_diagnostics(model, ref, kDim) -> Dict

Radiation diagnostics read from the sidecar NetCDF (`Scythe.read_radiation`,
src/radiation_io.jl), merged into `o01_diagnostics` ONLY when a sidecar exists
(`SCYTHE_O01_RAD` on and `:radiation_output` not disabled) — an empty `Dict` on every
non-radiation arm, exactly the pattern `o01_ice_diagnostics` uses for ice.

INFORMATIONAL ONLY (S5): no `expected_values` targets exist for these keys, so nothing
here can fail `check_targets` or gate `passed` — they land in `diagnostics.csv` and the
JSONL record as plain rows, per the plan's stated policy (a reference is seeded only
after the user has tested a radiation arm on real output).

`ref`/`kDim` are accepted for signature symmetry with the other `o01_*_diagnostics`
functions (`o01_diagnostics` calls all three the same way) but unused — the sidecar
already carries its own vertical coordinate and reference-column subtraction.

- `olr_mean_t0`/`olr_min_run`/`olr_mean_final`: domain-mean OLR [W/m^2] at the first
  snapshot, the minimum over every snapshot (the cloud's OLR depression), and the
  domain mean at the last snapshot.
- `lw_cooling_5km_kday_t0`/`sw_heating_5km_kday_t0`: domain-mean `dT_lw`/`sw_scale*dT_sw`
  [K/day] at the layer nearest 5 km, at the FIRST snapshot (before the bubble has done
  much) — the number the S1 offline clear-sky anchor is directly comparable to. The SW
  one is 0 when the arm carries no shortwave (`options[:solar] = :none`).
- `cloud_top_cooling_min_kday`: the minimum `dT_lw` over every layer, column and
  snapshot — the anvil/cloud-top cooling extreme the run ever produces.
- `max_lwp_gm2`/`max_iwp_gm2`: the largest column liquid/ice water path [g/m^2] seen at
  any snapshot.
- `re_liq_clamps`/`re_ice_clamps`: the cloud-optics saturation counters from the FINAL
  sidecar (cumulative over the run, exactly what `radiation_trace!`'s log line reports).
"""
function o01_radiation_diagnostics(model, ref, kDim)
    tags = Scythe.radiation_snapshots(model.output_dir)
    isempty(tags) && return Dict{String,Float64}()

    rad0 = Scythe.read_radiation(model.output_dir, tags[1])
    radN = Scythe.read_radiation(model.output_dir, tags[end])
    k5 = argmin(abs.(rad0.z .- 5.0e3))
    has_sw = !(rad0.solar in ("none", ""))

    olr_min_run = Inf
    cloud_top_cooling_min = Inf
    max_lwp = 0.0
    max_iwp = 0.0
    for tag in tags
        r = Scythe.read_radiation(model.output_dir, tag)
        olr_min_run = min(olr_min_run, minimum(r.olr))
        cloud_top_cooling_min = min(cloud_top_cooling_min, minimum(r.dT_lw))
        max_lwp = max(max_lwp, maximum(r.lwp))
        max_iwp = max(max_iwp, maximum(r.iwp))
    end

    return Dict(
        "olr_mean_t0" => sum(rad0.olr) / length(rad0.olr),
        "olr_min_run" => olr_min_run,
        "olr_mean_final" => sum(radN.olr) / length(radN.olr),
        "lw_cooling_5km_kday_t0" => sum(view(rad0.dT_lw, :, k5)) / size(rad0.dT_lw, 1),
        "sw_heating_5km_kday_t0" => has_sw ?
            rad0.sw_scale * sum(view(rad0.dT_sw, :, k5)) / size(rad0.dT_sw, 1) : 0.0,
        "cloud_top_cooling_min_kday" => cloud_top_cooling_min,
        "max_lwp_gm2" => max_lwp,
        "max_iwp_gm2" => max_iwp,
        "re_liq_clamps" => Float64(radN.n_clamp_re_liq),
        "re_ice_clamps" => Float64(radN.n_clamp_re_ice),
    )
end

