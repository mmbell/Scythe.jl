# test/test_slot3_audit.jl -- no source line indexes the moist-compressible state by a literal
# slot that is not in the ALLOWLIST (Stage 4 of the rho_t retirement deleted slot 3, the
# prognostic total density "rho_t"; the vapor is slot 3 now and every reader resolves it by
# name: `mc_slots(model).rho_v`, `mc_slot(vars, "rho_v")`).
#
# Every non-comment line of src/, benchmarks/, simulations/ and tools/ (*.jl) matching one of
#     `, 3]`   `[3]`   `[..., 3, ...]`   `view(..., 3, ...)` / `view(..., 3)`  (any spacing)
#     `slots.rho_t`   `"rho_t`
#     a state array (physical, spectral, var_np1, vnp1, expdot, impdot, diffdot, hacdot, phys,
#     and the `ph`/`sp` aliases) indexed with ANY literal slot 3..24, in `[...]` or `view(...)`
# must be in the ALLOWLIST below (exact stripped text and count, with the reason), and every
# allowlisted line must still exist (the list cannot rot). Spacing is free in every pattern:
# `expdot[colstart:colend,3]` and `view(phys, :, 3, 1)` are the two forms the first version of
# this audit missed (7 of the 18 real slot-3 reads at bf577b2; see `slot3_audit_hits`).
#
# Whole files skipped (SLOT3_WHOLE_FILE): src/checkpoint_conversion.jl, whose subject is the
# pre-Stage-4 layout, and the four legacy model families, which carry their OWN state layouts
# and never touch the moist-compressible one -- asserted below, so a file that starts to
# reference the MC state loses its exemption.
#
# Standalone:
#   julia --project=. -e 'using Test; include("test/test_slot3_audit.jl")'
# Against another tree (e.g. a pre-Stage-4 checkout), after the include:
#   slot3_audit_hits("/path/to/tree")

using Test

const SLOT3_STATE = "(?:physical|spectral|var_np1|vnp1|expdot|impdot|diffdot|hacdot|phys|ph|sp)"
const SLOT3_LIT = "(?:[3-9]|1[0-9]|2[0-4])"
const SLOT3_PATTERNS = [
    r",\s*3\s*\]",                                                    # x[.., 3]
    r"\[\s*3\s*\]",                                                   # x[3]
    r"\[[^\[\]]*,\s*3\s*,[^\[\]]*\]",                                 # x[.., 3, ..]
    r"view\([^()]*,\s*3\s*[,)]",                                      # view(x, .., 3, ..) / view(x, .., 3)
    r"slots\.rho_t\b",
    r"\"rho_t",
    Regex("\\b$(SLOT3_STATE)\\[[^\\[\\]]*,\\s*$(SLOT3_LIT)\\s*[,\\]]"),   # state[.., N, ..] / state[.., N]
    Regex("view\\(\\s*$(SLOT3_STATE)\\s*,[^()]*,\\s*$(SLOT3_LIT)\\s*[,)]"),  # view(state, .., N, ..)
]
const SLOT3_DIRS = ("src", "benchmarks", "simulations", "tools")
const SLOT3_LEGACY = ["src/primitive_equations.jl", "src/testModels.jl",
                      "src/shallowWaterModels.jl", "src/tcblModels.jl"]
const SLOT3_WHOLE_FILE = Set(["src/checkpoint_conversion.jl"; SLOT3_LEGACY])
const SLOT3_MC_TOKENS = r"mc_slots|mc_slot\(|MC_VARS|moist_compressible|mc_var_names|mc_state"

"""Every matching non-comment line under `root`'s SLOT3_DIRS, keyed by (relative path,
stripped text), with its count."""
function slot3_audit_hits(root::AbstractString)
    hits = Dict{Tuple{String,String},Int}()
    for d in SLOT3_DIRS
        isdir(joinpath(root, d)) || continue
        for (dir, _, fs) in walkdir(joinpath(root, d)), f in sort(fs)
            endswith(f, ".jl") || continue
            rel = relpath(joinpath(dir, f), root)
            rel in SLOT3_WHOLE_FILE && continue
            # Hidden directories (Jupyter's .ipynb_checkpoints, editor caches) are not source:
            # the main checkout carries gitignored 2024-25 checkpoint copies of old files.
            any(startswith(p, ".") for p in splitpath(rel)) && continue
            for l in eachline(joinpath(dir, f))
                t = strip(l)
                startswith(t, "#") && continue
                any(p -> occursin(p, t), SLOT3_PATTERNS) || continue
                key = (rel, String(t))
                hits[key] = get(hits, key, 0) + 1
            end
        end
    end
    return hits
end

const SLOT3_ALLOWLIST = [
    # benchmarks/bf02_moist.jl: thermo tuple element 3 = T
    ("benchmarks/bf02_moist.jl", "Tk = [x[3] for x in thermo]", 1),
    # benchmarks/common/diagnostics.jl: thermo tuple element 3 = T; detection of a pre-Stage-4 CSV (a "rho_t" column)
    ("benchmarks/common/diagnostics.jl", "Tk = [x[3] for x in thermo]", 1),
    ("benchmarks/common/diagnostics.jl", "legacy = \"rho_t\" in names(df)", 1),
    # benchmarks/condensate_transform_diagnostic.jl: a pre-Stage-4 CSV's rho_t column; derivative planes 1 and 4 (value, d/dz) of a named slot
    ("benchmarks/condensate_transform_diagnostic.jl", "\"rho_t\" in names(df) && return df.rho_t .+ tile(R.rho_tbar)", 1),
    ("benchmarks/condensate_transform_diagnostic.jl", "return copy(patch.physical[:, slot, 1]), copy(patch.physical[:, slot, 4])", 1),
    # benchmarks/mass_error_diagnostic.jl: derivative planes 2 and 4 of a named slot; trapezoid weights
    ("benchmarks/mass_error_diagnostic.jl", "return copy(patch.physical[:, slot, 2]), copy(patch.physical[:, slot, 4])   # ∂/∂x, ∂/∂z", 1),
    ("benchmarks/mass_error_diagnostic.jl", "trap(r) = dt * (0.5 * r[1] + r[2] + 0.5 * r[3])", 1),
    # benchmarks/o01_movie.jl: Makie figure grid position
    ("benchmarks/o01_movie.jl", "ax_kh = is_bl ? Axis(fig[1, 3], xlabel = \"x (km)\", ylabel = \"z (km)\") : nothing", 1),
    ("benchmarks/o01_movie.jl", "ax_rad = is_rad ? Axis(fig[2, 3], xlabel = \"x (km)\", ylabel = \"z (km)\") : nothing", 1),
    # benchmarks/o01_rainfall.jl: the ice mu list, element 3 = a
    ("benchmarks/o01_rainfall.jl", "physical_params[:mu_ice_a] = imus[3]", 1),
    # simulations/ocean_2D_rce/rce_movie.jl: Makie figure grid position
    ("simulations/ocean_2D_rce/rce_movie.jl", "ax_cond = Axis(fig[2, 3]; xlabel = \"x (km)\", ylabel = \"z (km)\", title = \"condensate mass (species)\")", 1),
    ("simulations/ocean_2D_rce/rce_movie.jl", "ax_rh = Axis(fig[3, 3]; xlabel = \"x (km)\", ylabel = \"z (km)\", title = \"RH\")", 1),
    # simulations/ocean_2D_rce/rce_sounding.jl: text-row column 3
    ("simulations/ocean_2D_rce/rce_sounding.jl", "@printf(io, \"%.4f %.4f %.5f %.2f %.2f\\n\", r[1], r[2], r[3], r[4], r[5])", 1),
    ("simulations/ocean_2D_rce/rce_sounding.jl", "parse(Float64, parts[3]), u, v))", 1),
    ("simulations/ocean_2D_rce/rce_sounding.jl", "qv_full = vcat(qv_mean, [r[3] / 1000.0 for r in splice_rows])   # kg/kg", 1),
    # simulations/tc_axisym/tc_movie.jl: Makie figure grid position
    ("simulations/tc_axisym/tc_movie.jl", "ax_bl = showbl ? Axis(fig[2, 3]; xlabel = \"radius (km)\", ylabel = \"height (km)\",", 1),
    ("simulations/tc_axisym/tc_movie.jl", "ax_rad = showrad ? Axis(fig[2, 3]; xlabel = \"radius (km)\", ylabel = \"height (km)\",", 1),
    # simulations/tc_axisym/tc_postprocess.jl: the DERIVED rho_t output field and the pre-Stage-4 file detection; reference-file column 3
    ("simulations/tc_axisym/tc_postprocess.jl", "(\"rho_tbar\", bg.rho_tbar, \"kg m-3\"),", 1),
    ("simulations/tc_axisym/tc_postprocess.jl", "handled = Set([\"time\", \"x\", \"z\", \"p\", \"rho_d\", \"rho_t\", \"u\", \"w\",", 1),
    ("simulations/tc_axisym/tc_postprocess.jl", "haskey(dc, \"rho_t\") && length(dc[\"x\"]) == n_i && length(dc[\"z\"]) == n_k", 1),
    ("simulations/tc_axisym/tc_postprocess.jl", "haskey(ds, \"rho_t\") && @warn(string(rawpath, \" carries a pre-Stage-4 rho_t slot; \",", 1),
    ("simulations/tc_axisym/tc_postprocess.jl", "rdm[i] = parse(Float64, parts[3]); rvm[i] = parse(Float64, parts[4])", 1),
    ("simulations/tc_axisym/tc_postprocess.jl", "rho_t = rd2d(dc, \"rho_t\", n_i, n_k)", 1),
    ("simulations/tc_axisym/tc_postprocess.jl", "wr(\"rho_t\", rho_t, \"kg m-3\", \"total density (derived: rho_d + sum_k alpha_k nu_k)\")", 1),
    # src/exact_si.jl: column 3 of exact_si's own unknown block (u*, w*, v*/p*), filled by name
    ("src/exact_si.jl", "PR[i, k] = xstar[r, 3]", 1),
    ("src/exact_si.jl", "hfields[r, 3] = dp", 1),
    ("src/exact_si.jl", "view(mtile.var_np1, colstart:colend, vars[\"p\"]) .+= view(hfields, rows, 3)", 1),
    ("src/exact_si.jl", "xstar[rows, 3] .= view(mtile.var_np1, :, vars[\"p\"])", 1),
    ("src/exact_si.jl", "xstar[rows, 3] .= view(mtile.var_np1, :, vars[\"v\"])", 1),
    # src/horizontal_si.jl: plane 3 of the horizontal SI level increments (its own layout)
    ("src/horizontal_si.jl", "_hsi_add_increment!(spectral, hsd, patch, hsd.rhod_index, view(dlev, :, :, 3))", 1),
    ("src/horizontal_si.jl", "dlev[i, k, 3] = -cd * dxu", 1),
    ("src/horizontal_si.jl", "x = hsd.wshare[k, n] * (dlev[i, k, 4] - dlev[i, k, 3])", 1),
    # src/idealized.jl: thermo tuple element 3 = T; the native-init working grid's ONE variable, named rho_t; derivative plane 4 (d/dz)
    ("src/idealized.jl", "Dict(\"rho_t\" => gp.max_wavenumber[\"rho_d\"]) :", 1),
    ("src/idealized.jl", "Dict(\"rho_t\" => gp.spline_filter[\"rho_d\"]) :", 1),
    ("src/idealized.jl", "H[i]  = patch.physical[i, p_i, 4] + (gravity * rtp)", 1),
    ("src/idealized.jl", "THE WORKING TOTAL DENSITY of [`balanced_vortex_native!`](@ref): a one-variable grid (`\"rho_t\"`)", 1),
    ("src/idealized.jl", "Tk = [x[3] for x in thermo]", 1),
    ("src/idealized.jl", "haskey(d, \"default\") ? Dict(\"rho_t\" => d[\"default\"]) : empty(d))", 1),
    ("src/idealized.jl", "haskey(gp.l_q, \"rho_d\" * sfx) && (lq[\"rho_t\" * sfx] = gp.l_q[\"rho_d\" * sfx])", 1),
    ("src/idealized.jl", "keep = Dict{Symbol,Any}(:vars => Dict(\"rho_t\" => 1),", 1),
    ("src/idealized.jl", "one(d) = (haskey(d, \"rho_d\") ? Dict(\"rho_t\" => d[\"rho_d\"]) :", 1),
    ("src/idealized.jl", "rhot_target[i] = -patch.physical[i, p_i, 4] / gravity   # slot 4 = ∂z", 1),
    ("src/idealized.jl", "wres[i] = -(patch.physical[i, p_i, 4] + (gravity * rtp)) / rho_t", 1),
    # src/ishmael.jl: ISHMAEL process/category tables
    ("src/ishmael.jl", "ip_33 = ISHMAEL_IPAIR[3, 3]; ip_44 = ISHMAEL_IPAIR[4, 4]", 1),
    ("src/ishmael.jl", "ip_34 = ISHMAEL_IPAIR[3, 4]; ip_43 = ISHMAEL_IPAIR[4, 3]", 1),
    ("src/ishmael.jl", "qi_qr_nrm, qi_qr_nrd, qi_qr_nrn = procr[1], procr[2], procr[3]", 1),
    # src/ishmael_tables.jl: quadrature abscissae
    ("src/ishmael_tables.jl", "(x[3] > xup) && error(\"ishmael_avint: less than 3 function values between integration limits\")", 1),
    # src/mc_boundary_layer.jl: the fixed MC layout (u 4, w 5, E_t 6, Q_ss 7, nu_r 8, nu_c 9, v 10); the anchor member alpha (member 3 = rain)
    ("src/mc_boundary_layer.jl", "@inbounds expdot[j, 10] += dv", 1),
    ("src/mc_boundary_layer.jl", "av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]", 1),
    ("src/mc_boundary_layer.jl", "expdot[j, 4] += du", 1),
    ("src/mc_boundary_layer.jl", "expdot[j, 5] += dw", 1),
    ("src/mc_boundary_layer.jl", "expdot[j, 6] += qdot + ((((Cvv * Tk[i]) * real_v) + (((Cpv * Tk[i]) - Lv[i]) * real_c)) +", 1),
    ("src/mc_boundary_layer.jl", "expdot[j, 7] += real_v - (S.drvs_dT[i] * dT_v) - (S.drvs_dp[i] * (dp_v + dp_q))", 1),
    ("src/mc_boundary_layer.jl", "expdot[j, 8] += nrr", 1),
    ("src/mc_boundary_layer.jl", "expdot[j, 9] += ncc", 1),
    # src/mc_geometry.jl: the fixed MC layout (u 4, w 5, E_t 6, v 10); derivative plane 3 (d2/dx2) of a named var
    ("src/mc_geometry.jl", "@. expdot[colstart:colend, 10] += (pp_l / r) / rho_tbar", 1),
    ("src/mc_geometry.jl", "@turbo diffdot[colstart:colend, 10] .= @. Kvdiff * v_zz", 1),
    ("src/mc_geometry.jl", "@turbo expdot[colstart:colend, 10] .= @. ADV + FORCING + KDIFF", 4),
    ("src/mc_geometry.jl", "AFTER `mc_v_tendency!` sets expdot[:,10]. No-op on every geometry whose v carries", 1),
    ("src/mc_geometry.jl", "expdot[j,10] += ray * v[i]", 1),
    ("src/mc_geometry.jl", "expdot[j,4] += ray * u[i]", 2),
    ("src/mc_geometry.jl", "expdot[j,5] += ray * w[i]", 2),
    ("src/mc_geometry.jl", "expdot[j,6] += 2.0 * rho_t[i] * ray * ke[i]", 2),
    ("src/mc_geometry.jl", "f_xx = view(grid.physical, colstart:colend, var, 3),", 1),
    ("src/mc_geometry.jl", "history level `L_v = −(1/(ρ̄_t r))∂λ p′` into hacdot[:,10], so v carries the same", 1),
    # src/mc_mynn_bl.jl: the fixed MC layout (u 4, w 5, E_t 6, Q_ss 7, nu_r 8, nu_c 9); the anchor member alpha
    ("src/mc_mynn_bl.jl", "av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]; ai = ws.alpha[4]", 1),
    ("src/mc_mynn_bl.jl", "expdot[j, 4] += du", 1),
    ("src/mc_mynn_bl.jl", "expdot[j, 5] += dw", 1),
    ("src/mc_mynn_bl.jl", "expdot[j, 6] += (qdot_E + dE_w) + dE_mom", 1),
    ("src/mc_mynn_bl.jl", "expdot[j, 7] += real_v - (drvs_dT[i]*dT_v) - (drvs_dp[i]*(dp_v + dp_q))", 1),
    ("src/mc_mynn_bl.jl", "expdot[j, 8] += div_r[i]", 1),
    ("src/mc_mynn_bl.jl", "expdot[j, 9] += div_c[i]", 1),
    # src/microphysics.jl: thermo tuple element 3 = T
    ("src/microphysics.jl", "Tk = [x[3] for x in thermo]     # Temperature in K", 7),
    # src/moist_compressible.jl: the fixed MC layout (u 4, w 5, E_t 6, Q_ss 7, nu_r 8, nu_c 9); derivative/profile planes, ice category 3, a stats row, the rho_t refusal and the derived-rho_t diagnostic text
    ("src/moist_compressible.jl", "\"rho_t = \$(rho_t[i]), rho_c = \$(rho_c[i]), rho_r = \$(rho_r[i])\\n\" *", 1),
    ("src/moist_compressible.jl", "((RfzQ[1] + RfzQ[2]) + RfzQ[3])", 1),
    ("src/moist_compressible.jl", "(-(r3.prdr_pre * (1.0 - r3.qcrimefrac)) - MLQ[3] - RfzQ[3]))", 1),
    ("src/moist_compressible.jl", "(-NRN[3] - MLN[3] - FzN[3]))", 1),
    ("src/moist_compressible.jl", "(dl += affine_defect_change(nu_c[i], vnp1[g, 9] + (nu_c[i] - phys[g, 9, 1]),", 1),
    ("src/moist_compressible.jl", "(dl += refit_recovered_change(nu_c[i], vnp1[g, 9] + (nu_c[i] - phys[g, 9, 1]),", 1),
    ("src/moist_compressible.jl", "(max(-MLN[3], 0.0) + max(-nagg3, 0.0) * i_dt) + nsr3, n3,", 1),
    ("src/moist_compressible.jl", "@turbo diffdot[colstart:colend,4] .= @. Kvdiff * u_zz", 1),
    ("src/moist_compressible.jl", "@turbo diffdot[colstart:colend,5] .= @. Kvdiff * w_zz", 1),
    ("src/moist_compressible.jl", "@turbo diffdot[colstart:colend,6] .= Kvdiff_heat .* stage_zz", 1),
    ("src/moist_compressible.jl", "@turbo expdot[colstart:colend,4] .+= @. pp_x / rho_tbar", 1),
    ("src/moist_compressible.jl", "@turbo expdot[colstart:colend,4] .= @. ADV + FORCING + KDIFF", 1),
    ("src/moist_compressible.jl", "@turbo expdot[colstart:colend,5] .= @. ADV + FORCING + KDIFF", 1),
    ("src/moist_compressible.jl", "@turbo expdot[colstart:colend,6] .+= @. (E_tbar + pbar) * LDIV", 1),
    ("src/moist_compressible.jl", "@turbo expdot[colstart:colend,6] .= @. ADV + FORCING + QDOT_TH + FRIC_KE +", 2),
    ("src/moist_compressible.jl", "@turbo expdot[colstart:colend,7] .= @. ADV + FORCING", 1),
    ("src/moist_compressible.jl", "@turbo expdot[colstart:colend,8] .= @. ADV + FORCING", 1),
    ("src/moist_compressible.jl", "@turbo expdot[colstart:colend,9] .= @. ADV + FORCING", 1),
    ("src/moist_compressible.jl", "_ice_live_span(Q[3], N[3], rho_d, idx)) :", 1),
    ("src/moist_compressible.jl", "abs((MLQ[1] + MLQ[2]) + MLQ[3]) * rhoair * dt)", 1),
    ("src/moist_compressible.jl", "ctr !== :none && (gl += _exchange_gap(vnp1[g, 9] + (nuc[i] - phys[g, 9, 1]), d9[i],", 1),
    ("src/moist_compressible.jl", "d8[i] != 0.0 && (vnp1[g, 8] += d8[i])", 1),
    ("src/moist_compressible.jl", "d9[i] != 0.0 && (vnp1[g, 9] += d9[i])", 1),
    ("src/moist_compressible.jl", "d_col = ((RfzQ[1] + RfzQ[2]) + RfzQ[3]) * iqr", 1),
    ("src/moist_compressible.jl", "d_mlt = -((MLQ[1] + MLQ[2]) + MLQ[3]) * iqr", 1),
    ("src/moist_compressible.jl", "error(\"grid_params.vars declares \\\"rho_t\\\", the prognostic total-density slot that \" *", 1),
    ("src/moist_compressible.jl", "etd_nl[i] != 0.0 && (expdot[colstart + i - 1, 7] += etd_nl[i])", 1),
    ("src/moist_compressible.jl", "expdot[colstart + i - 1, 6] -= E_sed_i_z[i]", 1),
    ("src/moist_compressible.jl", "expdot[colstart + i - 1, 8] += Jr[i] * S.ICE_R[i]", 1),
    ("src/moist_compressible.jl", "expdot[colstart + i - 1, 9] += Jc[i] * S.ICE_C[i]", 1),
    ("src/moist_compressible.jl", "hacdot[colstart:colend,6] .= @. -(E_tbar + pbar) * LDIV", 1),
    ("src/moist_compressible.jl", "haskey(model.grid_params.vars, \"rho_t\") &&", 1),
    ("src/moist_compressible.jl", "impdot[colstart:colend,6] .= @. -((imp_c_e * imp_phi_z) + (imp_c_e_z * phi_n))", 1),
    ("src/moist_compressible.jl", "mc_donor_census!(stx, tid_x, MC_DONOR_I3, max(-MLQ[3], 0.0) - qagg3 * i_dt, q3,", 1),
    ("src/moist_compressible.jl", "min(f_nr * r3.dNfzri, (RfzQ[3] + IfzQ[3]) / ISHMAEL_M_MIN))", 1),
    ("src/moist_compressible.jl", "min_r = minimum(view(st, 3, :)),", 1),
    ("src/moist_compressible.jl", "n0 = expdot[g, 7]", 1),
    ("src/moist_compressible.jl", "nIfz_in = target == 1 ? (FzN[2] + FzN[3]) : (FzN[1] + FzN[3])", 1),
    ("src/moist_compressible.jl", "n_i != 0.0 && (expdot[colstart + i - 1, 7] += n_i)", 1),
    ("src/moist_compressible.jl", "prof[:, 3] .= Ixxtransform(column)", 1),
    ("src/moist_compressible.jl", "qIfz_in = target == 1 ? (IfzQ[2] + IfzQ[3]) : (IfzQ[1] + IfzQ[3])", 1),
    ("src/moist_compressible.jl", "rtr !== :none && (dl += affine_defect_change(nu_r[i], vnp1[g, 8], rtr, rmu))", 1),
    ("src/moist_compressible.jl", "rtr !== :none && (dl += refit_recovered_change(nu_r[i], vnp1[g, 8], rtr, rmu))", 1),
    ("src/moist_compressible.jl", "rtr !== :none && (gl += _exchange_gap(vnp1[g, 8], d8[i], rq, rtr, rmu))", 1),
    ("src/moist_compressible.jl", "vnp1[g, 7] += dq", 1),
    ("src/moist_compressible.jl", "vnp1[g, 7] += qnp1[i]", 1),
    ("src/moist_compressible.jl", "vnp1[g, 7] = qnp1[i]", 1),
    ("src/moist_compressible.jl", "wall_du[r, p_i, s, 3] = d2[r]", 1),
    ("src/moist_compressible.jl", "xwm = -((MLQ[1] + MLQ[2]) + MLQ[3]) * dt / q_ice", 1),
    # src/nesting.jl: gridpoint coordinate column 3 (z); xstar column 3 (exact SI block)
    ("src/nesting.jl", "collar_z = ppts[1:pgp.kDim, 3]     # shared vertical mish", 1),
    ("src/nesting.jl", "xstar[rows, 3] .= view(mtile.var_np1, :, vars[\"p\"])", 1),
    # src/netcdf_output.jl: ice member 3 width; the DERIVED rho_t / rho_tbar output fields
    ("src/netcdf_output.jl", "\"ice_mu_a\" => cfg.imu[3],", 1),
    ("src/netcdf_output.jl", "(\"rho_tbar\", d.rho_tbar_w, \"kg m-3\"),", 1),
    ("src/netcdf_output.jl", "wr(\"rho_t\", d.rho_t, \"kg m-3\",", 1),
    # src/radiation.jl: the fixed MC layout (u 4, w 5, E_t 6, nu_r 8, nu_c 9); RRTMGP ice-column layout
    ("src/radiation.jl", "E_tp = view(phys, colstart:colend, 6, 1)", 1),
    ("src/radiation.jl", "ic[k, 11] = ic[k, 3]", 1),
    ("src/radiation.jl", "ic[k, 3] = recover_total(0.0, itrans, imu_a)", 1),
    ("src/radiation.jl", "ic[k, 3] = recover_total(i1a[k], itrans, imu_a)", 1),
    ("src/radiation.jl", "ic[k, 7] = ic[k, 3]", 1),
    ("src/radiation.jl", "nu_c = view(phys, colstart:colend, 9, 1)", 1),
    ("src/radiation.jl", "nu_r = view(phys, colstart:colend, 8, 1)", 1),
    ("src/radiation.jl", "u = view(phys, colstart:colend, 4, 1)", 1),
    ("src/radiation.jl", "w = view(phys, colstart:colend, 5, 1)", 1),
    # src/reference_state.jl: reference-file column 3
    ("src/reference_state.jl", "xibar[i,1] = parse(Float64,lineparts[3])", 1),
    # src/semiimplicit.jl: the loader docstring naming the pre-Stage-4 layout
    ("src/semiimplicit.jl", "THE PRE-STAGE-4 LAYOUT (a `\"rho_t\"` slot 3, the vapor appended). A `.jld2` checkpoint in it is", 1),
    # src/si_water_slaving.jl: the fixed MC layout (Q_ss 7, nu_r 8, nu_c 9); SI increment plane 3; the :slot refusal text; the anchor member alpha
    ("src/si_water_slaving.jl", "\"rho_t is the derived rho_d + sum_k alpha_k nu_k). The :slot lineage is in git history \" *", 1),
    ("src/si_water_slaving.jl", "av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]; ai = ws.alpha[4]", 3),
    ("src/si_water_slaving.jl", "dc != 0.0 && (vnp1[g, 9] += dc)", 1),
    ("src/si_water_slaving.jl", "dc = lobe_relaxation_cap(dlc[i], vnp1[g, 9] + (nuc[i] - phys[g, 9, 1]))", 1),
    ("src/si_water_slaving.jl", "dr != 0.0 && (vnp1[g, 8] += dr)", 1),
    ("src/si_water_slaving.jl", "dr = lobe_relaxation_cap(dlr[i], vnp1[g, 8])", 1),
    ("src/si_water_slaving.jl", "expdot[g, 7] += dq", 1),
    ("src/si_water_slaving.jl", "hacdot_n[l, s] = sh * (incr[r, 4] - incr[r, 3])", 1),
    ("src/si_water_slaving.jl", "the anchor's stored water increment `incr[:,4] − incr[:,3]` (ρ_t minus ρ_d, already divided by", 1),
    ("src/si_water_slaving.jl", "vnp1[g, 7] += dq", 1),
    # tools/mynn_dump_columns.jl: reference-file column 3; a pre-Stage-4 CSV's rho_t column
    ("tools/mynn_dump_columns.jl", "rdbar[i] = parse(Float64, parts[3]); rvbar[i] = parse(Float64, parts[4]); rcbar[i] = parse(Float64, parts[5])", 1),
    ("tools/mynn_dump_columns.jl", "rt3 = if \"rho_t\" in names(d)", 1),
]

@testset "slot-3 literal audit (src/, benchmarks/, simulations/, tools/)" begin
    root = joinpath(@__DIR__, "..")
    hits = slot3_audit_hits(root)
    allowed = Dict((f, t) => n for (f, t, n) in SLOT3_ALLOWLIST)
    for (k, n) in sort(collect(hits))
        get(allowed, k, 0) == n ||
            println("  slot-3 audit: NOT ALLOWLISTED $(k[1]): $(k[2])  (x$(n))")
    end
    for (k, n) in sort(collect(allowed))
        get(hits, k, 0) == n || println("  slot-3 audit: STALE ALLOWLIST $(k[1]): $(k[2])  (x$(n))")
    end
    @test hits == allowed
    # the legacy-model exemption holds only while those files never reference the MC state
    for f in SLOT3_LEGACY
        @test !occursin(SLOT3_MC_TOKENS, read(joinpath(root, f), String))
    end
    # the patterns are live: every form of a slot-3 (or any literal-slot) state read is caught
    live(s) = any(p -> occursin(p, s), SLOT3_PATTERNS)
    @test live("x = phys[g, 3, 1]")
    @test live("S.rho_t .= view(vnp1, :, 3]")
    @test live("i = slots.rho_t")
    @test live("@turbo diffdot[colstart:colend,3] .= @. K * q_zz")     # no space after the comma
    @test live("rho_tp = view(phys, colstart:colend, 3, 1)")           # view(...)
    @test live("rho_tp = view(phys,:,3,1)")
    @test live("expdot[g,7] += dq")                                    # any literal slot 3..24
    @test live("vnp1[g, 24] = 0.0")
    @test live("G = ph[g, 3, 1]")                                      # the ph/sp aliases
    @test live("q = view(spectral, :, 12)")
    @test !live("x = phys[g, vars[\"u\"], 1]")                          # by name: not a hit
    @test !live("y = expdot[g, 2]")                                    # slots 1, 2 (p, rho_d) are fixed
end
