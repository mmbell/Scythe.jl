# test/test_slot3_audit.jl -- no source line indexes the moist-compressible state by the literal
# slot 3 (Stage 4 of the rho_t retirement deleted slot 3, the prognostic total density "rho_t";
# the vapor is slot 3 now and every reader resolves it by name: `mc_slots(model).rho_v`,
# `mc_slot(vars, "rho_v")`).
#
# Every non-comment line of src/*.jl matching one of
#     `, 3]`   `[3]`   `[..., 3, ...]`   `slots.rho_t`   `"rho_t`
# must be in the ALLOWLIST below (exact stripped text and count, with the reason: a literal 3
# that indexes something other than the model state, or a "rho_t" that names the DERIVED total
# or the old layout), and every allowlisted line must still exist (the list cannot rot).
# src/checkpoint_conversion.jl is allowlisted whole: its subject is the pre-Stage-4 layout.
#
# Standalone:
#   julia --project=. -e 'using Test; include("test/test_slot3_audit.jl")'

using Test

const SLOT3_PATTERNS = [r", 3\]", r"\[3\]", r"slots\.rho_t", r"\"rho_t", r"\[[^\[\]]*, 3, [^\[\]]*\]"]
const SLOT3_WHOLE_FILE = Set(["checkpoint_conversion.jl"])
const SLOT3_ALLOWLIST = [
    # exact_si.jl: column 3 of exact_si's own unknown block (u*, w*, v*/p*), filled by name
    ("exact_si.jl", "PR[i, k] = xstar[r, 3]", 1),
    ("exact_si.jl", "hfields[r, 3] = dp", 1),
    ("exact_si.jl", "xstar[rows, 3] .= view(mtile.var_np1, :, vars[\"v\"])", 1),
    ("exact_si.jl", "xstar[rows, 3] .= view(mtile.var_np1, :, vars[\"p\"])", 1),
    # horizontal_si.jl: plane 3 of the horizontal SI level increments (its own layout)
    ("horizontal_si.jl", "dlev[i, k, 3] = -cd * dxu", 1),
    ("horizontal_si.jl", "x = hsd.wshare[k, n] * (dlev[i, k, 4] - dlev[i, k, 3])", 1),
    # idealized.jl: thermo tuple element 3 = T; the native-init working grid's ONE variable, named rho_t
    ("idealized.jl", "Tk = [x[3] for x in thermo]", 1),
    ("idealized.jl", "THE WORKING TOTAL DENSITY of [`balanced_vortex_native!`](@ref): a one-variable grid (`\"rho_t\"`)", 1),
    ("idealized.jl", "one(d) = (haskey(d, \"rho_d\") ? Dict(\"rho_t\" => d[\"rho_d\"]) :", 1),
    ("idealized.jl", "haskey(d, \"default\") ? Dict(\"rho_t\" => d[\"default\"]) : empty(d))", 1),
    ("idealized.jl", "haskey(gp.l_q, \"rho_d\" * sfx) && (lq[\"rho_t\" * sfx] = gp.l_q[\"rho_d\" * sfx])", 1),
    ("idealized.jl", "keep = Dict{Symbol,Any}(:vars => Dict(\"rho_t\" => 1),", 1),
    ("idealized.jl", "Dict(\"rho_t\" => gp.max_wavenumber[\"rho_d\"]) :", 1),
    ("idealized.jl", "Dict(\"rho_t\" => gp.spline_filter[\"rho_d\"]) :", 1),
    # ishmael.jl: ISHMAEL process/category tables
    ("ishmael.jl", "qi_qr_nrm, qi_qr_nrd, qi_qr_nrn = procr[1], procr[2], procr[3]", 1),
    ("ishmael.jl", "ip_34 = ISHMAEL_IPAIR[3, 4]; ip_43 = ISHMAEL_IPAIR[4, 3]", 1),
    ("ishmael.jl", "ip_33 = ISHMAEL_IPAIR[3, 3]; ip_44 = ISHMAEL_IPAIR[4, 4]", 1),
    # ishmael_tables.jl: quadrature abscissae
    ("ishmael_tables.jl", "(x[3] > xup) && error(\"ishmael_avint: less than 3 function values between integration limits\")", 1),
    # mc_boundary_layer.jl: the anchor member set's alpha (member 3 = rain)
    ("mc_boundary_layer.jl", "av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]", 1),
    # mc_mynn_bl.jl: the anchor member set's alpha (member 3 = rain)
    ("mc_mynn_bl.jl", "av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]; ai = ws.alpha[4]", 1),
    # microphysics.jl: thermo tuple element 3 = T
    ("microphysics.jl", "Tk = [x[3] for x in thermo]     # Temperature in K", 7),
    # moist_compressible.jl: derivative/profile planes, ice category 3, the rho_t refusal and the derived-rho_t diagnostic text
    ("moist_compressible.jl", "prof[:, 3] .= Ixxtransform(column)", 1),
    ("moist_compressible.jl", "wall_du[r, p_i, s, 3] = d2[r]", 1),
    ("moist_compressible.jl", "haskey(model.grid_params.vars, \"rho_t\") &&", 1),
    ("moist_compressible.jl", "error(\"grid_params.vars declares \\\"rho_t\\\", the prognostic total-density slot that \" *", 1),
    ("moist_compressible.jl", "_ice_live_span(Q[3], N[3], rho_d, idx)) :", 1),
    ("moist_compressible.jl", "min(f_nr * r3.dNfzri, (RfzQ[3] + IfzQ[3]) / ISHMAEL_M_MIN))", 1),
    ("moist_compressible.jl", "qIfz_in = target == 1 ? (IfzQ[2] + IfzQ[3]) : (IfzQ[1] + IfzQ[3])", 1),
    ("moist_compressible.jl", "nIfz_in = target == 1 ? (FzN[2] + FzN[3]) : (FzN[1] + FzN[3])", 1),
    ("moist_compressible.jl", "((RfzQ[1] + RfzQ[2]) + RfzQ[3])", 1),
    ("moist_compressible.jl", "(-(r3.prdr_pre * (1.0 - r3.qcrimefrac)) - MLQ[3] - RfzQ[3]))", 1),
    ("moist_compressible.jl", "(-NRN[3] - MLN[3] - FzN[3]))", 1),
    ("moist_compressible.jl", "mc_donor_census!(stx, tid_x, MC_DONOR_I3, max(-MLQ[3], 0.0) - qagg3 * i_dt, q3,", 1),
    ("moist_compressible.jl", "(max(-MLN[3], 0.0) + max(-nagg3, 0.0) * i_dt) + nsr3, n3,", 1),
    ("moist_compressible.jl", "d_col = ((RfzQ[1] + RfzQ[2]) + RfzQ[3]) * iqr", 1),
    ("moist_compressible.jl", "d_mlt = -((MLQ[1] + MLQ[2]) + MLQ[3]) * iqr", 1),
    ("moist_compressible.jl", "xwm = -((MLQ[1] + MLQ[2]) + MLQ[3]) * dt / q_ice", 1),
    ("moist_compressible.jl", "abs((MLQ[1] + MLQ[2]) + MLQ[3]) * rhoair * dt)", 1),
    ("moist_compressible.jl", "\"rho_t = \$(rho_t[i]), rho_c = \$(rho_c[i]), rho_r = \$(rho_r[i])\\n\" *", 1),
    # nesting.jl: gridpoint coordinate column 3 (z); xstar column 3 (exact SI block)
    ("nesting.jl", "collar_z = ppts[1:pgp.kDim, 3]     # shared vertical mish", 1),
    ("nesting.jl", "xstar[rows, 3] .= view(mtile.var_np1, :, vars[\"p\"])", 1),
    # netcdf_output.jl: ice member 3 width; the DERIVED rho_t / rho_tbar output fields
    ("netcdf_output.jl", "\"ice_mu_a\" => cfg.imu[3],", 1),
    ("netcdf_output.jl", "wr(\"rho_t\", d.rho_t, \"kg m-3\",", 1),
    ("netcdf_output.jl", "(\"rho_tbar\", d.rho_tbar_w, \"kg m-3\"),", 1),
    # primitive_equations.jl: thermo tuple element 3 = T
    ("primitive_equations.jl", "Tk = [x[3] for x in thermo]     # Temperature in K", 6),
    # radiation.jl: RRTMGP ice-column layout
    ("radiation.jl", "ic[k, 3] = recover_total(0.0, itrans, imu_a)", 1),
    ("radiation.jl", "ic[k, 7] = ic[k, 3]", 1),
    ("radiation.jl", "ic[k, 11] = ic[k, 3]", 1),
    ("radiation.jl", "ic[k, 3] = recover_total(i1a[k], itrans, imu_a)", 1),
    # reference_state.jl: reference-file column 3
    ("reference_state.jl", "xibar[i,1] = parse(Float64,lineparts[3])", 1),
    # semiimplicit.jl: the loader docstring naming the pre-Stage-4 layout
    ("semiimplicit.jl", "THE PRE-STAGE-4 LAYOUT (a `\"rho_t\"` slot 3, the vapor appended). A `.jld2` checkpoint in it is", 1),
    # si_water_slaving.jl: SI increment plane 3; the :slot refusal text; the anchor member alpha
    ("si_water_slaving.jl", "hacdot_n[l, s] = sh * (incr[r, 4] - incr[r, 3])", 1),
    ("si_water_slaving.jl", "\"rho_t is the derived rho_d + sum_k alpha_k nu_k). The :slot lineage is in git history \" *", 1),
    ("si_water_slaving.jl", "av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]; ai = ws.alpha[4]", 3),
    # testModels.jl: thermo tuple element 3 = T
    ("testModels.jl", "Tk = [x[3] for x in thermo]     # Temperature in K", 4),
]

@testset "slot-3 literal audit (src/)" begin
    src = joinpath(@__DIR__, "..", "src")
    hits = Dict{Tuple{String,String},Int}()
    for f in sort(readdir(src))
        endswith(f, ".jl") || continue
        f in SLOT3_WHOLE_FILE && continue
        for l in eachline(joinpath(src, f))
            t = strip(l)
            startswith(t, "#") && continue
            any(p -> occursin(p, t), SLOT3_PATTERNS) || continue
            key = (f, String(t))
            hits[key] = get(hits, key, 0) + 1
        end
    end
    allowed = Dict((f, t) => n for (f, t, n) in SLOT3_ALLOWLIST)
    for (k, n) in sort(collect(hits))
        get(allowed, k, 0) == n ||
            println("  slot-3 audit: NOT ALLOWLISTED $(k[1]): $(k[2])  (x$(n))")
    end
    for (k, n) in sort(collect(allowed))
        get(hits, k, 0) == n || println("  slot-3 audit: STALE ALLOWLIST $(k[1]): $(k[2])  (x$(n))")
    end
    @test hits == allowed
    # the patterns are live: a slot-3 read is caught
    @test any(p -> occursin(p, "x = phys[g, 3, 1]"), SLOT3_PATTERNS)
    @test any(p -> occursin(p, "S.rho_t .= view(vnp1, :, 3]"), SLOT3_PATTERNS)
    @test any(p -> occursin(p, "i = slots.rho_t"), SLOT3_PATTERNS)
end
