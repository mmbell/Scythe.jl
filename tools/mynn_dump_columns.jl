#!/usr/bin/env julia
# Write the single-column input files for tools/mynn_fortran_driver/ref_driver.f90
# (MYNN-EDMF port, plan stage S0). Every number is printed with %.17g and read back
# list-directed by the Fortran driver, so the round trip is exact.
#
#   julia --project=. tools/mynn_dump_columns.jl [--tc-dir tc/output/tc_rad_24h_full] [--tc-t 86400.0]
#
# Files written to tools/mynn_fortran_driver/columns/:
#   constants.txt      the 14 host constants bl_mynn_common takes from the dycore, in the
#                      order  cp cpv cliq cice p608 ep_2 grav karman t0c rcp r_d r_v xlf xlv,
#                      taken from Springsteel.Thermodynamics so Fortran and the Julia port
#                      (MYNNConstants) agree bitwise
#   case1_rest.txt     humidified Dunion sounding at rest, no surface fluxes (inertness)
#   case2_o01_sea.txt  Dunion + sheared wind u = 10 tanh(z/150 m) over SST 301.15 K with
#                      Scythe's own bulk surface layer (Komori Cd, Ck = 1e-3, U_min = 2)
#   case3_tc_rmw.txt   the nest-1 column of maximum lowest-level tangential wind from a TC
#                      run's mish output, rebuilt exactly as tc_postprocess.jl rebuilds it
#   case5_highwind.txt  Dunion + u = 50 tanh(z/300 m) over SST 302.65 K: the hurricane-force
#                      surface-layer regime (Komori Cd capped), dx = 3 km
#   case4_convective.txt  a dry-neutral mixed layer capped by a 3 K inversion over a heated
#                      surface (hfx = 200 W/m^2), the DMP_mf activation case
#   <case>.meta        human-readable provenance for each case (not read by Fortran)
#
# Column file format (list-directed, no comments):
#   line 1:  n
#   line 2:  ps ts qsfc ust hfx qfx wspd znt xland dx rmol delt
#   lines 3..n+2:  z dz u v w T th exner p rho sqv sqc sqi
# Layers are Scythe's mish points; MYNN's layer thickness dz comes from faces at the
# midpoints between mish points plus z = 0 and z = z_top (the radiation convention,
# src/radiation.jl), so sum(dz) = z_top and zw(k) = sum(dz(1:k-1)) in the driver.
# `sqv` is SPECIFIC humidity rho_v/rho_t and `rho` the MOIST density rho_t, the
# conventions of mynnedmf_wrapper.F90; `ts` is T_sfc/exner(1) (the wrapper's "theta").
#
# case3_tc_rmw reads `<tc-dir>/nest1/<tc-t>_physical.csv`, so --tc-dir must point at a
# TC run made with `--csv` (options[:output_formats] including :csv; the default
# [:netcdf] alone writes no CSV). Without one, case 3 is skipped -- see the "case 3
# SKIPPED" message below -- and the other cases still write normally.

using Printf
using Springsteel
using Scythe
using CSV, DataFrames

const OUT = joinpath(@__DIR__, "mynn_fortran_driver", "columns")
mkpath(OUT)
const T = Springsteel.Thermodynamics
const DUNION = joinpath(@__DIR__, "..", "benchmarks", "reference_data", "o01_rainfall",
                        "dunion_MT_hum90.ref")
const DELT = 20.0
const KARMAN = 0.4

tc_dir = "tc/output/tc_rad_24h_full"
tc_t = "86400.0"
let i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--tc-dir"; global tc_dir = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--tc-t"; global tc_t = ARGS[i+1]; i += 2
        else; error("unknown argument $(ARGS[i])"); end
    end
end

g17(x) = @sprintf("%.17g", x)

# ── constants ────────────────────────────────────────────────────────────────
let
    vals = (T.Cpd, T.Cpv, T.Cl, T.Ci, T.Rv / T.Rd - 1.0, T.Rd / T.Rv, T.gravity, KARMAN,
            T.T_0, T.Rd / T.Cpd, T.Rd, T.Rv, T.L_f0, T.L_v0)
    open(joinpath(OUT, "constants.txt"), "w") do f
        println(f, join(g17.(vals), " "))
    end
end

# ── faces at midpoints (radiation convention) ─────────────────────────────────
function layer_dz(z, ztop)
    n = length(z)
    zf = zeros(n + 1)
    for k in 2:n
        zf[k] = 0.5 * (z[k-1] + z[k])
    end
    zf[n+1] = ztop
    return diff(zf)
end

function write_case(name, z, dz, u, v, w, Tk, th, ex, p, rho, sqv, sqc, sqi, sc, meta)
    n = length(z)
    open(joinpath(OUT, name * ".txt"), "w") do f
        println(f, n)
        println(f, join(g17.((sc.ps, sc.ts, sc.qsfc, sc.ust, sc.hfx, sc.qfx, sc.wspd, sc.znt,
                              sc.xland, sc.dx, sc.rmol, DELT)), " "))
        for k in 1:n
            println(f, join(g17.((z[k], dz[k], u[k], v[k], w[k], Tk[k], th[k], ex[k], p[k],
                                  rho[k], sqv[k], sqc[k], sqi[k])), " "))
        end
    end
    open(joinpath(OUT, name * ".meta"), "w") do f
        println(f, meta)
        println(f, "n = $n, z1 = $(z[1]) m, ztop = $(z[end]) m")
        for (k, v) in pairs(sc)
            println(f, "$k = $(g17(v))")
        end
    end
    println("wrote $name (n = $n)")
end

# Scythe's bulk surface layer (mc_boundary_layer.jl:182-205) on the lowest mish level
function bulk_surface(u1, v1, U_min, Ck, SST, T1, rho_d1, rho_v1, p1_Pa, z1, ex1, rho1)
    U1 = max(sqrt(u1 * u1 + v1 * v1), U_min)
    Cd = Scythe.komori_cd(U1)
    ust = sqrt(Cd) * U1
    hfx = rho_d1 * T.Cpd * Ck * U1 * (SST - T1)
    qfx = Ck * U1 * (T.rho_v_sat(SST, p1_Pa / 100.0) - rho_v1)
    znt = Cd > 0.0 ? z1 * exp(-KARMAN / sqrt(Cd)) : 1.0e-4
    ps = p1_Pa + rho1 * T.gravity * z1          # hydrostatic extrapolation to z = 0
    qsfc = T.rho_v_sat(SST, ps / 100.0) / rho1
    # the wrapper's fallback 1/L when no surface-layer scheme supplies it (hfx >= 0 branch;
    # the stable branch needs a bulk Richardson number the wrapper gets from elsewhere)
    return (; U1, Cd, ust, hfx, qfx, znt, ps, qsfc)
end

# ── Dunion reference on a 50-cell (500 m) RiRk mish, kDim = 150 ──────────────
function dunion_column()
    vars = Scythe.MC_VARS
    bc = Dict(v => NeumannBC() for v in vars)
    gp = Scythe.compute_derived_params(GridParameters(; geometry = "RiRk",
        iMin = 0.0, iMax = 10.0e3, num_cells_i = 2,
        kMin = 0.0, kMax = 25.0e3, num_cells_k = 50, mubar = 3,
        BCL = bc, BCR = bc, BCB = bc, BCT = bc,
        vars = Dict(v => i for (i, v) in enumerate(vars))))
    patch = createGrid(gp)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = gp.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, gp)
    ref = Springsteel.calculate_pressure_reference_state(DUNION, z, column)
    p = Springsteel.ref_pressure(ref)[:, 1]
    rho_d = Springsteel.ref_rho_d(ref)[:, 1]
    rho_v = Springsteel.ref_rho_v(ref)[:, 1]
    return z, p, rho_d, rho_v
end

function thermo_from_state(p, rho_d, rho_v, rho_c)
    Tk = p ./ (rho_d .* T.Rd .+ rho_v .* T.Rv)
    ex = (p ./ 1.0e5) .^ (T.Rd / T.Cpd)
    th = Tk ./ ex
    rho = rho_d .+ rho_v .+ rho_c
    sqv = rho_v ./ rho
    sqc = rho_c ./ rho
    return Tk, ex, th, rho, sqv, sqc
end

z, p, rho_d, rho_v = dunion_column()
n = length(z)
dz = layer_dz(z, 25.0e3)
Tk, ex, th, rho, sqv, sqc = thermo_from_state(p, rho_d, rho_v, zeros(n))
zero_n = zeros(n)

# case 1: rest, U_min = 0 so Cd*U1 = 0 exactly -> ust = hfx = qfx = 0; SST = T1 for good measure
let
    s = bulk_surface(0.0, 0.0, 0.0, 1.0e-3, Tk[1], Tk[1], rho_d[1], rho_v[1], p[1], z[1], ex[1], rho[1])
    sc = (; ps = s.ps, ts = Tk[1] / ex[1], qsfc = s.qsfc, ust = 0.0, hfx = 0.0, qfx = 0.0,
          wspd = 0.1, znt = 1.0e-4, xland = 2.0, dx = 2000.0, rmol = 0.0, delt = DELT)
    write_case("case1_rest", z, dz, zero_n, zero_n, zero_n, Tk, th, ex, p, rho, sqv, zero_n, zero_n, sc,
        "case 1: humidified Dunion sounding (benchmarks/reference_data/o01_rainfall/dunion_MT_hum90.ref)\n" *
        "at rest on the 50-cell RiRk mish; u = v = w = 0, ust = hfx = qfx = 0 (U_min = 0), wspd floored\n" *
        "at 0.1 m/s only so the driver's ust^2/wspd is 0/0.1 rather than 0/0. Inertness reference.")
end

# case 2: sheared wind over a 301.15 K sea, Scythe's bulk fluxes
let
    u = 10.0 .* tanh.(z ./ 150.0)
    SST = 301.15
    s = bulk_surface(u[1], 0.0, 2.0, 1.0e-3, SST, Tk[1], rho_d[1], rho_v[1], p[1], z[1], ex[1], rho[1])
    rmol0 = s.hfx >= 0.0 ? -s.hfx / (200.0 * dz[1] * 0.5) : 0.0
    sc = (; ps = s.ps, ts = SST / ex[1], qsfc = s.qsfc, ust = s.ust, hfx = s.hfx, qfx = s.qfx,
          wspd = s.U1, znt = s.znt, xland = 2.0, dx = 2000.0, rmol = rmol0, delt = DELT)
    write_case("case2_o01_sea", z, dz, u, zero_n, zero_n, Tk, th, ex, p, rho, sqv, zero_n, zero_n, sc,
        "case 2: Dunion sounding + u = 10 tanh(z/150 m), v = w = 0, over SST = 301.15 K.\n" *
        "Surface layer = Scythe bulk (mc_boundary_layer.jl): U1 = max(|u1|, 2), Cd = komori_cd(U1),\n" *
        "ust = sqrt(Cd) U1, hfx = rho_d1 Cpd Ck U1 (SST - T1), qfx = Ck U1 (rho_vs(SST,p1) - rho_v1),\n" *
        "znt = z1 exp(-kappa/sqrt(Cd)) (neutral log law consistent with Cd), rmol = wrapper fallback.\n" *
        "Cd = $(g17(s.Cd)), U1 = $(g17(s.U1))")
end

# case 4: convective boundary layer (synthetic, hydrostatic)
let
    theta = [zz <= 1000.0 ? 300.0 : 303.0 + 3.0e-3 * (zz - 1000.0) for zz in z]
    qv = [zz <= 1000.0 ? 0.012 : max(0.012 * exp(-(zz - 1000.0) / 1500.0), 5.0e-5) for zz in z]
    ps = 1.0e5
    # hydrostatic march in ln p with the layer-mean virtual temperature; T = theta*exner
    pp = zeros(n); Tk4 = zeros(n)
    Tv(th_, ex_, q) = th_ * ex_ * (1.0 + 0.608 * q)
    exs = (ps / 1.0e5)^(T.Rd / T.Cpd)
    Tv_prev = Tv(theta[1], exs, qv[1]); p_prev = ps; z_prev = 0.0
    for k in 1:n
        # two fixed-point passes on the layer-mean Tv
        pk = p_prev * exp(-T.gravity * (z[k] - z_prev) / (T.Rd * Tv_prev))
        for _ in 1:2
            exk = (pk / 1.0e5)^(T.Rd / T.Cpd)
            Tvk = Tv(theta[k], exk, qv[k])
            pk = p_prev * exp(-T.gravity * (z[k] - z_prev) / (T.Rd * 0.5 * (Tv_prev + Tvk)))
        end
        pp[k] = pk
        exk = (pk / 1.0e5)^(T.Rd / T.Cpd)
        Tk4[k] = theta[k] * exk
        Tv_prev = Tv(theta[k], exk, qv[k]); p_prev = pk; z_prev = z[k]
    end
    ex4 = (pp ./ 1.0e5) .^ (T.Rd / T.Cpd)
    rho4 = pp ./ (T.Rd .* Tk4 .* (1.0 .+ 0.608 .* qv))          # the wrapper's moist density
    sqv4 = qv ./ (1.0 .+ qv)                                     # mixing ratio -> specific
    u4 = fill(5.0, n)
    Cd = 1.2e-3; U1 = 5.0
    sc = (; ps = ps, ts = 302.0 / exs, qsfc = 0.02, ust = sqrt(Cd) * U1, hfx = 200.0, qfx = 5.0e-5,
          wspd = U1, znt = z[1] * exp(-KARMAN / sqrt(Cd)), xland = 2.0, dx = 2000.0,
          rmol = -200.0 / (200.0 * dz[1] * 0.5), delt = DELT)
    write_case("case4_convective", z, dz, u4, zero_n, zero_n, Tk4, theta, ex4, pp, rho4, sqv4, zero_n, zero_n, sc,
        "case 4: synthetic convective BL. theta = 300 K to 1 km, +3 K jump then +3 K/km; q_v = 12 g/kg\n" *
        "to 1 km then exp decay (1.5 km scale, floor 0.05 g/kg); hydrostatic from ps = 1000 hPa;\n" *
        "u = 5 m/s, v = w = 0; hfx = 200 W/m^2, qfx = 5e-5 kg/m^2/s, Cd = 1.2e-3 -> ust = 0.173 m/s;\n" *
        "T_sfc = 302 K. Built to activate DMP_mf (fltv > 0.002 and superadiabatic near the surface).")
end

# case 5: high-wind TC regime, synthetic: Dunion + u = 50 tanh(z/300 m) over the TC SST
let
    u = 50.0 .* tanh.(z ./ 300.0)
    SST = 302.65
    s = bulk_surface(u[1], 0.0, 2.0, 1.0e-3, SST, Tk[1], rho_d[1], rho_v[1], p[1], z[1], ex[1], rho[1])
    rmol0 = s.hfx >= 0.0 ? -s.hfx / (200.0 * dz[1] * 0.5) : 0.0
    sc = (; ps = s.ps, ts = SST / ex[1], qsfc = s.qsfc, ust = s.ust, hfx = s.hfx, qfx = s.qfx,
          wspd = s.U1, znt = s.znt, xland = 2.0, dx = 3000.0, rmol = rmol0, delt = DELT)
    write_case("case5_highwind", z, dz, u, zero_n, zero_n, Tk, th, ex, p, rho, sqv, zero_n, zero_n, sc,
        "case 5: Dunion sounding + u = 50 tanh(z/300 m) (hurricane-force lowest-level wind), v = w = 0,\n" *
        "over SST = 302.65 K (the TC's SST_K); Scythe bulk surface layer with Komori Cd in its capped\n" *
        "high-wind regime (2.55e-3), Ck = 1e-3, U_min = 2; dx = 3 km (TC nest-1 production cells).\n" *
        "The surface-layer / K-limit regime the port must survive. Cd = $(g17(s.Cd)), U1 = $(g17(s.U1)), hfx = $(g17(s.hfx)), qfx = $(g17(s.qfx))")
end

# case 3: TC nest-1 RMW column from the mish CSV + tc_exact.ref, tc_postprocess.jl conventions
let
    csv = joinpath(tc_dir, "nest1", tc_t * "_physical.csv")
    reffile = joinpath(tc_dir, "tc_exact.ref")
    if !(isfile(csv) && isfile(reffile))
        println("case 3 SKIPPED: $csv or $reffile missing")
    else
        df = CSV.read(csv, DataFrame)
        reflines = readlines(reffile)
        kDim = length(reflines)
        zm = zeros(kDim); pbar = zeros(kDim); rdbar = zeros(kDim); rvbar = zeros(kDim); rcbar = zeros(kDim)
        for (i, l) in enumerate(reflines)
            parts = split(l)
            zm[i] = parse(Float64, parts[1]); pbar[i] = parse(Float64, parts[2])
            rdbar[i] = parse(Float64, parts[3]); rvbar[i] = parse(Float64, parts[4]); rcbar[i] = parse(Float64, parts[5])
        end
        ncols = div(nrow(df), kDim)
        maximum(abs.(df.z[1:kDim] .- zm)) < 1.0e-9 || error("CSV mish z differs from tc_exact.ref z")
        rtbar = rdbar .+ rvbar .+ rcbar
        Tbar = pbar ./ (rdbar .* T.Rd .+ rvbar .* T.Rv)
        E_tbar = rdbar .* T.internal_energy_bf02.(Tbar, rvbar ./ rdbar, rcbar ./ rdbar) .+
                 rtbar .* T.gravity .* zm
        cmu = 1.0e-7; rmu = 1.0e-7          # detect_water_transforms defaults (log prints none)
        # lowest-level tangential wind picks the RMW column
        vlow = [df.v[(c - 1) * kDim + 1] for c in 1:ncols]
        c = argmax(vlow)
        rows = ((c - 1) * kDim + 1):(c * kDim)
        d = df[rows, :]
        p3 = d.p .+ pbar; rd3 = d.rho_d .+ rdbar; rt3 = d.rho_t .+ rtbar; E3 = d.E_t .+ E_tbar
        rc3 = Scythe.recover_rho_c.(d.nu_c, rcbar, :bhyp, cmu)
        rr3 = Scythe.recover_rho_r.(d.nu_r, :bhyp, rmu)
        rv3 = d.rho_v .+ (rtbar .- rdbar .- rcbar)
        ke = 0.5 .* (d.u .^ 2 .+ d.v .^ 2 .+ d.w .^ 2)          # cylinders carry v in ke (mc_ke!)
        M = p3 .+ E3 .- rt3 .* (ke .+ T.gravity .* zm)
        Tk3 = Scythe.retrieve_temperature.(M, rd3, rt3, rc3 .+ rr3)
        ex3 = (p3 ./ 1.0e5) .^ (T.Rd / T.Cpd)
        th3 = Tk3 ./ ex3
        rho3 = rt3
        sqv3 = rv3 ./ rho3; sqc3 = rc3 ./ rho3
        dz3 = layer_dz(zm, 25.0e3)
        SST = 302.65
        s = bulk_surface(d.u[1], d.v[1], 2.0, 1.0e-3, SST, Tk3[1], rd3[1], rv3[1], p3[1], zm[1], ex3[1], rho3[1])
        rmol0 = s.hfx >= 0.0 ? -s.hfx / (200.0 * dz3[1] * 0.5) : 0.0
        sc = (; ps = s.ps, ts = SST / ex3[1], qsfc = s.qsfc, ust = s.ust, hfx = s.hfx, qfx = s.qfx,
              wspd = s.U1, znt = s.znt, xland = 2.0, dx = 10000.0, rmol = rmol0, delt = DELT)
        write_case("case3_tc_rmw", zm, dz3, d.u, d.v, d.w, Tk3, th3, ex3, p3, rho3, sqv3, sqc3, zeros(kDim), sc,
            "case 3: $csv column $c of $ncols (r = $(d.r[1]) m, max lowest-level v = $(vlow[c]) m/s),\n" *
            "rebuilt with $reffile (E_tbar and T as tc_postprocess.jl; bhyp cloud/rain, mu = 1e-7;\n" *
            "ke includes v). SST = $SST K (tc_params SST_K), Komori Cd, Ck = 1e-3, U_min = 2; dx = 10 km\n" *
            "(nest-1 coarse cells). Cd = $(g17(s.Cd)), U1 = $(g17(s.U1)), hfx = $(g17(s.hfx)), qfx = $(g17(s.qfx))")
    end
end
