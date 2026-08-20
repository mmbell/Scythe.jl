# Published benchmark target values and tolerances.
#
# Format: "case" => Dict("diagnostic" => (target, atol_full, atol_quick, source))
#
# Full-mode tolerances are wider than the inter-reference-model spread reported
# in the papers (Scythe is a spectral-transform model; the references are finite
# difference codes) and are the paper-grade validation criteria. Quick-mode
# tolerances are routine-regression guards: they span the measured coarse-grid
# behavior of both equation-set stages (tuned 2026-06 from converged quick
# runs), so a quick failure means "the solution moved", not "differs from the
# paper". The primitive equation stage (--stage pe) uses the quick tolerances.
# Tight version-to-version regression is enforced separately by the committed
# reference comparison at rel 1e-6.
#
# Sources:
# - Straka et al. (1993), Int. J. Numer. Methods Fluids 17, 1-22.
#   Table IV: 200-m statistics; REFC25 row is the 25-m compressible reference
#   sampled at 200 m (front location 15537.44 m, theta'_min -9.77375 K).
#   Table V: 25-m extrema (REFC: u 36.46/-15.19, w 12.93/-15.95 m/s);
#   REFC/REFS/REFQ spread is a few percent.
# - Bryan & Fritsch (2002), Mon. Wea. Rev. 130, 2917-2928.
#   Fig. 1 (dry, 100 m): theta' max 2.07178 / min -0.144409 K,
#   w max 14.5396 / min -8.58069 m/s.
#   Fig. 3 (moist, theta_e=320 K, r_t=0.020): theta_e' max 4.09521 /
#   min -0.305695 K, w max 15.713 / min -9.92698 m/s.

const BENCHMARK_EXPECTED = Dict{String,Dict{String,Tuple{Float64,Float64,Float64,String}}}(
    "straka93" => Dict(
        "min_theta_p"    => (-9.77,    0.3, 1.5, "Straka93 Table V REFC (25 m)"),
        "max_theta_p"    => (0.0,      0.5, 1.0, "Straka93 Table V REFC"),
        "max_u"          => (36.46,    2.0, 6.0, "Straka93 Table V REFC"),
        "min_u"          => (-15.19,   2.0, 6.0, "Straka93 Table V REFC"),
        "max_w"          => (12.93,    1.5, 3.0, "Straka93 Table V REFC"),
        "min_w"          => (-15.95,   1.5, 3.0, "Straka93 Table V REFC"),
        "front_location" => (15537.44, 250.0, 750.0, "Straka93 Table IV REFC25"),
    ),
    "bf02_dry" => Dict(
        "max_theta_p" => (2.07178,   0.15, 0.5, "BF02 Fig. 1a (100 m)"),
        "min_theta_p" => (-0.144409, 0.10, 0.3, "BF02 Fig. 1a"),
        "max_w"       => (14.5396,   0.75, 5.5, "BF02 Fig. 1b"),
        "min_w"       => (-8.58069,  0.75, 2.0, "BF02 Fig. 1b"),
    ),
    "bf02_moist" => Dict(
        "max_theta_e_p" => (4.09521,   0.3, 2.5, "BF02 Fig. 3a (100 m)"),
        "min_theta_e_p" => (-0.305695, 0.2, 1.5, "BF02 Fig. 3a"),
        "max_w"         => (15.713,    1.0, 5.5, "BF02 Fig. 3b"),
        "min_w"         => (-9.92698,  1.0, 3.0, "BF02 Fig. 3b"),
    ),
    # O01 warm rain: sanity WINDOWS, RE-SEEDED 2026-07-29 from the two runs that
    # define the shipped configuration of that date — prognostic rho_c (the phantom-
    # cloud fix) and the regime-blended vapor retrieval as the default
    # (reference/HANDOFF_VAPOR_RETRIEVAL.md). The previous seeds (2026-07-13,
    # re-validated 2026-07-15: peak 55, max rho_r 8.5, max w 9.0, onset 24, accum
    # 1.05) predate both and had been FAILING on accum_rainfall_mm and max_rho_r_gm3
    # ever since prognostic rho_c; the storm rains more and holds more rain now.
    #
    # The measurements the windows are built from, both --stage mc --grid rirk:
    #   quick (2 km / 250 m): peak 59.94, onset 23 min, max w 9.87,
    #                         max rho_r 13.12, accum 2.500
    #   full  (500 m / 83 m): peak 121.88, onset 31 min, max w 17.29,
    #                         max rho_r 14.76, accum 3.200
    # The mc stage always takes the QUICK tolerance (load_targets), so each window
    # must span BOTH resolutions, and the resolution sensitivity is the same one
    # O01 reports in his own Dx = 1 vs 2 km comparison (Fig. 6: ~125 vs ~75
    # g m^-2 s^-1) — which the pair above now brackets almost exactly. The equation
    # set, tau_r microphysics closure (no separate Qevap), humidified-Dunion
    # sounding and single non-nested grid all differ from Ooyama (2001), so only
    # comparable magnitude and timing are enforced here; tight version-to-version
    # regression comes from the committed reference CSVs.
    "o01_rainfall" => Dict(
        "peak_rain_rate_gm2s" => (90.0, 35.0, 40.0, "re-seeded 2026-07-29; O01 Fig. 6 range 75-125"),
        "max_rho_r_gm3"       => (14.0, 4.0,  5.0,  "re-seeded 2026-07-29; O01 Figs. 4-6, 11 show ~4"),
        "max_w"               => (13.5, 4.0,  5.5,  "re-seeded 2026-07-29 (quick 9.9, full 17.3)"),
        "rain_onset_min"      => (27.0, 6.0,  8.0,  "re-seeded 2026-07-29; O01 Fig. 6 ~25-40 min"),
        "accum_rainfall_mm"   => (2.85, 1.0,  1.2,  "re-seeded 2026-07-29 (domain-mean rain-out)"),
    ),
    # Axisymmetric-cylinder O01 at large mean radius (r in [1000, 1150] km, f = 0):
    # the metric terms are O(dx/r) ~ 0.2%, so the rain windows WERE the o01_rainfall
    # quick values; the tangential wind must stay identically zero (no source with
    # f = 0 and v0 = 0), which validates the 9-var cylindrical machinery.
    #
    # DELIBERATELY NOT RE-SEEDED WITH o01_rainfall ON 2026-07-29, and the mirror above
    # is now broken. A quick run of this case under the same shipped configuration as
    # the o01_rainfall re-seed measures peak 43.24, onset 22 min, max w 4.63,
    # max rho_r 6.84, accum 3.791 — against o01_rainfall's quick 59.94 / 23 / 9.87 /
    # 13.12 / 2.500. The two cases tracked each other to ~2 % as recently as
    # 2026-07-16 (1.377 / 5.79 / 9.93 / 27.7 / 24 there against 1.37 / 5.8 / 10.1 /
    # 27.9 / 24 here), so this is a DIVERGENCE that appeared on the prognostic-rho_c
    # branch, not a metric-term effect, and it is a measurement to explain rather than
    # a window to move: the cylindrical storm now has less than half the peak updraft
    # and rains 50 % more. This case's committed field reference
    # (o01_axisym/quick_mc_rirk_final.csv, 2026-07-16) is stale for the same reason —
    # it has no rho_c column at all, so compare_reference throws on it. Both are left
    # for the axisym owner; the values below are the untouched 2026-07-13 seeds.
    # Ice arm (SCYTHE_O01_ICE=1 + SCYTHE_O01_RAIN_MOMENTS=2): arm-qualified key, looked up
    # via load_targets(name, opts; arm="ice") -> "o01_rainfall_ice" (benchmarks/common/
    # harness.jl). EMPTY on purpose (seeded 2026-08-19, values to follow at S9 close-out):
    # the ice port (S8, 1357f81) only runs to the glaciation-onset divergence at t ~ 1031 s
    # so far, which is not a stable configuration to draw sanity windows from yet. DO NOT
    # borrow the warm o01_rainfall windows above -- the ice arm's rain/rain-number/ice
    # control-variable transforms and the ice thermodynamics (rho_i in the temperature
    # retrieval, mixture heat capacities, entropy, rho_t budget) make it a different run,
    # and load_targets() will not fall back to the base case for exactly that reason: an
    # armed run against this empty entry gets zero targets, not a spurious pass/fail.
    "o01_rainfall_ice" => Dict{String,Tuple{Float64,Float64,Float64,String}}(),
    # Nested 3-level arm (--nests 3): its OWN windows, looked up by the nest-qualified
    # key in run_nested_benchmark. Two reasons it cannot borrow the single-grid windows
    # above: (1) max_w/min_w are the RUN-MAXIMUM over all snapshots on the nested arm
    # (user decision 2026-07-14) vs the final snapshot on the single grid — a different
    # unit (the single grid's own run-maximum is 65.1 m/s against its reported
    # final-time 16.5); (2) the fine nests resolve the convective core (1|0.5|0.25 km
    # since 82a0fde), which legitimately rains ~25% more than the 500 m single grid.
    # Centers/spreads from the four preserved nested full runs 2026-07-22..08-20
    # (n3_stage5_pre, n3_ladder_none, n3_ladder_bhyp, and the 2026-08-20 re-seed):
    # accum 3.85-4.05, max_rho_r 19.9-21.3, max_w 41.9-44.6 — family spread ~5%.
    # PROVISIONAL: author review at the S9 close-out.
    "o01_rainfall_n3" => Dict(
        "peak_rain_rate_gm2s" => (175.0, 50.0, 60.0, "seeded 2026-08-20 (nested family; single-grid full 118 + refinement)"),
        "max_rho_r_gm3"       => (20.5,  4.0,  5.0,  "seeded 2026-08-20 (nested family 19.9-21.3)"),
        "max_w"               => (43.0,  8.0, 10.0,  "seeded 2026-08-20 (RUN-MAX definition; family 41.9-44.6)"),
        "rain_onset_min"      => (26.0,  6.0,  8.0,  "seeded 2026-08-20 (family 24-25; single grid 23-31)"),
        "accum_rainfall_mm"   => (3.95,  1.0,  1.2,  "seeded 2026-08-20 (nested family 3.85-4.05)"),
    ),
    "o01_axisym" => Dict(
        "peak_rain_rate_gm2s" => (55.0, 35.0, 35.0, "o01_rainfall window (metric terms ~0.2%)"),
        "max_rho_r_gm3"       => (8.5,  4.0,  4.0,  "o01_rainfall window"),
        "max_w"               => (9.0,  4.0,  4.0,  "o01_rainfall window"),
        "rain_onset_min"      => (24.0, 6.0,  8.0,  "o01_rainfall window"),
        "accum_rainfall_mm"   => (1.05, 0.5,  0.6,  "o01_rainfall window"),
        "max_abs_v"           => (0.0,  1.0e-12, 1.0e-12, "v has no source at f = 0, v0 = 0"),
    ),
)
