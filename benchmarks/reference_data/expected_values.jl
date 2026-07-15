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
    # O01 warm rain: sanity WINDOWS seeded from the accepted 2026-07-13 runs and
    # re-validated 2026-07-15 after the cloud-presence rain-condensation gate,
    # the stratospheric Rayleigh sponge (alpha 0.02, z_damp 17 km) and the
    # 20 -> 25 km model top (full 500 m: peak rate 78.5 g m^-2 s^-1, onset
    # 24 min, max w 7.9, max rho_r 11.2, accum 0.80 mm; quick 2 km: 27.9 /
    # 24 min / 10.1 / 5.8 / 1.37 mm — all inside the original windows). The mc
    # stage always takes the quick tolerance, so each window must span BOTH
    # resolutions — resolution halves the peak rate exactly as in O01's own
    # Dx = 1 vs 2 km comparison (Fig. 6: ~125 vs ~75 g m^-2 s^-1). The equation
    # set, tau_r microphysics closure (no separate Qevap), humidified-Dunion
    # sounding and single non-nested grid all differ from Ooyama (2001), so only
    # comparable magnitude and timing are enforced here; tight version-to-version
    # regression comes from the committed reference CSVs.
    "o01_rainfall" => Dict(
        "peak_rain_rate_gm2s" => (55.0, 35.0, 35.0, "seeded 2026-07-13; O01 Fig. 6 range 75-125"),
        "max_rho_r_gm3"       => (8.5,  4.0,  4.0,  "seeded 2026-07-13; O01 Figs. 4-6, 11 show ~4"),
        "max_w"               => (9.0,  4.0,  4.0,  "seeded 2026-07-13"),
        "rain_onset_min"      => (24.0, 6.0,  8.0,  "seeded 2026-07-13; O01 Fig. 6 ~25-40 min"),
        "accum_rainfall_mm"   => (1.05, 0.5,  0.6,  "seeded 2026-07-13 (domain-mean rain-out)"),
    ),
    # Axisymmetric-cylinder O01 at large mean radius (r in [1000, 1150] km, f = 0):
    # the metric terms are O(dx/r) ~ 0.2%, so the rain windows are the o01_rainfall
    # quick values; the tangential wind must stay identically zero (no source with
    # f = 0 and v0 = 0), which validates the 9-var cylindrical machinery.
    "o01_axisym" => Dict(
        "peak_rain_rate_gm2s" => (55.0, 35.0, 35.0, "o01_rainfall window (metric terms ~0.2%)"),
        "max_rho_r_gm3"       => (8.5,  4.0,  4.0,  "o01_rainfall window"),
        "max_w"               => (9.0,  4.0,  4.0,  "o01_rainfall window"),
        "rain_onset_min"      => (24.0, 6.0,  8.0,  "o01_rainfall window"),
        "accum_rainfall_mm"   => (1.05, 0.5,  0.6,  "o01_rainfall window"),
        "max_abs_v"           => (0.0,  1.0e-12, 1.0e-12, "v has no source at f = 0, v0 = 0"),
    ),
)
