# Published benchmark target values and tolerances.
#
# Format: "case" => Dict("diagnostic" => (target, atol_full, atol_quick, source))
#
# Full-mode tolerances are wider than the inter-reference-model spread reported
# in the papers (Scythe is a spectral-transform model; the references are finite
# difference codes). Quick-mode tolerances additionally allow for the coarser
# grid. The primitive equation stage (--stage pe) uses the quick tolerances.
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
        "max_u"          => (36.46,    2.0, 4.0, "Straka93 Table V REFC"),
        "min_u"          => (-15.19,   2.0, 4.0, "Straka93 Table V REFC"),
        "max_w"          => (12.93,    1.5, 3.0, "Straka93 Table V REFC"),
        "min_w"          => (-15.95,   1.5, 3.0, "Straka93 Table V REFC"),
        "front_location" => (15537.44, 250.0, 750.0, "Straka93 Table IV REFC25"),
    ),
    "bf02_dry" => Dict(
        "max_theta_p" => (2.07178,   0.15, 0.5, "BF02 Fig. 1a (100 m)"),
        "min_theta_p" => (-0.144409, 0.10, 0.3, "BF02 Fig. 1a"),
        "max_w"       => (14.5396,   0.75, 2.0, "BF02 Fig. 1b"),
        "min_w"       => (-8.58069,  0.75, 2.0, "BF02 Fig. 1b"),
    ),
    "bf02_moist" => Dict(
        "max_theta_e_p" => (4.09521,   0.3, 0.8, "BF02 Fig. 3a (100 m)"),
        "min_theta_e_p" => (-0.305695, 0.2, 0.4, "BF02 Fig. 3a"),
        "max_w"         => (15.713,    1.0, 2.5, "BF02 Fig. 3b"),
        "min_w"         => (-9.92698,  1.0, 2.5, "BF02 Fig. 3b"),
    ),
)
