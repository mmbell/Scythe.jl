using Test
using Scythe
using SpecialFunctions: gamma

# ──────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────
const ISHMAEL_JLD2_PATH = joinpath(@__DIR__, "..", "data", "ishmael_tables.jld2")

if !isfile(ISHMAEL_JLD2_PATH)
    # The converted tables are not tracked in git (44.5 MB). Regenerate from the
    # CM1 binaries with: julia --project=. tools/convert_ishmael_tables.jl
    @warn "ISHMAEL tables data file missing; skipping ISHMAEL table tests" path = ISHMAEL_JLD2_PATH
    @testset "ISHMAEL tables (skipped: data file missing)" begin
        @test_skip isfile(ISHMAEL_JLD2_PATH)
    end
else

const ISHMAEL_TABLES = Scythe.load_ishmael_tables(ISHMAEL_JLD2_PATH)

@testset "ISHMAEL tables" begin

    # ──────────────────────────────────────────────
    # 1. Loading: shapes, finiteness, spot values
    # ──────────────────────────────────────────────
    @testset "load_ishmael_tables" begin
        @test size(ISHMAEL_TABLES.itab)  == (51, 51, 51, 11, 2)
        @test size(ISHMAEL_TABLES.itabr) == (51, 51, 51, 11, 6)
        @test size(ISHMAEL_TABLES.coltab)  == (60, 60, 35)
        @test size(ISHMAEL_TABLES.coltabn) == (60, 60, 35)
        @test length(ISHMAEL_TABLES.igrdata) == 60

        @test all(isfinite, ISHMAEL_TABLES.itab)
        @test all(isfinite, ISHMAEL_TABLES.itabr)
        @test all(isfinite, ISHMAEL_TABLES.coltab)
        @test all(isfinite, ISHMAEL_TABLES.coltabn)
        @test all(isfinite, ISHMAEL_TABLES.igrdata)

        # Spot values -- must match what tools/convert_ishmael_tables.jl
        # printed when it built data/ishmael_tables.jld2 (see report).
        @test ISHMAEL_TABLES.itab[1, 1, 1, 1, 1]   == 0.0
        @test ISHMAEL_TABLES.itab[26, 26, 26, 6, 1] ≈ 3.180200038e-19  rtol=1.0e-6
        @test ISHMAEL_TABLES.itab[51, 51, 51, 11, 2] ≈ 3.262589985e-07 rtol=1.0e-6

        @test ISHMAEL_TABLES.itabr[1, 1, 1, 1, 1]   == 0.0
        @test ISHMAEL_TABLES.itabr[26, 26, 26, 6, 1] ≈ 1.032860033e-14 rtol=1.0e-6
        @test ISHMAEL_TABLES.itabr[51, 51, 51, 11, 2] ≈ 1.282860041    rtol=1.0e-6
        @test ISHMAEL_TABLES.itabr[51, 51, 51, 11, 6] ≈ 0.0002640539897 rtol=1.0e-6

        # Loading twice gives bitwise-identical tables (deterministic round trip)
        tables2 = Scythe.load_ishmael_tables(ISHMAEL_JLD2_PATH)
        @test tables2.itab == ISHMAEL_TABLES.itab
        @test tables2.itabr == ISHMAEL_TABLES.itabr
    end

    # ──────────────────────────────────────────────
    # 2. access_lookup_table
    # ──────────────────────────────────────────────
    @testset "access_lookup_table" begin
        itab = ISHMAEL_TABLES.itab

        # Interpolating exactly at a grid node reproduces the stored value
        for (jj, ii, i, k, idx) in ((1, 1, 1, 1, 1), (10, 10, 10, 5, 2), (50, 50, 50, 10, 1))
            v = Scythe.access_lookup_table(itab, jj, ii, i, k, idx,
                                            Float64(i), Float64(k), Float64(ii), Float64(jj))
            @test v ≈ itab[jj, ii, i, k, idx] atol=1.0e-12
        end

        # Continuity: interpolated value between two adjacent nodes lies
        # between the node values (monotone-bracket property of quadrilinear
        # interpolation applied along a single axis at a time)
        jj, ii, i, k, idx = 5, 5, 5, 5, 1
        v0 = itab[jj, ii, i, k, idx]
        v1 = itab[jj, ii, i+1, k, idx]
        vmid = Scythe.access_lookup_table(itab, jj, ii, i, k, idx,
                                           Float64(i) + 0.5, Float64(k), Float64(ii), Float64(jj))
        lo, hi = minmax(v0, v1)
        @test lo - 1.0e-9 <= vmid <= hi + 1.0e-9

        # No NaN near the table edges (last valid base index so that +1 stays in bounds)
        v_edge = Scythe.access_lookup_table(itab, 50, 50, 50, 10, 2,
                                             50.7, 10.3, 50.2, 50.9)
        @test isfinite(v_edge)
        v_edge_r = Scythe.access_lookup_table(ISHMAEL_TABLES.itabr, 50, 50, 50, 10, 6,
                                               50.1, 10.9, 50.5, 50.4)
        @test isfinite(v_edge_r)

        # Zero allocations (warm up first)
        Scythe.access_lookup_table(itab, 5, 5, 5, 5, 1, 5.3, 5.6, 5.2, 5.4)
        allocs = @allocated Scythe.access_lookup_table(itab, 5, 5, 5, 5, 1, 5.3, 5.6, 5.2, 5.4)
        @test allocs == 0
    end

    # ──────────────────────────────────────────────
    # 3. capacitance_gamma
    # ──────────────────────────────────────────────
    @testset "capacitance_gamma" begin
        # Sphere limit: dsdum=1, NU=1, alphstr=1 -> capacitance = ani (analytic
        # sphere capacitance = radius), per the Fortran's own special case
        # (lines 3517-3519) and the general formula collapsing to it.
        for ani in (1.0e-6, 5.0e-5, 2.0e-3)
            c = Scythe.capacitance_gamma(ani, 1.0, 1.0, 1.0, 1.0 / gamma(1.0))
            @test c ≈ ani rtol=1.0e-10
        end

        # Continuity of the oblate (dsdum<=1) / prolate (dsdum>1) branches
        # across dsdum = 1
        NU = 2.0
        i_gammnu = 1.0 / gamma(NU)
        ani = 3.0e-5
        alphstr = 1.0
        c_below = Scythe.capacitance_gamma(ani, 1.0 - 1.0e-6, NU, alphstr, i_gammnu)
        c_at    = Scythe.capacitance_gamma(ani, 1.0,          NU, alphstr, i_gammnu)
        c_above = Scythe.capacitance_gamma(ani, 1.0 + 1.0e-6, NU, alphstr, i_gammnu)
        @test c_below ≈ c_at rtol=1.0e-4
        @test c_above ≈ c_at rtol=1.0e-4

        # Positive over a physically reasonable range of shape/size
        for dsdum in (0.55, 0.8, 1.0, 1.1, 1.3), ani in (1.0e-6, 1.0e-4, 1.0e-3)
            NU3 = 4.0
            c = Scythe.capacitance_gamma(ani, dsdum, NU3, 1.0, 1.0 / gamma(NU3))
            @test c > 0.0
            @test isfinite(c)
        end
    end

    # ──────────────────────────────────────────────
    # 4. get_igr
    # ──────────────────────────────────────────────
    @testset "get_igr" begin
        igr = ISHMAEL_TABLES.igrdata
        T0 = 273.15

        # Matches igrdata exactly at integer-degree sample temperatures
        for n in (1, 15, 30, 45, 59)
            @test Scythe.get_igr(igr, T0 - Float64(n)) ≈ igr[n] atol=1.0e-10
        end

        # Between 0 C and -1 C: interpolates from igr1=1.0 toward igrdata[1]
        v_half = Scythe.get_igr(igr, T0 - 0.5)
        @test 0.5 * (1.0 + igr[1]) ≈ v_half atol=1.0e-10
        @test Scythe.get_igr(igr, T0) == 1.0

        # Above 0 C: always 1.0 (else branch)
        @test Scythe.get_igr(igr, T0 + 5.0) == 1.0
        @test Scythe.get_igr(igr, T0 + 30.0) == 1.0

        # Clamps beyond -60 C to igrdata[60]
        @test Scythe.get_igr(igr, T0 - 60.0) ≈ igr[60] atol=1.0e-10
        @test Scythe.get_igr(igr, T0 - 90.0) == igr[60]

        # igr crosses 1 where the data says: igrdata goes below 1 for the
        # first few entries (plate regime near -1..-4 C) and again briefly
        # dips below 1 mid-table, per the raw data.
        @test igr[1] < 1.0 && igr[4] < 1.0
        @test igr[5] > 1.0   # -5 C: crosses back above 1
        @test all(v -> v > 1.0, igr[31:end])  # settles > 1 for the cold branch
    end

    # ──────────────────────────────────────────────
    # 5. ishmael_var_check
    # ──────────────────────────────────────────────
    @testset "ishmael_var_check" begin
        NU = 4.0
        ao = 1.0e-4
        fourthirdspi = 4.0 / 3.0 * pi
        gammnu = gamma(NU)
        gam7 = gamma(NU + 3.0)     # gamma(NU+2+dsdum) at dsdum=1
        alphv1 = fourthirdspi       # alphv at dsdum=1 (alphstr = ao^0 = 1)

        # 5a. Spherical case: solid ice (rhobar -> 920), deltastr -> 1
        ani_s, ni_s = 5.0e-5, 1.0e6
        qi_s = 920.0 * ni_s * alphv1 * ani_s^3 * gam7 / gammnu
        r_sphere = Scythe.ishmael_var_check(NU, ao, fourthirdspi, gammnu, qi_s,
                                             1.0, ani_s, ani_s, 920.0, ni_s, 0.0, 0.0)
        @test r_sphere.deltastr ≈ 1.0 atol=1.0e-12
        @test r_sphere.rhobar ≈ 920.0 atol=1.0e-6
        @test r_sphere.ani ≈ ani_s rtol=1.0e-6
        @test r_sphere.rni ≈ ani_s rtol=1.0e-6   # sphere identity: rni == ani when dsdum=1

        # 5b. Idempotence: feeding the (already-consistent) output straight
        # back in as input returns the identical NamedTuple (fixed point)
        r_again = Scythe.ishmael_var_check(NU, ao, fourthirdspi, gammnu, qi_s,
                                            r_sphere.deltastr, r_sphere.ani, r_sphere.cni,
                                            r_sphere.rhobar, r_sphere.ni, r_sphere.ai, r_sphere.ci)
        @test r_again.deltastr ≈ r_sphere.deltastr atol=1.0e-10
        @test r_again.ani      ≈ r_sphere.ani      rtol=1.0e-8
        @test r_again.cni      ≈ r_sphere.cni      rtol=1.0e-8
        @test r_again.rni      ≈ r_sphere.rni      rtol=1.0e-8
        @test r_again.rhobar   ≈ r_sphere.rhobar   atol=1.0e-6
        @test r_again.ni       ≈ r_sphere.ni       rtol=1.0e-8

        # 5c. deltastr floor (< 0.55 clamps to 0.55)
        r = Scythe.ishmael_var_check(NU, ao, fourthirdspi, gammnu, 1.0e-4, 0.3,
                                      50.0e-6, 50.0e-6, 500.0, 1.0e6, 0.0, 0.0)
        @test r.deltastr == 0.55

        # 5d. deltastr ceiling (> 1.3 clamps to 1.3)
        r = Scythe.ishmael_var_check(NU, ao, fourthirdspi, gammnu, 1.0e-4, 2.0,
                                      50.0e-6, 50.0e-6, 500.0, 1.0e6, 0.0, 0.0)
        @test r.deltastr == 1.3

        # 5e. Bulk density ceiling (rbdum > RHOI clamps to RHOI=920), isolated
        # (ani chosen so the computed rbdum overshoots without also tripping
        # the small/large-ice clamps)
        ani3, ni3 = 20.0e-6, 1.0e6
        r = Scythe.ishmael_var_check(NU, ao, fourthirdspi, gammnu, 1.0, 1.0,
                                      ani3, ani3, 920.0, ni3, 0.0, 0.0)
        @test r.rhobar == 920.0
        @test r.deltastr == 1.0

        # 5f. Bulk density floor (rbdum < 50 clamps to 50)
        r = Scythe.ishmael_var_check(NU, ao, fourthirdspi, gammnu, 1.0e-8, 1.0,
                                      ani3, ani3, 920.0, ni3, 0.0, 0.0)
        @test r.rhobar == 50.0

        # 5g. Small-ice limit (rni floored at 2 micron), isolated via the
        # sphere identity rni==ani at dsdum=1: pick ani well under 2 micron.
        ani_tiny, ni_tiny = 1.0e-6, 1.0e6
        qi_tiny = 920.0 * ni_tiny * alphv1 * ani_tiny^3 * gam7 / gammnu
        r = Scythe.ishmael_var_check(NU, ao, fourthirdspi, gammnu, qi_tiny, 1.0,
                                      ani_tiny, ani_tiny, 920.0, ni_tiny, 0.0, 0.0)
        @test r.rni == 2.0e-6
        @test r.ani < 1.0e-3   # large-ice branch not also triggered

        # 5h. Large-ice limit (max axis capped at 1 mm), isolated: ani chosen
        # so the density check is satisfied (rbdum -> 920) before the
        # large-ice check fires.
        ani_big, ni_big = 2.0e-3, 1.0e6
        qi_big = 920.0 * ni_big * alphv1 * ani_big^3 * gam7 / gammnu
        r = Scythe.ishmael_var_check(NU, ao, fourthirdspi, gammnu, qi_big, 1.0,
                                      ani_big, ani_big, 920.0, ni_big, 0.0, 0.0)
        @test r.ani == 1.0e-3
        @test r.cni == 1.0e-3
        @test r.rhobar == 920.0   # unaffected by the large-ice branch
        @test isfinite(r.rni) && r.rni > 0.0
    end

    # ──────────────────────────────────────────────
    # 6. mkcoltb / xjnum / incomplete-gamma / avint
    # ──────────────────────────────────────────────
    @testset "mkcoltb" begin
        coltab, coltabn = Scythe.mkcoltb()

        @test size(coltab) == (60, 60, 35)
        @test size(coltabn) == (60, 60, 35)
        @test all(isfinite, coltab)
        @test all(isfinite, coltabn)
        @test all(v -> v >= 0.0, coltab)
        @test all(v -> v >= 0.0, coltabn)

        # Structural check on ISHMAEL_IPAIR ("symmetric where physics says"):
        # every one of the 35 collection-table slots is used exactly once
        # (no duplicate or missing slot in the ipair map), and self-
        # collection (diagonal) slots exist only for the ice species that
        # actually self-aggregate (planar=3, columnar=4, aggregates=5) --
        # cloud/rain/graup/hail (1,2,6,7) have no self-collection slot.
        vals = sort(filter(v -> v > 0, vec(Scythe.ISHMAEL_IPAIR)))
        @test vals == collect(1:35)
        @test [Scythe.ISHMAEL_IPAIR[i, i] for i in (3, 4, 5)] == [22, 28, 35]
        @test all(Scythe.ISHMAEL_IPAIR[i, i] == 0 for i in (1, 2, 6, 7))

        # coltab/coltabn actually populated (nonzero) at the self-collection
        # slots, and remain zero at any pair that was never built (shouldn't
        # happen here since every ipair>0 slot is visited, but this guards
        # against an indexing mistake silently leaving a slot at its zeros()
        # initial value)
        @test maximum(coltab[:, :, 22]) > 0.0     # planar self-collection
        @test maximum(coltabn[:, :, 22]) > 0.0

        # Incomplete-gamma replacements: gammap+gammaq == 1 (regularization
        # identity) and match a brute-force numerical integral of the
        # defining integral to rtol=1e-8, at 3 hand-picked (a,x) points.
        function brute_force_lower_gamma_p(a, x; n = 400_000)
            f(t) = t^(a - 1.0) * exp(-t)
            h = x / n
            s = f(1.0e-12) + f(x)
            for i in 1:2:(n - 1)
                s += 4.0 * f(i * h)
            end
            for i in 2:2:(n - 2)
                s += 2.0 * f(i * h)
            end
            return (s * h / 3.0) / gamma(a)
        end
        for (a, x) in ((4.0, 2.0), (4.25, 5.0), (4.5, 0.5))
            p_sf = Scythe.ishmael_gammap(a, x)
            q_sf = Scythe.ishmael_gammaq(a, x)
            @test p_sf + q_sf ≈ 1.0 atol=1.0e-12
            p_bf = brute_force_lower_gamma_p(a, x)
            @test p_sf ≈ p_bf rtol=1.0e-8
        end

        # AVINT replacement (overlapping-parabola quadrature): exact for a
        # quadratic integrand (the method's polynomial degree of exactness),
        # and within rtol=1e-6 of a fine composite-Simpson reference for a
        # smooth non-polynomial integrand on a modestly-resolved node grid
        # (convergence itself confirms the transliteration -- see report).
        xq = collect(range(0.0, 10.0, length = 15))
        yq = xq .^ 2 .+ 3.0 .* xq .+ 1.0
        ans_quad = Scythe.ishmael_avint(xq, yq, 0.0, 10.0)
        exact_quad = 10.0^3 / 3.0 + 3.0 * 10.0^2 / 2.0 + 10.0
        @test ans_quad ≈ exact_quad rtol=1.0e-10

        function simpson_ref(f, a, b, n)
            n = iseven(n) ? n : n + 1
            h = (b - a) / n
            s = f(a) + f(b)
            for i in 1:2:(n - 1)
                s += 4.0 * f(a + i * h)
            end
            for i in 2:2:(n - 2)
                s += 2.0 * f(a + i * h)
            end
            return s * h / 3.0
        end
        xs = collect(range(0.1, 5.0, length = 201))
        ys = exp.(-xs) .* sin.(xs)
        ans_smooth = Scythe.ishmael_avint(xs, ys, 0.1, 5.0)
        ref_smooth = simpson_ref(t -> exp(-t) * sin(t), 0.1, 5.0, 200_000)
        @test ans_smooth ≈ ref_smooth rtol=1.0e-6

        # ishmael_avint error paths (mirrors the Fortran's fatal STOPs)
        @test_throws ErrorException Scythe.ishmael_avint([1.0, 0.5, 2.0], [1.0, 1.0, 1.0], 1.0, 2.0)  # not increasing
        @test_throws ErrorException Scythe.ishmael_avint([1.0, 2.0], [1.0, 1.0], 2.0, 1.0)            # xup < xlo
        @test Scythe.ishmael_avint([1.0, 2.0], [1.0, 1.0], 1.5, 1.5) == 0.0                             # xlo == xup
    end
end

end # isfile(ISHMAEL_JLD2_PATH) guard
