using Test
using Scythe
using Springsteel
using SparseArrays

# The tile/state/budget fixtures, shared with test_mynn_bl.jl.
isdefined(Main, :MYNNTestFixtures) || include("mynn_test_fixtures.jl")
using .MYNNTestFixtures

# Tests for the MYNN fidelity switches (plan stage F1): `options[:mynn_fidelity]` turned
# from a LABEL into seven named, individually measurable deviations from the verbatim
# Fortran closure and coupling (`Scythe.MYNN_DEVIATIONS`).
#
# Three things are being asserted, in this order of importance:
#
#   1. the DEFAULT is unchanged. `:fortran` -- and no key at all -- must produce bitwise
#      the same tendencies, the same `K`, the same mixing length. Every deviation is one
#      branch, and this is the test that says the branch is never taken by accident.
#   2. each switch does the ONE thing it names, checked against a value recomputed here
#      from the Fortran arm's own output rather than against a stored number. Where the
#      deviation is a pure factor (`gtr_local` on `rmol`, `sqfac1` on `D_gal`,
#      `K_interface` on `K_m`), the check is that factor and nothing else.
#   3. the D3 column energy identity still closes to round-off under every one of them.
#      A switch that quietly opens the energy budget is not a fidelity experiment, it is
#      a bug, and the whole point of the stage is that the deviations can be MEASURED
#      against the control without the measurement being confounded.
#
# `test_mynn_closure.jl` and `test_mynn_edmf.jl` (the Fortran parity anchors) are
# deliberately NOT touched by this stage: they call the kernels positionally with no
# fidelity keyword, which is the default path.

@testset "MYNN-EDMF fidelity switches (F1)" begin

    import Springsteel.Thermodynamics: gravity

    DEV = Scythe.MYNN_DEVIATIONS

    # ──────────────────────────────────────────────
    # 1. Option validation
    # ──────────────────────────────────────────────
    @testset "options[:mynn_fidelity] resolves to named switches" begin
        V(o, p = Dict{Symbol,Float64}(); ts = 0.5, kDim = 24) =
            Scythe.validate_mynn_options(o, p, "moist_compressible_XZ", ts, kDim)
        on() = Dict{Symbol,Any}(:mynn => true)
        with(fid; extra...) = merge(on(), Dict{Symbol,Any}(:mynn_fidelity => fid),
                                    Dict{Symbol,Any}(extra...))

        # The default, an explicit `:fortran` and an empty list are the SAME switch set.
        @test isempty(V(on()).fidelity.names)
        @test V(on()).fidelity == Scythe.MYNN_FORTRAN_FIDELITY
        @test V(with(:fortran)).fidelity == Scythe.MYNN_FORTRAN_FIDELITY
        @test V(with(Symbol[])).fidelity == Scythe.MYNN_FORTRAN_FIDELITY
        # ...and no deviation is on in it.
        f0 = V(on()).fidelity
        @test !f0.gtr_local && !f0.K_interface && !f0.pdk1 && !f0.exner_single &&
              !f0.rmol_sfc && !f0.flux_clip
        @test f0.sqfac == Scythe.MYNN_SQFAC

        # Each name sets EXACTLY its own field.
        @test length(DEV) == 7
        for nm in DEV
            o = with([nm])
            nm === :rmol_sfc && (o[:sfc_stability] = true)
            f = V(o).fidelity
            @test f.names == [nm]
            @test f.gtr_local == (nm === :gtr_local)
            @test f.K_interface == (nm === :K_interface)
            @test f.pdk1 == (nm === :pdk1)
            @test f.exner_single == (nm === :exner_single)
            @test f.rmol_sfc == (nm === :rmol_sfc)
            @test f.flux_clip == (nm === :flux_clip)
            @test f.sqfac == (nm === :sqfac1 ? 1.0 : Scythe.MYNN_SQFAC)
        end
        # ...and a list sets all of them.
        fall = V(with(collect(DEV); sfc_stability = true)).fidelity
        @test fall.names == collect(DEV)
        @test fall.gtr_local && fall.K_interface && fall.pdk1 && fall.exner_single &&
              fall.rmol_sfc && fall.flux_clip && fall.sqfac == 1.0

        # A misspelling is a typo, not a no-op -- the whole reason the option is
        # validated at setup (the Louis-BL and Khdiff_water blockers).
        @test_throws ErrorException V(with([:gtr_locl]))
        @test_throws ErrorException V(with([:scythe]))
        @test_throws ErrorException V(with([:pdk1, :pdk1]))         # a repeat
        @test_throws ErrorException V(with(Any[:pdk1, "sqfac1"]))   # not all Symbols
        @test_throws ErrorException V(with("gtr_local"))            # not a list
        @test_throws ErrorException V(with(:fast))                  # not a fidelity
        @test_throws ErrorException V(with(1))
        # `:rmol_sfc` without the Monin-Obukhov surface layer would measure a NEUTRAL
        # surface layer (inv_L is identically 0.0 there), not a better one.
        @test_throws ErrorException V(with([:rmol_sfc]))
        @test_throws ErrorException V(with([:gtr_local, :rmol_sfc]))
        @test V(with([:rmol_sfc]; sfc_stability = true)).fidelity.rmol_sfc
        # ...but the OFF path refuses nothing, a broken fidelity included.
        @test V(Dict{Symbol,Any}(:mynn => false,
                                 :mynn_fidelity => :nonsense)).active == false

        # String round trip: the setup line, the NetCDF attribute and the env knobs all
        # go through these two.
        @test Scythe.mynn_fidelity_string(Scythe.MYNN_FORTRAN_FIDELITY) == "fortran"
        f = Scythe.MYNNFidelity([:gtr_local, :pdk1])
        @test Scythe.mynn_fidelity_string(f) == "gtr_local,pdk1"
        @test Scythe.parse_mynn_fidelity("fortran") === :fortran
        @test Scythe.parse_mynn_fidelity("") === :fortran
        @test Scythe.parse_mynn_fidelity("gtr_local,pdk1") == [:gtr_local, :pdk1]
        @test Scythe.parse_mynn_fidelity(" gtr_local , pdk1 ") == [:gtr_local, :pdk1]
        @test Scythe.MYNNFidelity(Scythe.parse_mynn_fidelity(
                  Scythe.mynn_fidelity_string(f))) == f
        # A parsed typo still dies at the validator, not silently in the run.
        @test_throws ErrorException Scythe.MYNNFidelity(
            Scythe.parse_mynn_fidelity("gtr_locl"))
    end

    @testset "the setup line states the fidelity" begin
        mktempdir() do tmp
            _, _, mo, gpo, _, _ = make_mynn_tile(tmp; mynn = true, num_cells_k = 25)
            cfg = Scythe.validate_mynn_options(mo.options, mo.physical_params,
                                               mo.equation_set, mo.ts, gpo.kDim)
            # `mynn_setup_line` prints (it is what lands in the per-nest
            # scythe_out.log), so the line is captured rather than returned. `trace` is
            # forced on here; the fixture turns it off so the suite stays quiet.
            say(c) = begin
                pipe = Pipe()
                Base.link_pipe!(pipe; reader_supports_async = true,
                                writer_supports_async = true)
                redirect_stdout(pipe) do
                    Scythe.mynn_setup_line(mo, merge(c, (; trace = true)), 4, gpo.kDim)
                end
                close(pipe.in)
                read(pipe, String)
            end
            @test occursin("fidelity=fortran", say(cfg))
            devs = Scythe.MYNNFidelity([:gtr_local, :pdk1])
            @test occursin("fidelity=gtr_local,pdk1", say(merge(cfg, (; fidelity = devs))))
        end
    end

    # ──────────────────────────────────────────────
    # 2. `:fortran` is BITWISE today's code
    # ──────────────────────────────────────────────
    @testset ":fortran is bitwise the no-key default" begin
        mktempdir() do tmp
            # `fidelity = nothing` leaves `:mynn_fidelity` out of the options dict
            # entirely; `fidelity = :fortran` sets it explicitly. Those two runs must not
            # differ in a single bit -- `==`, never `isapprox`.
            m_a, p_a, mo, gpo, z, col = make_mynn_tile(tmp; mynn = true,
                                                       fidelity = nothing)
            m_b, p_b, mob, _, _, _ = make_mynn_tile(tmp; mynn = true,
                                                    fidelity = :fortran)
            kDim = gpo.kDim
            set_state!(p_a, mo, kDim, z, col)
            set_state!(p_b, mob, kDim, z, col)
            step_all!(m_a, p_a, kDim)
            step_all!(m_b, p_b, kDim)

            @test m_a.mynn.fidelity == Scythe.MYNN_FORTRAN_FIDELITY
            @test m_b.mynn.fidelity == Scythe.MYNN_FORTRAN_FIDELITY
            @test size(m_a.expdot_n) == size(m_b.expdot_n)
            @test m_a.expdot_n == m_b.expdot_n           # every slot, every gridpoint
            @test m_a.mynn.K_m == m_b.mynn.K_m
            @test m_a.mynn.K_h == m_b.mynn.K_h
            @test m_a.mynn.el == m_b.mynn.el
            @test m_a.mynn.gh == m_b.mynn.gh
            @test m_a.mynn.rmol == m_b.mynn.rmol
            # ...and the run is not trivially zero, or the equality above says nothing.
            @test maximum(m_a.mynn.K_m) > 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 3. Each deviation does its one thing
    # ──────────────────────────────────────────────
    """Fortran and deviation tiles on the SAME state, each advanced one step.

    Returns `(m_fortran, m_dev, model, grid_params, z, col)`. `state = false` leaves every
    perturbation at exactly zero -- a resting column whose `theta_v`, `p` and `q_v` at the
    surface are the reference's own, so a surface-block identity can be recomputed here
    from the sounding with nothing assumed about the retrieval."""
    function fid_pair(tmp; fidelity, state = true, kw...)
        m_f, p_f, mo_f, gp, z, col = make_mynn_tile(tmp; mynn = true, kw...)
        m_d, p_d, mo_d, _, _, _ = make_mynn_tile(tmp; mynn = true, fidelity = fidelity,
                                                 kw...)
        kDim = gp.kDim
        if state
            set_state!(p_f, mo_f, kDim, z, col)
            set_state!(p_d, mo_d, kDim, z, col)
        end
        step_all!(m_f, p_f, kDim)
        step_all!(m_d, p_d, kDim)
        return (m_f, m_d, mo_f, gp, z, col)
    end

    "The column whose numbers are asserted: interior, away from both side walls."
    CI = 6
    colrange(kDim, c = CI) = ((c - 1) * kDim + 1):(c * kDim)

    @testset ":gtr_local -- g/theta_v per level, in the kernel and at the surface" begin
        c = Scythe.MYNNConstants()
        # (a) the KERNEL, where the deviation is exactly one factor on G_H. `dtq` does
        #     not depend on the buoyancy parameter, so the ratio of the two `gh` columns
        #     IS the interface-interpolated `g/theta_v` over `c.gtr`, and nothing else.
        n = 8
        dz = [40.0, 45.0, 50.0, 60.0, 70.0, 85.0, 100.0, 120.0]
        u = [2.0 + 0.9 * k for k in 1:n]
        v = [0.5 - 0.2 * k for k in 1:n]
        thl = [300.0 + 0.004 * sum(dz[1:k]) for k in 1:n]
        qw = [8.0e-3 * exp(-sum(dz[1:k]) / 2000.0) for k in 1:n]
        ql = zeros(n)
        thetav = [thl[k] * (1.0 + 0.608 * qw[k]) for k in 1:n]
        vt = fill(0.02, n)
        vq = fill(0.5, n)
        out() = (zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))
        dtl, dqw, dtv, gm, gh, sm, sh = out()
        Scythe.mym_level2!(1, n, dz, u, v, thl, thetav, qw, ql, vt, vq,
                           dtl, dqw, dtv, gm, gh, sm, sh, c)
        gh_F = copy(gh); gm_F = copy(gm)
        gtr_k = [c.grav / thetav[k] for k in 1:n]
        dtl, dqw, dtv, gm, gh, sm, sh = out()
        Scythe.mym_level2!(1, n, dz, u, v, thl, thetav, qw, ql, vt, vq,
                           dtl, dqw, dtv, gm, gh, sm, sh, c; gtr_k = gtr_k)
        # G_M is a pure shear and must not move at all.
        @test gm == gm_F
        for k in 2:n
            afk = dz[k] / (dz[k] + dz[k-1])
            G = gtr_k[k] * (1.0 - afk) + gtr_k[k-1] * afk
            @test gh_F[k] != 0.0
            @test isapprox(gh[k], gh_F[k] * G / c.gtr; rtol = 1.0e-13)
            @test gh[k] != gh_F[k]          # theta_v is not 300 K anywhere here
        end
        # ...and passing `nothing` explicitly is the Fortran, bitwise.
        dtl, dqw, dtv, gm, gh, sm, sh = out()
        Scythe.mym_level2!(1, n, dz, u, v, thl, thetav, qw, ql, vt, vq,
                           dtl, dqw, dtv, gm, gh, sm, sh, c; gtr_k = nothing)
        @test gh == gh_F

        # (b) WIRED, at the surface. `rmol = -karman gtr fltv/max(u*^3, 1e-6)` and `fltv`
        #     does not depend on the buoyancy parameter, so the two tiles' `rmol` differ
        #     by exactly `g/theta_v(1)` over `c.gtr`. On the resting column (no imposed
        #     perturbation) `theta_v(1)` is the sounding's own `T(1)/exner(1)` with
        #     `q_v = 0`, which is recomputed here rather than read out of the model.
        mktempdir() do tmp
            m_f, m_d, mo, gp, z, col = fid_pair(tmp; fidelity = [:gtr_local],
                                                state = false, num_cells_k = 25)
            p1 = Scythe.ref_pressure(m_f.ref_state)[1, 1]
            e1 = (p1 / Scythe.MYNN_P0)^c.rcp
            gtr_1 = c.grav / (col.Tk[1] / e1)
            @test m_f.mynn.rmol[CI] != 0.0
            @test m_d.mynn.rmol[CI] != m_f.mynn.rmol[CI]
            # The DEVIATION moves `rmol` by 300 K / theta_v(1) ~ 0.22 % here, and the
            # recomputation above is good to ~1e-5: `theta_v(1)` is rebuilt from the
            # SOUNDING while the model's is the retrieved state at the first mish point,
            # and the exact-reference fit puts ~1.5 Pa (0.002 K) between them. So the
            # tolerance is 1e-4 -- twenty times tighter than the effect being asserted,
            # and the line below says so rather than leaving it implied.
            @test isapprox(m_d.mynn.rmol[CI], m_f.mynn.rmol[CI] * gtr_1 / c.gtr;
                           rtol = 1.0e-4)
            @test abs(m_d.mynn.rmol[CI] / m_f.mynn.rmol[CI] - 1.0) > 1.0e-3
        end
    end

    @testset ":K_interface -- K from the wall average of el*S" begin
        mktempdir() do tmp
            m_f, m_d, mo, gp, z, col = fid_pair(tmp; fidelity = [:K_interface],
                                                num_cells_k = 25)
            kDim = gp.kDim
            rng = colrange(kDim)
            # The CLOSURE is untouched: this deviation lives entirely in
            # `_mynn_diffusivities!`, after `el`/`S_M`/`S_H` are already decided.
            @test m_d.mynn.el == m_f.mynn.el
            @test m_d.mynn.sm == m_f.mynn.sm
            @test m_d.mynn.sh == m_f.mynn.sh
            el = m_f.mynn.el; sm = m_f.mynn.sm; sh = m_f.mynn.sh
            Km_f = m_f.mynn.K_m; Km_d = m_d.mynn.K_m
            Kh_f = m_f.mynn.K_h; Kh_d = m_d.mynn.K_h
            j1 = first(rng); jn = last(rng)
            # The wall carries no interface, on either arm.
            @test Km_f[j1] == 0.0
            @test Km_d[j1] == 0.0
            @test Kh_d[j1] == 0.0
            # The top level has only the wall below it, so it is unchanged.
            @test Km_d[jn] == Km_f[jn]
            @test Kh_d[jn] == Kh_f[jn]
            # ...and every interior level is `q` times the WALL AVERAGE. Cross-multiplied
            # so `q` (which the state does not expose) cancels and no division by a
            # possibly-zero level is needed.
            nmoved = 0
            for j in (j1 + 1):(jn - 1)
                avg = 0.5 * (el[j] * sm[j] + el[j+1] * sm[j+1])
                @test isapprox(Km_d[j] * (el[j] * sm[j]), Km_f[j] * avg;
                               rtol = 1.0e-12, atol = 1.0e-30)
                avh = 0.5 * (el[j] * sh[j] + el[j+1] * sh[j+1])
                @test isapprox(Kh_d[j] * (el[j] * sh[j]), Kh_f[j] * avh;
                               rtol = 1.0e-12, atol = 1.0e-30)
                Km_d[j] != Km_f[j] && (nmoved += 1)
            end
            @test nmoved > 0            # ...and it actually moved something
        end
    end

    @testset ":sqfac1 -- K_e = K_m, and a third of the diffusion number" begin
        mktempdir() do tmp
            m_f, m_d, mo, gp, z, col = fid_pair(tmp; fidelity = [:sqfac1],
                                                num_cells_k = 25)
            re_i = mo.grid_params.vars["rho_e"]
            # `K_m`/`K_h` are untouched: Sqfac multiplies the TKE's own diffusivity only.
            @test m_d.mynn.K_m == m_f.mynn.K_m
            @test m_d.mynn.K_h == m_f.mynn.K_h
            # `D_gal = max(K_e) ts 10/dz_cell^2`, so it is exactly a third.
            @test m_f.mynn.D_gal[CI] > 0.0
            @test isapprox(m_d.mynn.D_gal[CI], m_f.mynn.D_gal[CI] / Scythe.MYNN_SQFAC;
                           rtol = 1.0e-14)
            # ...and the TKE slot felt it (the transport column is fitted on `K_e`).
            @test m_d.expdot_n[:, re_i] != m_f.expdot_n[:, re_i]
        end
    end

    @testset ":pdk1 -- the log-layer surface production, energy exactly" begin
        mktempdir() do tmp
            nk = 25
            m_f, m_d, mo, gp, z, col = fid_pair(tmp; fidelity = [:pdk1], num_cells_k = nk,
                                                Cd = -1.0, fluxes = true)
            kDim = gp.kDim
            rng = colrange(kDim)
            re_i = mo.grid_params.vars["rho_e"]
            wq = gauss_weights(kDim, nk, gp.kMax - gp.kMin)

            d_re = m_d.expdot_n[rng, re_i] .- m_f.expdot_n[rng, re_i]
            d_Et = m_d.expdot_n[rng, 6] .- m_f.expdot_n[rng, 6]
            I_re = sum(wq .* d_re)
            I_Et = sum(wq .* d_Et)
            @test abs(I_re) > 0.0
            # What leaves the TKE slot arrives in E_t as heat: `<dE_t + drho_e>` is
            # unchanged by the switch, which is why the D3 identity still closes below.
            @test isapprox(I_Et, -I_re; rtol = 1.0e-10)

            # ...and the amount is the Fortran's own `pdk1`, converted from a `qke`
            # production to a layer-integrated `e` production and delivered on `g(z)`:
            # `rho_t(1) u*^3 pmz/karman` in place of the drag work `tau_u u`.
            cM = Scythe.MYNNConstants()
            MY = m_d.mynn
            @test MY.pmz[CI] != 0.0                 # written under :pdk1
            @test m_f.mynn.pmz[CI] == 0.0           # ...and never under :fortran
            rho_t1 = m_d.tile.physical[first(rng), 3, 1] +
                     Scythe.ref_rho_t(m_d.ref_state)[1, 1]
            Psfc_e = rho_t1 * MY.ust[CI]^3 * MY.pmz[CI] / cM.karman
            delta = 2.0 * z[2]
            gz = [zk < delta ? 2.0 / delta * (1.0 - zk / delta) : 0.0 for zk in z]
            tau_u = m_d.surface.tau_u[CI]
            u_col = m_d.tile.physical[rng, mo.grid_params.vars["u"], 1]
            # XZ carries no `v`, so the drag work is `tau_u u` alone.
            expected = sum(wq .* gz .* (Psfc_e .- (tau_u .* u_col)))
            @test isapprox(I_re, expected; rtol = 1.0e-10)
        end
    end

    @testset ":exner_single -- one division by the surface Exner function" begin
        mktempdir() do tmp
            SST = 302.65
            m_f, m_d, mo, gp, z, col = fid_pair(tmp; fidelity = [:exner_single],
                                                state = false, num_cells_k = 25,
                                                SST = SST)
            cM = Scythe.MYNNConstants()
            p1 = Scythe.ref_pressure(m_f.ref_state)[1, 1]
            e1 = (p1 / Scythe.MYNN_P0)^cM.rcp
            F_sh = m_f.surface.F_sh[CI]
            F_q = m_f.surface.F_q[CI]
            @test F_sh != 0.0 && F_q != 0.0
            # `rmol` is proportional to `fltv = flt + flqv p608 th_sfc`, and on this
            # resting DRY column `cpm = cp` exactly and `rho_t(1)` cancels in the ratio,
            # so `th_sfc` is the only thing that moves.
            fltv_like(th) = F_sh / cM.cp + F_q * cM.p608 * th
            th_double = (SST / e1) / e1
            th_single = SST / e1
            @test th_single != th_double
            @test isapprox(m_d.mynn.rmol[CI],
                           m_f.mynn.rmol[CI] * fltv_like(th_single) /
                           fltv_like(th_double); rtol = 1.0e-8)
            @test m_d.mynn.rmol[CI] != m_f.mynn.rmol[CI]
        end
    end

    @testset ":rmol_sfc -- 1/L straight from the Monin-Obukhov surface layer" begin
        mktempdir() do tmp
            m_f, m_d, mo, gp, z, col = fid_pair(tmp; fidelity = [:rmol_sfc],
                                                num_cells_k = 25, sfc_stability = true,
                                                Cd = -1.0, fluxes = true)
            # `surface_exchange`'s own inverse Obukhov length, held by the surface group
            # (N2) -- this IS `sx.inv_L`, the value the deviation hands the closure.
            @test m_d.surface.active
            @test m_d.surface.inv_L[CI] != 0.0
            @test m_d.mynn.rmol[CI] === m_d.surface.inv_L[CI]
            # ...and it is NOT what the closure computes for itself.
            @test m_f.mynn.rmol[CI] != m_d.mynn.rmol[CI]
            @test m_f.mynn.rmol[CI] !== m_f.surface.inv_L[CI]
        end
    end

    @testset ":flux_clip -- the wrapper's limits, and only on what the closure sees" begin
        mktempdir() do tmp
            # A large enthalpy coefficient and a hot sea drive the sensible-heat flux far
            # past the wrapper's 1200 W/m^2. The column is left resting (no imposed
            # perturbation), so `rho_e = 0`, every `K` is an exact zero and the ONLY
            # thing the clip can move is the surface block itself.
            SST = 320.0
            m_f, m_d, mo, gp, z, col = fid_pair(tmp; fidelity = [:flux_clip],
                                                state = false, num_cells_k = 25,
                                                SST = SST, Ck = 0.5, Cd = -1.0,
                                                fluxes = true)
            kDim = gp.kDim
            rng = colrange(kDim)
            F_sh = m_f.surface.F_sh[CI]
            F_q = m_f.surface.F_q[CI]
            @test F_sh > 1200.0
            @test F_q > 5.0e-4
            # The COUNTERS run on both arms -- a run always reports whether the Fortran
            # would have clipped, whether or not it did.
            @test m_f.mynn.n_hfx_clip_col[CI] > 0
            @test m_d.mynn.n_hfx_clip_col[CI] > 0
            @test m_f.mynn.n_qfx_clip_col[CI] > 0
            @test m_d.mynn.n_qfx_clip_col[CI] > 0

            cM = Scythe.MYNNConstants()
            p1 = Scythe.ref_pressure(m_f.ref_state)[1, 1]
            e1 = (p1 / Scythe.MYNN_P0)^cM.rcp
            th = (SST / e1) / e1
            fltv_like(a, b) = a / cM.cp + b * cM.p608 * th
            F_sh_c = clamp(F_sh, Scythe.MYNN_HFX_MIN, Scythe.MYNN_HFX_MAX)
            F_q_c = clamp(F_q, Scythe.MYNN_QFX_MIN, Scythe.MYNN_QFX_MAX)
            @test isapprox(m_d.mynn.rmol[CI],
                           m_f.mynn.rmol[CI] * fltv_like(F_sh_c, F_q_c) /
                           fltv_like(F_sh, F_q); rtol = 1.0e-8)
            @test abs(m_d.mynn.rmol[CI]) < abs(m_f.mynn.rmol[CI])

            # The MODEL's surface delivery stays UNCLIPPED: with every K an exact zero
            # the whole E_t increment is `QDOT_V + F_sh g(z)` plus the water carry, so if
            # the clip had reached it this would not be an equality.
            @test maximum(m_d.mynn.K_h) == 0.0
            @test m_d.expdot_n[rng, 6] == m_f.expdot_n[rng, 6]
            @test maximum(abs.(m_f.expdot_n[rng, 6])) > 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 4. The D3 column energy identity, under every deviation
    # ──────────────────────────────────────────────
    @testset "column energy budget closes under every deviation" begin
        # The acceptance criterion of test_mynn_bl.jl's own D3 test, re-run once per
        # named deviation. A switch that opens the budget would make its own measurement
        # meaningless -- the treatment/control difference would be part scheme, part leak.
        mktempdir() do tmp
            for nm in DEV, (Cd, fluxes) in ((-1.0, true), (0.0, true))
                stab = nm === :rmol_sfc
                r, scale, = budget_residual(tmp; num_cells_k = 50, ts = 0.5, Cd = Cd,
                                            fluxes = fluxes, fidelity = [nm],
                                            sfc_stability = stab)
                @test isfinite(r)
                @test abs(r) <= 1.0e-12 * scale
                if !(abs(r) <= 1.0e-12 * scale)
                    @info "fidelity budget residual" nm Cd fluxes r scale rel = abs(r)/scale
                end
            end
        end
    end
end
