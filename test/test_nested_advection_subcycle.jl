using Test
using Scythe
using Springsteel
using LinearAlgebra

isdefined(@__MODULE__, :AdvectionNestPatch) || include("nested_advection_helpers.jl")

# Stage 4 of the grid-nesting plan: per-patch timesteps. The coarse patches
# take one step of Δt_c while the fine patch subcycles n_sub steps of
# Δt_c/n_sub, with the interface trio payloads linearly interpolated in time
# (lerp_payload!). Covers exact 2:1, a float ratio (requested fine ts not
# dividing the coarse ts), and the degenerate n_sub=1 case, which must be
# bit-identical to the same-Δt schedule.

function build_twoway_nest(ts_c, ts_f, nsteps_c, n_sub)
    coarseL = make_advection_patch(-60.0, -19.0, 41,
                                   DirichletBC(), NaturalBC();
                                   ts=ts_c, nsteps=nsteps_c)
    fine = make_advection_patch(-20.0, 20.0, 80,
                                Springsteel.CubicBSpline.R3X,
                                Springsteel.CubicBSpline.R3X;
                                ts=ts_f, nsteps=nsteps_c * n_sub)
    coarseR = make_advection_patch(19.0, 60.0, 41,
                                   NaturalBC(), NaturalBC();
                                   ts=ts_c, nsteps=nsteps_c)
    ifaceL = PatchInterface(coarseL.patch, fine.patch, :right, :left, :i;
                            is_stacked=true)
    ifaceR = PatchInterface(coarseR.patch, fine.patch, :left, :right, :i;
                            is_stacked=true)
    collarL_idx, collarL_x = collar_points(coarseL, -20.0, :right)
    collarR_idx, collarR_x = collar_points(coarseR, 20.0, :left)

    f0 = gaussian_pulse(-40.0, 5.0)
    set_advection_ic!(coarseL, f0)
    set_advection_ic!(coarseR, f0)
    set_advection_ic!(fine, f0)
    gridTransform!(coarseL.patch)
    gridTransform!(coarseR.patch)
    update_interface!(ifaceL)
    update_interface!(ifaceR)
    gridTransform!(fine.patch)

    return (coarseL, fine, coarseR, ifaceL, ifaceR,
            collarL_idx, collarL_x, collarR_idx, collarR_x)
end

@testset "Nested advection: subcycling" begin

    @testset "degenerate n_sub=1 is bit-identical to the same-Δt schedule" begin
        nsteps = 200
        # Reference: the stage-3 inline same-Δt loop
        nestA = build_twoway_nest(0.1, 0.1, nsteps, 1)
        (cL, fi, cR, ifL, ifR, cLi, cLx, cRi, cRx) = nestA
        for t in 1:nsteps
            inject_collar!(cL, fi, cLi, cLx)
            inject_collar!(cR, fi, cRi, cRx)
            step_patch!(cL, t); step_patch!(cR, t); step_patch!(fi, t)
            gridTransform!(cL.patch); gridTransform!(cR.patch)
            update_interface!(ifL); update_interface!(ifR)
            gridTransform!(fi.patch)
        end

        # Same setup through the subcycled driver with n_sub = 1
        nestB = build_twoway_nest(0.1, 0.1, nsteps, 1)
        run_twoway_subcycled!(nestB[1], nestB[2], nestB[3], nestB[4], nestB[5],
                              nestB[6], nestB[7], nestB[8], nestB[9], nsteps, 1)

        @test nestB[1].patch.physical == cL.patch.physical
        @test nestB[2].patch.physical == fi.patch.physical
        @test nestB[3].patch.physical == cR.patch.physical
    end

    @testset "2:1 subcycling (ts = 0.1 / 0.05 / 0.1)" begin
        nsteps_c = 800                 # 80 s
        nest = build_twoway_nest(0.1, 0.05, nsteps_c, 2)
        run_twoway_subcycled!(nest[1], nest[2], nest[3], nest[4], nest[5],
                              nest[6], nest[7], nest[8], nest[9], nsteps_c, 2)
        (coarseL, fine, coarseR) = nest[1:3]

        T = nsteps_c * 0.1
        fT = gaussian_pulse(-40.0 + NESTED_ADV_C0 * T, 5.0)
        err = l2_error(coarseR, fT)
        peak = maximum(coarseR.patch.physical[:, 1, 1])
        reflL = maximum(abs.(coarseL.patch.physical[:, 1, 1] .- fT.(coarseL.pts)))
        reflF = maximum(abs.(fine.patch.physical[:, 1, 1] .- fT.(fine.pts)))
        println("subcycle 2:1: coarseR L2 = $(round(err, digits=6)), peak = $(round(peak, digits=6)), residual L/F = $(round(reflL, sigdigits=3))/$(round(reflF, sigdigits=3))")
        @test err < 0.06
        @test 0.9 < peak < 1.1
        @test reflL < 0.03
        @test reflF < 0.03
    end

    @testset "float ratio (coarse ts 0.09, fine requested 0.05 → actual 0.045)" begin
        ts_c = 0.09
        n_sub = ceil(Int, ts_c / 0.05)         # 2
        ts_f = ts_c / n_sub                     # 0.045
        @test n_sub == 2 && isapprox(ts_f, 0.045)
        nsteps_c = ceil(Int, 80.0 / ts_c)       # 889 → T = 80.01 s
        nest = build_twoway_nest(ts_c, ts_f, nsteps_c, n_sub)
        run_twoway_subcycled!(nest[1], nest[2], nest[3], nest[4], nest[5],
                              nest[6], nest[7], nest[8], nest[9], nsteps_c, n_sub)
        (coarseL, fine, coarseR) = nest[1:3]

        T = nsteps_c * ts_c
        fT = gaussian_pulse(-40.0 + NESTED_ADV_C0 * T, 5.0)
        err = l2_error(coarseR, fT)
        peak = maximum(coarseR.patch.physical[:, 1, 1])
        reflL = maximum(abs.(coarseL.patch.physical[:, 1, 1] .- fT.(coarseL.pts)))
        reflF = maximum(abs.(fine.patch.physical[:, 1, 1] .- fT.(fine.pts)))
        println("subcycle float: coarseR L2 = $(round(err, digits=6)), peak = $(round(peak, digits=6)), residual L/F = $(round(reflL, sigdigits=3))/$(round(reflF, sigdigits=3))")
        @test err < 0.06
        @test 0.9 < peak < 1.1
        @test reflL < 0.03
        @test reflF < 0.03
    end
end
