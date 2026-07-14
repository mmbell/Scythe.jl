using Test
using Scythe
using Springsteel
using LinearAlgebra

isdefined(@__MODULE__, :AdvectionNestPatch) || include("nested_advection_helpers.jl")

# Stage 3 of the grid-nesting plan: two-way coupling, same Δt. A Gaussian
# pulse advects coarse→fine→coarse across two interfaces. Coarse→fine is the
# R3X trio (as in the one-way test); fine→coarse is collar injection: each
# coarse patch extends one cell into the fine domain, and its physical values
# and derivative slices at the collar mish points are overwritten from the
# fine representation before every tendency step (DeMaria et al. 1992).

@testset "Nested advection: two-way coarse→fine→coarse" begin

    ts = 0.1
    nsteps = 800                    # 80 s → pulse travels −40 → +40
    sigma = 5.0
    f0 = gaussian_pulse(-40.0, sigma)
    fT = gaussian_pulse(-40.0 + NESTED_ADV_C0 * nsteps * ts, sigma)

    # Coarse left: nominal [−60,−20] + collar → [−60,−19], DX = 1.
    # The outer left edge is an inflow boundary (c > 0): it needs the
    # physical condition u = 0 — a free fit there is ill-posed and admits
    # the large-scale "infrared creep" of Ooyama (2002) §9. The collar
    # termination (right edge) is outflow → free.
    coarseL = make_advection_patch(-60.0, -19.0, 41,
                                   DirichletBC(), NaturalBC(); ts=ts, nsteps=nsteps)
    # Fine: [−20, 20], DX = 0.5, R3X both sides
    fine = make_advection_patch(-20.0, 20.0, 80,
                                Springsteel.CubicBSpline.R3X,
                                Springsteel.CubicBSpline.R3X; ts=ts, nsteps=nsteps)
    # Coarse right: nominal [20,60] + collar → [19,60], DX = 1
    coarseR = make_advection_patch(19.0, 60.0, 41,
                                   NaturalBC(), NaturalBC(); ts=ts, nsteps=nsteps)

    ifaceL = PatchInterface(coarseL.patch, fine.patch, :right, :left, :i;
                            is_stacked=true)
    ifaceR = PatchInterface(coarseR.patch, fine.patch, :left, :right, :i;
                            is_stacked=true)

    collarL_idx, collarL_x = collar_points(coarseL, -20.0, :right)
    collarR_idx, collarR_x = collar_points(coarseR, 20.0, :left)
    @test length(collarL_idx) == coarseL.mtile.model.grid_params.mubar
    @test length(collarR_idx) == coarseR.mtile.model.grid_params.mubar

    # Initial conditions + initial exchange
    set_advection_ic!(coarseL, f0)
    set_advection_ic!(coarseR, f0)
    set_advection_ic!(fine, f0)
    gridTransform!(coarseL.patch)
    gridTransform!(coarseR.patch)
    update_interface!(ifaceL)
    update_interface!(ifaceR)
    gridTransform!(fine.patch)

    # Conservation monitor, partitioned at the nominal interfaces
    total0 = patch_integral(coarseL; xmax=-20.0) +
             patch_integral(fine) +
             patch_integral(coarseR; xmin=20.0)

    for t in 1:nsteps
        # Fine→coarse: inject the fine representation at the collar points
        # before the coarse tendency computations (fields + derivatives)
        inject_collar!(coarseL, fine, collarL_idx, collarL_x)
        inject_collar!(coarseR, fine, collarR_idx, collarR_x)

        step_patch!(coarseL, t)
        step_patch!(coarseR, t)
        step_patch!(fine, t)

        gridTransform!(coarseL.patch)
        gridTransform!(coarseR.patch)
        update_interface!(ifaceL)
        update_interface!(ifaceR)
        gridTransform!(fine.patch)
    end

    # (a) Pulse arrived intact in the right coarse patch
    err_analytic = l2_error(coarseR, fT)
    println("two-way: coarseR L2 vs analytic = $(round(err_analytic, digits=6))")
    @test err_analytic < 0.05

    # (b) Peak preserved after two interface crossings
    peak = maximum(coarseR.patch.physical[:, 1, 1])
    println("two-way: peak = $(round(peak, digits=6)) at x=$(round(coarseR.pts[argmax(coarseR.patch.physical[:, 1, 1])], digits=2))")
    @test 0.9 < peak < 1.1

    # (c) Reflection: residual left behind on the upstream patches
    reflL = maximum(abs.(coarseL.patch.physical[:, 1, 1] .- fT.(coarseL.pts)))
    reflF = maximum(abs.(fine.patch.physical[:, 1, 1] .- fT.(fine.pts)))
    println("two-way: residual coarseL = $(round(reflL, sigdigits=3)), fine = $(round(reflF, sigdigits=3))")
    @test reflL < 0.03
    @test reflF < 0.03

    # (d) Loose conservation monitor (state exchange is not flux-conservative;
    # the pulse integral should still be tracked to ~1%)
    totalT = patch_integral(coarseL; xmax=-20.0) +
             patch_integral(fine) +
             patch_integral(coarseR; xmin=20.0)
    drift = abs(totalT - total0) / abs(total0)
    println("two-way: integral drift = $(round(drift, sigdigits=3))")
    @test drift < 0.01
end
