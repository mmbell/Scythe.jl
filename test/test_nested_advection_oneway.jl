using Test
using Scythe
using Springsteel
using LinearAlgebra

include("nested_advection_helpers.jl")

# Stage 2 of the grid-nesting plan: one-way coarse→fine coupling, same Δt.
# A Gaussian pulse advects from a coarse patch (DX=1) into a fine patch
# (DX=0.5) through an R3X interface at x=0. The coarse patch carries a
# one-cell collar past the interface so the extraction trio is interior.

@testset "Nested advection: one-way coarse→fine" begin

    ts = 0.1
    nsteps = 400                    # 40 s → pulse travels −20 → +20
    sigma = 5.0
    f0 = gaussian_pulse(-20.0, sigma)
    fT = gaussian_pulse(-20.0 + NESTED_ADV_C0 * nsteps * ts, sigma)

    # Coarse: nominal [−50, 0] + 1-cell collar → [−50, 1], DX = 1
    coarse = make_advection_patch(-50.0, 1.0, 51,
                                  NaturalBC(), NaturalBC(); ts=ts, nsteps=nsteps)
    # Fine: [0, 50], DX = 0.5, R3X on the interface (left) side
    fine = make_advection_patch(0.0, 50.0, 100,
                                Springsteel.CubicBSpline.R3X, NaturalBC();
                                ts=ts, nsteps=nsteps)

    iface = PatchInterface(coarse.patch, fine.patch, :right, :left, :i;
                           is_stacked=true)

    set_advection_ic!(coarse, f0)
    gridTransform!(coarse.patch)          # coarse .a for the initial exchange
    set_advection_ic!(fine, f0)
    update_interface!(iface)              # fine ahat before its first fit
    gridTransform!(fine.patch)

    for t in 1:nsteps
        step_patch!(coarse, t)
        step_patch!(fine, t)
        gridTransform!(coarse.patch)   # coarse .a fresh at new time level
        update_interface!(iface)       # coarse trio → fine ahat
        gridTransform!(fine.patch)     # fine reconstruction honors new ahat
    end

    # (a) Fine solution vs analytic shifted Gaussian
    err_analytic = l2_error(fine, fT)
    println("one-way: fine L2 vs analytic = $(round(err_analytic, digits=6))")
    @test err_analytic < 0.05

    # (b) Fine solution vs uniform-fine single-grid reference
    ref = make_advection_patch(-50.0, 50.0, 200,
                               NaturalBC(), NaturalBC(); ts=ts, nsteps=nsteps)
    set_advection_ic!(ref, f0)
    gridTransform!(ref.patch)
    for t in 1:nsteps
        step_patch!(ref, t)
        gridTransform!(ref.patch)
    end
    # Fine-patch mish points coincide with the reference's on [0, 50]
    ref_lookup = Dict(round(x, digits=9) => i for (i, x) in enumerate(ref.pts))
    num = 0.0; den = 0.0
    for (i, x) in enumerate(fine.pts)
        j = ref_lookup[round(x, digits=9)]
        r = ref.patch.physical[j, 1, 1]
        num += (fine.patch.physical[i, 1, 1] - r)^2
        den += r^2
    end
    err_ref = sqrt(num) / sqrt(den)
    println("one-way: fine L2 vs uniform-fine reference = $(round(err_ref, digits=6))")
    @test err_ref < 0.02

    # (c) Peak preserved
    peak = maximum(fine.patch.physical[:, 1, 1])
    println("one-way: peak = $(round(peak, digits=6)) at x=$(round(fine.pts[argmax(fine.patch.physical[:, 1, 1])], digits=2))")
    @test 0.9 < peak < 1.1

    # (d) Reflection on the coarse patch after the pulse has left.
    # Analytic residual at x=0 is exp(-20²/50) ≈ 3e-4; anything much larger
    # on the nominal coarse region is reflection/noise.
    nominal = findall(x -> x <= 0.0, coarse.pts)
    refl = maximum(abs.(coarse.patch.physical[nominal, 1, 1] .-
                        fT.(coarse.pts[nominal])))
    collar_idx, _ = collar_points(coarse, 0.0, :right)
    collar_dev = maximum(abs.(coarse.patch.physical[collar_idx, 1, 1] .-
                              fT.(coarse.pts[collar_idx])))
    println("one-way: coarse nominal-region reflection = $(round(refl, sigdigits=3)), collar deviation = $(round(collar_dev, sigdigits=3))")
    @test refl < 0.02
end
