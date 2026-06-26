using Test
using Scythe
using Springsteel

# Validates that the new physical-density Springsteel reference builder, viewed back as
# a legacy ReferenceState, reproduces Scythe's original xi/mu sounding builder. This is
# the equivalence guarantee behind switching createModelTile onto the shared builder.

@testset "Reference state migration equivalence" begin

    function rz_grid(; num_cells=16, kDim=48, iMax=8000.0, kMax=6000.0)
        vars = Dict("s" => 1, "xi" => 2, "mu" => 3, "u" => 4, "w" => 5)
        bc = Dict(v => NeumannBC() for v in keys(vars))
        gp = GridParameters(geometry="RZ", num_cells=num_cells,
            iMin=0.0, iMax=iMax, kMin=0.0, kMax=kMax, kDim=kDim,
            BCL=bc, BCR=bc, BCB=bc, BCT=bc, vars=vars)
        return gp, createGrid(gp)
    end

    # Compare every legacy field between the original builder and the physical view.
    function compare(old::Scythe.ReferenceState, new::Scythe.ReferenceState; tol=1e-9)
        for f in (:sbar, :xibar, :rhobar, :mubar, :satbar)
            o = getfield(old, f); n = getfield(new, f)
            denom = max(maximum(abs.(o)), 1.0)
            err = maximum(abs.(o .- n)) / denom
            @test err < tol
        end
        @test isapprox(old.Pxi_bar, new.Pxi_bar; rtol=tol)
    end

    @testset "Dry sounding" begin
        mktempdir() do tmp
            gp, patch = rz_grid()
            sounding = Scythe.write_dry_sounding(joinpath(tmp, "dry.ref"); theta=300.0, zmax=6000.0)
            z = getGridpoints(patch)[1:gp.kDim, 2]

            model = ModelParameters(ts=0.1, equation_set="Euler_test",
                ref_state_file=sounding, grid_params=gp, physical_params=Dict(:K => 0.0))
            old = Scythe.calculate_reference_state(model, z, Scythe.reference_column(patch, gp))

            phys = Springsteel.calculate_reference_state(sounding, z,
                Scythe.reference_column(patch, gp); moisture=true)
            new = Scythe.legacy_reference_view(phys, Scythe.reference_column(patch, gp))

            compare(old, new)
        end
    end

    @testset "Moist sounding" begin
        mktempdir() do tmp
            gp, patch = rz_grid()
            # Simple moist sounding: theta=300, q_v decreasing with height
            sounding = joinpath(tmp, "moist.ref")
            open(sounding, "w") do io
                println(io, "1000.0 300.0 12.0")
                for zi in 250.0:250.0:6000.0
                    qv = max(0.0, 12.0 - 1.5e-3 * zi)
                    println(io, "$(zi) 300.0 $(qv)")
                end
            end
            z = getGridpoints(patch)[1:gp.kDim, 2]

            model = ModelParameters(ts=0.1, equation_set="primitive_equation_XZ",
                ref_state_file=sounding, grid_params=gp, physical_params=Dict(:K => 0.0))
            old = Scythe.calculate_reference_state(model, z, Scythe.reference_column(patch, gp))

            phys = Springsteel.calculate_reference_state(sounding, z,
                Scythe.reference_column(patch, gp); moisture=true)
            new = Scythe.legacy_reference_view(phys, Scythe.reference_column(patch, gp))

            compare(old, new)
        end
    end
end
