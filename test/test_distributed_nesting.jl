using Test
using Scythe
using Springsteel
using Distributed
using LinearAlgebra

isdefined(@__MODULE__, :AdvectionNestPatch) || include("nested_advection_helpers.jl")

# Stage 6 of the grid-nesting plan: integrate_nested_model — per-patch worker
# groups exchanging trio payloads (down) and collar evaluations (up) over
# RemoteChannels. Verified against the in-process subcycled driver: with the
# same ICs and schedule, the distributed trajectory must agree to near
# machine precision.

@testset "Distributed nested integration" begin

    added = addprocs(4; exeflags="--project=$(Base.active_project())")
    @everywhere added using Springsteel
    @everywhere added using Scythe

    try
        sigma = 5.0
        f0 = gaussian_pulse(-30.0, sigma)
        nsteps_c = 300                        # 30 s → pulse −30 → 0 (fine center)
        T = nsteps_c * 0.1
        fT = gaussian_pulse(-30.0 + NESTED_ADV_C0 * T, sigma)

        function make_nest(tmp; workers_per_patch)
            base = ModelParameters(
                ts = 0.1,
                integration_time = T,
                output_interval = T,
                equation_set = "LinearAdvection1D",
                initial_conditions = joinpath(tmp, "adv_ics.csv"),
                output_dir = joinpath(tmp, "out"),
                grid_params = GridParameters(
                    geometry = "R",
                    num_cells = 10, iMin = 0.0, iMax = 10.0,   # placeholder
                    BCL = Dict("u" => DirichletBC()),
                    BCR = Dict("u" => NaturalBC()),
                    vars = Dict("u" => 1)),
                physical_params = Dict(:c_0 => NESTED_ADV_C0, :K => NESTED_ADV_K),
            )
            return NestedModelParameters(
                boundaries = [-60.0, -20.0, 20.0, 60.0],
                num_cells = [40, 80, 40],
                ts = [0.1, 0.05, 0.1],
                workers_per_patch = workers_per_patch,
                base = base)
        end

        function write_nest_ics(models)
            for m in models
                g = createGrid(m.grid_params)
                pts = vec(getGridpoints(g))
                open(m.initial_conditions, "w") do io
                    println(io, "u")
                    for x in pts
                        println(io, f0(x))
                    end
                end
            end
        end

        # In-process reference trajectory (same ICs, same schedule)
        function run_inprocess()
            coarseL = make_advection_patch(-60.0, -19.0, 41,
                                           DirichletBC(), NaturalBC(); ts=0.1, nsteps=nsteps_c)
            fine = make_advection_patch(-20.0, 20.0, 80,
                                        Springsteel.CubicBSpline.R3X,
                                        Springsteel.CubicBSpline.R3X; ts=0.05, nsteps=2nsteps_c)
            coarseR = make_advection_patch(19.0, 60.0, 41,
                                           NaturalBC(), NaturalBC(); ts=0.1, nsteps=nsteps_c)
            ifaceL = PatchInterface(coarseL.patch, fine.patch, :right, :left, :i; is_stacked=true)
            ifaceR = PatchInterface(coarseR.patch, fine.patch, :left, :right, :i; is_stacked=true)
            cLi, cLx = collar_points(coarseL, -20.0, :right)
            cRi, cRx = collar_points(coarseR, 20.0, :left)
            set_advection_ic!(coarseL, f0)
            set_advection_ic!(coarseR, f0)
            set_advection_ic!(fine, f0)
            gridTransform!(coarseL.patch)
            gridTransform!(coarseR.patch)
            update_interface!(ifaceL)
            update_interface!(ifaceR)
            gridTransform!(fine.patch)
            run_twoway_subcycled!(coarseL, fine, coarseR, ifaceL, ifaceR,
                                  cLi, cLx, cRi, cRx, nsteps_c, 2)
            return coarseL, fine, coarseR
        end
        ipL, ipF, ipR = run_inprocess()

        rel_l2(a, b) = norm(a .- b) / max(norm(b), eps())

        @testset "1 worker per patch matches in-process trajectory" begin
            tmp = mktempdir()
            nest = make_nest(tmp; workers_per_patch=[1, 1, 1])
            models, _ = build_nest(nest)
            write_nest_ics(models)

            _, _, groups = integrate_nested_model(nest)
            finals = [Scythe.get_val_from(groups[i][1], :(patch.physical)) for i in 1:3]

            errs = [rel_l2(finals[1][:, 1, 1], ipL.patch.physical[:, 1, 1]),
                    rel_l2(finals[2][:, 1, 1], ipF.patch.physical[:, 1, 1]),
                    rel_l2(finals[3][:, 1, 1], ipR.patch.physical[:, 1, 1])]
            println("distributed nesting [1,1,1]: rel L2 vs in-process = $(round.(errs, sigdigits=3))")
            @test all(errs .< 1e-10)

            # Physical sanity on the distributed result itself
            fine_pts = vec(getGridpoints(createGrid(models[2].grid_params)))
            l2a = norm(finals[2][:, 1, 1] .- fT.(fine_pts)) / norm(fT.(fine_pts))
            @test l2a < 0.05
            @test 0.9 < maximum(finals[2][:, 1, 1]) < 1.1

            # Per-nest output dirs written
            for i in 1:3
                @test isfile(joinpath(models[i].output_dir, "scythe_out.log"))
            end
        end

        @testset "2-worker parent group (tiled injection + payload apply)" begin
            tmp = mktempdir()
            nest = make_nest(tmp; workers_per_patch=[2, 1, 1])
            models, _ = build_nest(nest)
            write_nest_ics(models)

            _, _, groups = integrate_nested_model(nest)
            finals = [Scythe.get_val_from(groups[i][1], :(patch.physical)) for i in 1:3]

            errs = [rel_l2(finals[1][:, 1, 1], ipL.patch.physical[:, 1, 1]),
                    rel_l2(finals[2][:, 1, 1], ipF.patch.physical[:, 1, 1]),
                    rel_l2(finals[3][:, 1, 1], ipR.patch.physical[:, 1, 1])]
            println("distributed nesting [2,1,1]: rel L2 vs in-process = $(round.(errs, sigdigits=3))")
            @test all(errs .< 1e-8)
        end
    finally
        rmprocs(added)
    end
end
