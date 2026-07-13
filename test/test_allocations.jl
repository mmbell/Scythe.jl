using Test
using Scythe
using Springsteel
using SparseArrays

using Scythe: createModelTile, moist_compressible_XZ, diffusion_timestep_mc

# Guards for the allocation / type-stability refactor.
#
# `ModelTile` used to declare `tile::AbstractGrid` and its state arrays as `Array{Float64}`
# (which is `Array{Float64,N} where N` — an ABSTRACT type). That erased the concrete grid
# type Springsteel hands us, so `mtile.tile.physical` inferred as `Any` and every view and
# broadcast in the per-column equation-set bodies boxed. A 900 s straka93 run made 12.6e9
# pool allocations and killed the worker inside `gc_sweep_pool`.
#
# These tests are the tripwire that stops the abstract declarations creeping back.

@testset "allocations / type stability" begin

    # Build a small moist_compressible tile on the RiRk (B-spline vertical) grid — the
    # configuration that was crashing.
    function build_mc_tile()
        vars = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r"]
        scalar_bc = Dict(v => NeumannBC() for v in vars)
        bc_side = merge(scalar_bc, Dict("u" => DirichletBC()))
        bc_topbot = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(
            geometry = "RiRk",
            iMin = 0.0, iMax = 25.6e3, num_cells_i = 16,
            kMin = 0.0, kMax = 6.4e3, num_cells_k = 8,
            BCL = bc_side, BCR = bc_side, BCB = bc_topbot, BCT = bc_topbot,
            vars = Dict(v => i for (i, v) in enumerate(vars)))

        outdir = mktempdir()
        model = ModelParameters(
            ts = 0.0625,
            equation_set = "moist_compressible_XZ",
            output_dir = outdir * "/",
            ref_state_file = joinpath(outdir, "ref.csv"),
            grid_params = gp,
            physical_params = Dict(:Khdiff => 75.0, :Kvdiff => 75.0, :Kv_mudiff => 0.0,
                                   :tau_qss => 10.0, :alpha => 0.0,
                                   :z_damp => 12.8e3),
            options = Dict(:semiimplicit => true, :exact_reference_state => true,
                           :precipitation => false, :vertical_mixing => false))

        patch = createGrid(model.grid_params)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = model.grid_params.kDim
        z = gridpoints[1:kDim, 2]

        # Dry adiabat, matching the analytic base used by the mc benchmarks.
        exner = @. 1.0 - (Scythe.gravity * z) / (Scythe.Cpd * 300.0)
        pbar = @. 100000.0 * exner^(Scythe.Cpd / Scythe.Rd)
        Tbar = 300.0 .* exner
        Scythe.write_exact_ref_mc(model.ref_state_file, z, pbar,
            pbar ./ (Scythe.Rd .* Tbar), zeros(kDim), zeros(kDim))

        return createModelTile(patch, patch, model, spzeros(1, 1)), kDim
    end

    mtile, kDim = build_mc_tile()
    MT = typeof(mtile)

    @testset "every ModelTile field is concretely typed" begin
        # Check the whole struct, not a hand-picked subset: any new abstract field trips this.
        # NOTE: must use `typeof(mtile)`, not the `ModelTile` UnionAll — `fieldtype` on the
        # UnionAll returns each parameter's upper bound, which is abstract by construction.
        for f in fieldnames(MT)
            @test isconcretetype(fieldtype(MT, f))
        end

        # The critical one: erasing this is what caused the crash.
        @test fieldtype(MT, :tile) === RiRk_Grid
        @test isconcretetype(fieldtype(MT, :tile))
    end

    @testset "splineBuffer is gone" begin
        # It was allocated in createModelTile and never read anywhere.
        @test !hasfield(MT, :splineBuffer)
    end

    @testset "tilepoints keeps its rank parameter" begin
        # `getGridpoints` returns a Vector for the 1-D geometries and a Matrix otherwise,
        # so this field must NOT be pinned to `Matrix{Float64}`.
        @test fieldtype(MT, :tilepoints) === Matrix{Float64}   # 2-D on RiRk

        gp1 = GridParameters(geometry = "R", iMin = 0.0, iMax = 1.0, num_cells_i = 8,
            vars = Dict("u" => 1),
            BCL = Dict("u" => NeumannBC()), BCR = Dict("u" => NeumannBC()))
        m1 = ModelParameters(ts = 0.1, equation_set = "LinearAdvection1D", grid_params = gp1,
            physical_params = Dict(:c_0 => 1.0, :K => 0.0),
            options = Dict(:semiimplicit => false, :exact_reference_state => false))
        p1 = createGrid(m1.grid_params)
        mt1 = createModelTile(p1, p1, m1, spzeros(1, 1))
        @test fieldtype(typeof(mt1), :tilepoints) === Vector{Float64}   # 1-D on R
        @test isconcretetype(fieldtype(typeof(mt1), :tilepoints))
    end

    @testset "mc_diffusion_matrices stays concrete despite mixed factorization types" begin
        # The ten entries are NOT all the same concrete type — differing BCs flip
        # `factorize`'s symmetry detection, so `u` is a BunchKaufman while `w` is an LU on
        # RiRk. A Dict would have to widen to the abstract `Factorization` join; a
        # NamedTuple is concrete AND heterogeneous.
        MC = fieldtype(MT, :mc_diffusion_matrices)
        @test MC <: NamedTuple
        @test isconcretetype(MC)
        @test Set(fieldnames(MC)) ==
            Set((:u, :u_first, :w, :w_first, :heat, :heat_first,
                 :water, :water_first, :water_r, :water_r_first))
        # The retrieved reference diagnostics for the moist diffusion are concrete too
        @test isconcretetype(fieldtype(MT, :mc_ref_diag))
    end

    @testset "ModelParameters is concretely typed" begin
        for f in fieldnames(ModelParameters)
            @test isconcretetype(fieldtype(ModelParameters, f))
        end
        @test fieldtype(ModelParameters, :equation_set) === String
        @test fieldtype(ModelParameters, :physical_params) === Dict{Symbol,Float64}
    end

    @testset "scratch columns replace the per-column deepcopy" begin
        # The equation sets used to deepcopy a fresh vertical column out of the basis for
        # every variable, every column, every timestep — 93 allocations each, and 60% of all
        # per-column allocations. They now borrow a persistent per-thread column.
        @test isconcretetype(fieldtype(MT, :scratch_columns))
        @test size(mtile.scratch_columns) == (Threads.maxthreadid(), 8)

        # Distinct object per (thread, variable): `semiimplicit_adjustment_p` holds the p- and
        # w-columns live simultaneously (p_nstar aliases the p-column's uMish), so handing it
        # the same object twice would silently corrupt the pressure update.
        @test mtile.scratch_columns[1, 1] !== mtile.scratch_columns[1, 5]

        # No equation set may reintroduce the per-column deepcopy.
        @test !occursin("deepcopy(mtile.tile.kbasis",
            read(joinpath(@__DIR__, "..", "src", "moist_compressible.jl"), String))
    end

    @testset "threaded column loop matches serial (scratch columns are not shared)" begin
        # The scratch columns are owned per-thread and handed out by threadid(), which is only
        # a valid owner tag because the column loop is `@threads :static`. If that assumption
        # ever breaks — or if a scratch column gets shared across columns — two columns would
        # scribble on one work buffer and the result would diverge from serial, and would vary
        # run to run.
        #
        # Needs a multithreaded Julia to mean anything: run the suite with `julia --threads=8`.
        if Threads.nthreads() < 2
            @info "Skipping threaded scratch-column race check (Threads.nthreads() == 1); " *
                  "run the suite with --threads=8 to exercise it"
        else
            seed = zeros(size(mtile.tile.physical))
            for i in axes(seed,1), v in axes(seed,2), d in axes(seed,3)
                seed[i,v,d] = 1.0e-3 * sin(0.7i + 1.3v + 2.1d)
            end

            function advance_all(threaded::Bool)
                mt = build_mc_tile()[1]
                mt.tile.physical .= seed
                ncols = Springsteel.num_columns(mt.tile)
                for t in 1:3
                    if threaded
                        Threads.@threads :static for c in 1:ncols
                            Scythe.advance_column(mt, c, t)
                        end
                    else
                        for c in 1:ncols
                            Scythe.advance_column(mt, c, t)
                        end
                    end
                end
                return copy(mt.var_np1)
            end

            serial = advance_all(false)
            @test advance_all(true) == serial      # bit-identical, not just close
            @test advance_all(true) == serial      # and deterministic across runs
        end
    end

    @testset "per-column allocation ceilings" begin
        # Regression tripwire, not a target. Measured after the refactor: moist_compressible_XZ
        # 7 allocations per column call (down from 1447) and diffusion_timestep_mc exactly 0
        # (down from 249). The 7 that remain are the allocating 1-arg Ixtransform/Ixxtransform
        # in Springsteel plus the dynamic equation-set dispatch.
        #
        # The ceilings leave headroom for churn but trip immediately on the mistakes that
        # actually happened here: an abstract ModelTile field, a per-column deepcopy, a
        # loop-invariant ref_*(...)[:,N] copy, or a broadcast that allocates a fresh column
        # instead of writing into mc_scratch. Each of those costs tens to hundreds of allocs.
        moist_compressible_XZ(mtile, 1, kDim, 2)      # compile
        diffusion_timestep_mc(mtile, 1, kDim, 2)

        @test (@allocations moist_compressible_XZ(mtile, 1, kDim, 2)) < 25
        @test (@allocations diffusion_timestep_mc(mtile, 1, kDim, 2)) < 10
    end

    @testset "mc scratch slots are unique and cover every temporary" begin
        # A NamedTuple cannot hold duplicate names, so construction itself is the guard against
        # two live temporaries silently sharing one buffer. Assert the shape anyway, and that
        # the three functions' slots stay namespaced apart.
        @test length(unique(Scythe.MC_SCRATCH_SLOTS)) == length(Scythe.MC_SCRATCH_SLOTS)
        @test isconcretetype(eltype(mtile.mc_scratch))
        @test length(mtile.mc_scratch) == Threads.maxthreadid()
        @test all(length(v) == kDim for v in mtile.mc_scratch[1])
    end
end
