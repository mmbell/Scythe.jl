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
                                   :Prandtl => 1.0, :tau_qss => 10.0, :alpha => 0.0,
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
        # The six entries are NOT all the same concrete type — differing BCs flip
        # `factorize`'s symmetry detection, so `u` is a BunchKaufman while `w` is an LU on
        # RiRk. A Dict would have to widen to the abstract `Factorization` join; a
        # NamedTuple is concrete AND heterogeneous.
        MC = fieldtype(MT, :mc_diffusion_matrices)
        @test MC <: NamedTuple
        @test isconcretetype(MC)
        @test Set(fieldnames(MC)) ==
            Set((:u, :u_first, :w, :w_first, :heat, :heat_first))
    end

    @testset "ModelParameters is concretely typed" begin
        for f in fieldnames(ModelParameters)
            @test isconcretetype(fieldtype(ModelParameters, f))
        end
        @test fieldtype(ModelParameters, :equation_set) === String
        @test fieldtype(ModelParameters, :physical_params) === Dict{Symbol,Float64}
    end

    @testset "per-column allocation ceilings" begin
        # Regression tripwire, not a target. Measured after the concrete-typing refactor:
        # moist_compressible_XZ 619 allocs, diffusion_timestep_mc 179 allocs per column
        # call (down from 1447 / 249). The ceilings sit ~40% above those so that normal
        # churn does not trip them, but a reintroduced abstract field — which roughly
        # doubles the count — does.
        moist_compressible_XZ(mtile, 1, kDim, 2)      # compile
        diffusion_timestep_mc(mtile, 1, kDim, 2)

        @test (@allocations moist_compressible_XZ(mtile, 1, kDim, 2)) < 900
        @test (@allocations diffusion_timestep_mc(mtile, 1, kDim, 2)) < 260
    end
end
