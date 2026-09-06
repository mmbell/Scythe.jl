using Test
using Scythe
using Springsteel
using SparseArrays

using Scythe: createModelTile, moist_compressible_XZ, diffusion_timestep_mc, Twoway_PV_mixing

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
    function build_mc_tile(; extra_params = Dict{Symbol,Float64}(), precipitation = false,
                             geometry = "RiRk", extra_options = Dict{Symbol,Any}(),
                             equation_set = "moist_compressible_XZ")
        cyl = equation_set != "moist_compressible_XZ"
        # From mc_var_names, not a literal, so the transformed slot names come along: with
        # no transform declared it returns exactly the list this used to hardcode.
        vars = Scythe.mc_var_names(extra_options; cyl = cyl)
        scalar_bc = Dict(v => NeumannBC() for v in vars)
        bc_side = merge(scalar_bc, Dict("u" => DirichletBC()))
        bc_topbot = merge(scalar_bc, Dict("w" => DirichletBC()))
        # The axisym set reinterprets x as radius, so keep the domain off the axis;
        # the RLR grid runs from the axis out (its mish points exclude r = 0).
        # The spherical shell's i-coordinate is the colatitude [rad]
        iMin = equation_set == "moist_compressible_axisym" ? 100.0e3 :
               geometry == "SLR" ? pi/4 - 0.035 : 0.0
        iSpan = geometry == "SLR" ? 0.07 : 25.6e3
        wavenumbers = geometry in ("RLR", "SLR") ? Dict(v => 2 for v in vars) :
                                                   Dict{String,Int64}()
        # The 3D Cartesian box needs the y direction: v (the y-wind) is the
        # normal component at the y walls
        bc_y = merge(scalar_bc, Dict("v" => DirichletBC()))
        ykw = geometry == "RRR" ?
              (jMin = 0.0, jMax = 25.6e3, BCU = bc_y, BCD = bc_y) : (;)
        gp = GridParameters(;
            geometry = geometry,
            iMin = iMin, iMax = iMin + iSpan, num_cells_i = 16,
            kMin = 0.0, kMax = 6.4e3, num_cells_k = 8,
            max_wavenumber = wavenumbers,
            BCL = bc_side, BCR = bc_side, BCB = bc_topbot, BCT = bc_topbot,
            vars = Dict(v => i for (i, v) in enumerate(vars)),
            ykw...)

        outdir = mktempdir()
        model = ModelParameters(
            ts = 0.0625,
            equation_set = equation_set,
            output_dir = outdir * "/",
            ref_state_file = joinpath(outdir, "ref.csv"),
            grid_params = gp,
            physical_params = merge(Dict(:Khdiff => 75.0, :Kvdiff => 75.0, :Kv_mudiff => 0.0,
                                         :tau_qss => 10.0, :alpha => 0.0,
                                         :z_damp => 12.8e3), extra_params),
            options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                             :exact_reference_state => true,
                                             :precipitation => precipitation,
                                             :vertical_mixing => false),
                            extra_options))

        patch = createGrid(model.grid_params)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = model.grid_params.kDim
        z = gridpoints[1:kDim, end]     # z is the LAST gridpoint column (2D and 3D)

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
        # The fourteen entries are NOT all the same concrete type — differing BCs flip
        # `factorize`'s symmetry detection, so `u` is a BunchKaufman while `w` is an LU on
        # RiRk. A Dict would have to widen to the abstract `Factorization` join; a
        # NamedTuple is concrete AND heterogeneous.
        MC = fieldtype(MT, :mc_diffusion_matrices)
        @test MC <: NamedTuple
        @test isconcretetype(MC)
        # `water_v` is the prognostic vapor's own solve: its increment is solved rather
        # than implied from the other three (see `_diffusion_water_step!`).
        @test Set(fieldnames(MC)) ==
            Set((:u, :u_first, :w, :w_first, :heat, :heat_first,
                 :water, :water_first, :water_r, :water_r_first,
                 :water_c, :water_c_first, :water_v, :water_v_first))
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
        @test size(mtile.scratch_columns) == (Threads.maxthreadid(), 10)

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
        # The per-column hot path is ALLOCATION-FREE: moist_compressible_XZ and
        # diffusion_timestep_mc both make exactly 0 allocations per call (down from
        # 1447/249 pre-refactor; the last 7 were Springsteel's allocating 1-arg
        # Ixtransform/Ixxtransform, now called through the in-place forms with
        # per-thread scratch). Any nonzero count is a real hot-path allocation —
        # historically an abstract ModelTile field, a per-column deepcopy, a
        # loop-invariant ref_*(...)[:,N] copy, a broadcast into a fresh column
        # instead of mc_scratch, or a broadcast-into-view whose SubArray stops
        # eliding once the function grows (the dE_w add needed an explicit loop).
        moist_compressible_XZ(mtile, 1, kDim, 2)      # compile
        diffusion_timestep_mc(mtile, 1, kDim, 2)

        @test (@allocations moist_compressible_XZ(mtile, 1, kDim, 2)) == 0
        @test (@allocations diffusion_timestep_mc(mtile, 1, kDim, 2)) == 0
    end

    @testset "per-column allocations stay zero with rain and water diffusion" begin
        # The warm-rain microphysics (sedimentation column transforms) and the
        # water-species diffusion (FOUR extra vertical solves + fixed-T maps -- the vapor
        # gained one of its own when it became prognostic) are branches the base
        # configuration never runs; guard them separately.
        mtile_w, kDim_w = build_mc_tile(extra_params = Dict(:Kvdiff_water => 25.0,
                                                            :N_r => 1.0e-3),
                                        precipitation = true)
        moist_compressible_XZ(mtile_w, 1, kDim_w, 2)  # compile
        diffusion_timestep_mc(mtile_w, 1, kDim_w, 2)

        @test (@allocations moist_compressible_XZ(mtile_w, 1, kDim_w, 2)) == 0
        @test (@allocations diffusion_timestep_mc(mtile_w, 1, kDim_w, 2)) == 0
    end

    @testset "per-column allocations stay zero with the Marshall-Palmer DSD" begin
        # N_0 > 0 routes the rain channel through invtau_rain_mp (mp_slope +
        # f_ventilation_mp); the keyword call and the MP branch must not box.
        mtile_mp, kDim_mp = build_mc_tile(extra_params = Dict(:Kvdiff_water => 25.0,
                                                              :N_r => 1.0e-3,
                                                              :N_0 => 8.0e6),
                                          precipitation = true)
        moist_compressible_XZ(mtile_mp, 1, kDim_mp, 2)  # compile
        diffusion_timestep_mc(mtile_mp, 1, kDim_mp, 2)

        @test (@allocations moist_compressible_XZ(mtile_mp, 1, kDim_mp, 2)) == 0
        @test (@allocations diffusion_timestep_mc(mtile_mp, 1, kDim_mp, 2)) == 0
    end

    @testset "per-column allocations stay zero with two-moment rain" begin
        # `options[:rain_moments] = 2` APPENDS a prognostic slot whose index is resolved by
        # NAME. That resolution is a `Dict{String,Int}` hash, and the driver runs once per
        # COLUMN — so it is done once, at tile creation, into the concrete `MCSlots` field.
        # If it ever creeps back into `mc_driver!` this test is what catches it, along with
        # the appended slot's own views, scratch columns and flux transform.
        mtile_2m, kDim_2m = build_mc_tile(extra_params = Dict(:N_r => 1.0e-3),
                                          precipitation = true,
                                          extra_options = Dict{Symbol,Any}(
                                              :rain_moments => 2))
        @test mtile_2m.mc_slots.rho_v == 10        # the unconditional vapor slot
        @test mtile_2m.mc_slots.n_r == 11          # appended after it
        moist_compressible_XZ(mtile_2m, 1, kDim_2m, 2)  # compile
        diffusion_timestep_mc(mtile_2m, 1, kDim_2m, 2)

        @test (@allocations moist_compressible_XZ(mtile_2m, 1, kDim_2m, 2)) == 0
        @test (@allocations diffusion_timestep_mc(mtile_2m, 1, kDim_2m, 2)) == 0

        # ...and with the number's own control-variable transform on (an extra recovery
        # and Jacobian per column), and on the cylinder where the slot moves to 11.
        mtile_2t, kDim_2t = build_mc_tile(equation_set = "moist_compressible_axisym",
                                          extra_params = Dict(:N_r => 1.0e-3,
                                                              :mu_rain_n => 1.0),
                                          precipitation = true,
                                          extra_options = Dict{Symbol,Any}(
                                              :rain_moments => 2,
                                              :rain_number_transform => :bhyp))
        @test mtile_2t.mc_slots.rho_v == 11        # after v on the cylinder
        @test mtile_2t.mc_slots.n_r == 12
        Scythe.moist_compressible_axisym(mtile_2t, 1, kDim_2t, 2)  # compile
        @test (@allocations Scythe.moist_compressible_axisym(mtile_2t, 1, kDim_2t, 2)) == 0
    end

    @testset "per-column allocations stay zero with ice microphysics" begin
        # Twelve more appended slots, each with its own views, its own control-variable
        # recovery, its own scratch columns and its own sedimentation flux transform — the
        # widest the per-column path gets. Every index is resolved by NAME, and every one of
        # those resolutions is a `Dict{String,Int}` hash that must have happened once, at
        # tile creation, into `MCSlots`. This is what catches it if one creeps back in.
        ice_opts = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael)
        mtile_ice, kDim_ice = build_mc_tile(extra_params = Dict(:N_r => 1.0e-3),
                                            precipitation = true,
                                            extra_options = ice_opts)
        @test Scythe.ice_registered(mtile_ice.mc_slots)
        @test Scythe.ice_slots(mtile_ice.mc_slots, 1) == (12, 13, 14, 15)
        @test Scythe.ice_slots(mtile_ice.mc_slots, 3) == (20, 21, 22, 23)
        moist_compressible_XZ(mtile_ice, 1, kDim_ice, 2)      # compile
        diffusion_timestep_mc(mtile_ice, 1, kDim_ice, 2)

        @test (@allocations moist_compressible_XZ(mtile_ice, 1, kDim_ice, 2)) == 0
        @test (@allocations diffusion_timestep_mc(mtile_ice, 1, kDim_ice, 2)) == 0

        # ...and with the ice control-variable transform on (twelve more recoveries and
        # Jacobians per column), on the cylinder where all twelve slots shift by one.
        mtile_it, kDim_it = build_mc_tile(equation_set = "moist_compressible_axisym",
                                          extra_params = Dict(:N_r => 1.0e-3,
                                                              :mu_rain_n => 1.0),
                                          precipitation = true,
                                          extra_options = merge(ice_opts,
                                              Dict{Symbol,Any}(:ice_transform => :bhyp)))
        @test Scythe.ice_slots(mtile_it.mc_slots, 1) == (13, 14, 15, 16)
        Scythe.moist_compressible_axisym(mtile_it, 1, kDim_it, 2)  # compile
        @test (@allocations Scythe.moist_compressible_axisym(mtile_it, 1, kDim_it, 2)) == 0

        # ...and with the `:local` population seeding, which is the one reconciliation
        # branch that walks the column instead of reading one gridpoint: three live-span
        # sweeps per column plus a bounded outward index walk per dead point, all on the
        # raw slots and with no workspace of their own.
        mtile_ls, kDim_ls = build_mc_tile(extra_params = Dict(:N_r => 1.0e-3),
                                          precipitation = true,
                                          extra_options = merge(ice_opts,
                                              Dict{Symbol,Any}(
                                                  :ice_population_seed => :local)))
        moist_compressible_XZ(mtile_ls, 1, kDim_ls, 2)        # compile
        @test (@allocations moist_compressible_XZ(mtile_ls, 1, kDim_ls, 2)) == 0

        # ...and with the ICE NUMBER REALIZATION on (Stage 1b): three more donor factors per
        # gridpoint, a second set of factors threaded into the aggregation kernel, and three
        # more scratch columns written. All scalars and pre-allocated columns — the kernel
        # gained keyword arguments, which must stay specialized rather than boxed.
        mtile_nr, kDim_nr = build_mc_tile(extra_params = Dict(:N_r => 1.0e-3),
                                          precipitation = true,
                                          extra_options = merge(ice_opts,
                                              Dict{Symbol,Any}(
                                                  :ice_number_realization => true)))
        moist_compressible_XZ(mtile_nr, 1, kDim_nr, 2)        # compile
        @test (@allocations moist_compressible_XZ(mtile_nr, 1, kDim_nr, 2)) == 0
    end

    @testset "per-column allocations stay zero on the axisymmetric cylinder" begin
        # The cylindrical trait path binds the extra v views and metric terms; the
        # trait dispatch must stay compile-time (no boxing) and the v machinery in
        # the RHS, sponge and diffusion solves must reuse mc_scratch like u/w.
        mtile_a, kDim_a = build_mc_tile(equation_set = "moist_compressible_axisym",
                                        extra_params = Dict(:f => 5.0e-5,
                                                            :alpha => 0.05,
                                                            :z_damp => 3.2e3))
        Scythe.moist_compressible_axisym(mtile_a, 1, kDim_a, 2)  # compile
        Scythe.diffusion_timestep_mc(mtile_a, 1, kDim_a, 2, Scythe.MCAxisymRZ())

        @test (@allocations Scythe.moist_compressible_axisym(mtile_a, 1, kDim_a, 2)) == 0
        @test (@allocations Scythe.diffusion_timestep_mc(mtile_a, 1, kDim_a, 2,
                                                         Scythe.MCAxisymRZ())) == 0
    end

    @testset "per-column allocations stay zero with the Louis BL + Smagorinsky" begin
        # The Louis boundary layer (flux-column transforms, Komori drag, the slot
        # apply loop) and the vector-K Smagorinsky call sites must not box or
        # allocate; guard the axisym trait path where the TC runs live.
        mtile_bl, kDim_bl = build_mc_tile(equation_set = "moist_compressible_axisym",
                                          extra_params = Dict(:f => 5.0e-5, :Cd => -1.0,
                                                              :Ls => 200.0, :K_min => 5.0,
                                                              :l_inf => 80.0, :Ck => 1.0e-3,
                                                              :SST => 301.15, :U_min => 1.0),
                                          extra_options = Dict{Symbol,Any}(:louis_bl => true,
                                                                           :surface_fluxes => true))
        Scythe.moist_compressible_axisym(mtile_bl, 1, kDim_bl, 2)  # compile
        @test (@allocations Scythe.moist_compressible_axisym(mtile_bl, 1, kDim_bl, 2)) == 0

        # And the RLR trait path (the production 3D geometry)
        mtile_blr, kDim_blr = build_mc_tile(geometry = "RLR",
                                            equation_set = "moist_compressible_RLR",
                                            extra_params = Dict(:f => 5.0e-5, :Cd => -1.0,
                                                                :Ls => 200.0, :K_min => 5.0,
                                                                :l_inf => 80.0, :Ck => 1.0e-3,
                                                                :SST => 301.15, :U_min => 1.0),
                                            extra_options = Dict{Symbol,Any}(:louis_bl => true,
                                                                             :surface_fluxes => true))
        Scythe.moist_compressible_RLR(mtile_blr, 1, kDim_blr, 2)  # compile
        @test (@allocations Scythe.moist_compressible_RLR(mtile_blr, 1, kDim_blr, 2)) == 0

        # WITH THE WATER TRANSFORMS ON. This is the arm the Louis-BL cloud fix is shaped
        # around: the perturbation density gradient is staged in mc_driver!, where
        # rho_cbar_z is already a live local, precisely so that a SubArray never escapes
        # into the @noinline callee. Do this the other way and it boxes once per column.
        # Khdiff_water is on here too, so the nu-space horizontal mixing is covered.
        mtile_ct, kDim_ct = build_mc_tile(equation_set = "moist_compressible_axisym",
                                          extra_params = Dict(:f => 5.0e-5, :Cd => -1.0,
                                                              :Ls => 200.0, :K_min => 5.0,
                                                              :l_inf => 80.0, :Ck => 1.0e-3,
                                                              :SST => 301.15, :U_min => 1.0,
                                                              :Khdiff_water => -1.0,
                                                              :Sc_t => 1.0),
                                          extra_options = Dict{Symbol,Any}(
                                              :louis_bl => true, :surface_fluxes => true,
                                              :condensate_transform => :bhyp,
                                              :rain_transform => :bhyp))
        Scythe.moist_compressible_axisym(mtile_ct, 1, kDim_ct, 2)  # compile
        @test (@allocations Scythe.moist_compressible_axisym(mtile_ct, 1, kDim_ct, 2)) == 0

        # AND EVERY SURFACE-LAYER ARM (stage S1b, src/mc_surface_layer.jl). The
        # SurfaceLayerParams struct carries a Symbol, so it is not isbits; it crosses the
        # @noinline boundary of mc_louis_bl! once per column and this is what proves the
        # compiler keeps it out of the heap. The :gfdl_v7 / :charnock / stability paths also
        # run log/exp/atan and a fixed-point loop inside that callee.
        for z0 in (:komori, :gfdl_v7, :charnock), stab in (false, true)
            mtile_sl, kDim_sl = build_mc_tile(equation_set = "moist_compressible_axisym",
                                              extra_params = Dict(:f => 5.0e-5, :Cd => -1.0,
                                                                  :l_inf => 80.0,
                                                                  :Ck => 1.0e-3,
                                                                  :SST => 301.15,
                                                                  :U_min => 1.0),
                                              extra_options = Dict{Symbol,Any}(
                                                  :louis_bl => true, :surface_fluxes => true,
                                                  :sfc_z0 => z0, :sfc_stability => stab))
            Scythe.moist_compressible_axisym(mtile_sl, 1, kDim_sl, 2)  # compile
            @test (@allocations Scythe.moist_compressible_axisym(mtile_sl, 1,
                                                                 kDim_sl, 2)) == 0
        end
    end

    @testset "per-column allocations stay zero on the 3D RLR cylinder" begin
        # The 7-slot layout (z at slots 6/7, raw λ at 4/5) and the azimuthal metric
        # terms; the column loop and vertical solves are identical to RiRk.
        mtile_r, kDim_r = build_mc_tile(geometry = "RLR",
                                        equation_set = "moist_compressible_RLR",
                                        extra_params = Dict(:f => 5.0e-5,
                                                            :alpha => 0.05,
                                                            :z_damp => 3.2e3),
                                        precipitation = true)
        Scythe.moist_compressible_RLR(mtile_r, 1, kDim_r, 2)  # compile
        Scythe.diffusion_timestep_mc(mtile_r, 1, kDim_r, 2, Scythe.MCCylindricalRLR())

        @test (@allocations Scythe.moist_compressible_RLR(mtile_r, 1, kDim_r, 2)) == 0
        @test (@allocations Scythe.diffusion_timestep_mc(mtile_r, 1, kDim_r, 2,
                                                         Scythe.MCCylindricalRLR())) == 0
    end

    @testset "per-column allocations stay zero on the 3D Cartesian box" begin
        # RRR: 7-slot layout with the full ∂y/∂yy at 4/5 and no metric terms.
        mtile_b, kDim_b = build_mc_tile(geometry = "RRR",
                                        equation_set = "moist_compressible_RRR",
                                        extra_params = Dict(:f => 5.0e-5,
                                                            :alpha => 0.05,
                                                            :z_damp => 3.2e3),
                                        precipitation = true)
        Scythe.moist_compressible_RRR(mtile_b, 1, kDim_b, 2)  # compile
        Scythe.diffusion_timestep_mc(mtile_b, 1, kDim_b, 2, Scythe.MCCartesianRRR())

        @test (@allocations Scythe.moist_compressible_RRR(mtile_b, 1, kDim_b, 2)) == 0
        @test (@allocations Scythe.diffusion_timestep_mc(mtile_b, 1, kDim_b, 2,
                                                         Scythe.MCCartesianRRR())) == 0
    end

    @testset "per-column allocations stay zero on the 3D spherical shell" begin
        # SLR: the metric handle is a (theta, a, Omega) NamedTuple and the
        # broadcasts carry sin/cos of the colatitude view — none of it may box.
        mtile_s, kDim_s = build_mc_tile(geometry = "SLR",
                                        equation_set = "moist_compressible_SLR",
                                        extra_params = Dict(:Omega => 7.292e-5,
                                                            :sphere_radius => 6.371e6,
                                                            :alpha => 0.05,
                                                            :z_damp => 3.2e3),
                                        precipitation = true)
        Scythe.moist_compressible_SLR(mtile_s, 1, kDim_s, 2)  # compile
        Scythe.diffusion_timestep_mc(mtile_s, 1, kDim_s, 2, Scythe.MCSphericalSLR())

        @test (@allocations Scythe.moist_compressible_SLR(mtile_s, 1, kDim_s, 2)) == 0
        @test (@allocations Scythe.diffusion_timestep_mc(mtile_s, 1, kDim_s, 2,
                                                         Scythe.MCSphericalSLR())) == 0
    end

    @testset "per-column allocations stay zero with the Rayleigh sponge active" begin
        # The sponge block only runs for alpha > 0 (the base configuration never
        # enters it) and must not allocate: `.+=` view broadcasts stopped eliding
        # at this function size (the dE_w lesson), so it is an explicit loop.
        mtile_s, kDim_s = build_mc_tile(extra_params = Dict(:alpha => 0.05,
                                                            :z_damp => 3.2e3))
        moist_compressible_XZ(mtile_s, 1, kDim_s, 2)  # compile

        @test (@allocations moist_compressible_XZ(mtile_s, 1, kDim_s, 2)) == 0
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

    # ------------------------------------------------------------------------------------------
    # Twoway_PV_mixing (RL shallow water). Same disease as the mc set, different geometry: this one
    # used to heap-allocate 13 FULL-TILE-LENGTH Vector{Float64} per call (9 `similar(r)`, `U`, and
    # the materializing `K_free`/`K_bl`/`w_` broadcasts). On an RL grid the whole tile is ONE
    # column, so that is ~19 MB of garbage per timestep at num_cells=100 — ~230 GB over a 10-hour
    # run, which is the memory build-up this refactor exists to kill.
    # ------------------------------------------------------------------------------------------

    """Small RL Twoway_PV_mixing tile with a nonzero max_wavenumber, so the azimuthal (_l, _ll)
    derivative slots the equation set reads are actually populated."""
    function build_pv_mixing_tile(; extra_params = Dict{Symbol,Float64}(),
                                    options = Dict{Symbol,Any}())
        vars = ["h", "u", "v", "ub", "vb", "wb"]
        gp = GridParameters(
            geometry = "RL",
            iMin = 0.0, iMax = 3.0e5, num_cells = 20,
            max_wavenumber = Dict(v => 8 for v in vars),
            BCL = Dict("h"  => Springsteel.CubicBSpline.R1T1,
                       "u"  => Springsteel.CubicBSpline.R1T0,
                       "v"  => Springsteel.CubicBSpline.R1T0,
                       "ub" => Springsteel.CubicBSpline.R1T0,
                       "vb" => Springsteel.CubicBSpline.R1T0,
                       "wb" => Springsteel.CubicBSpline.R1T1),
            BCR = Dict("h"  => Springsteel.CubicBSpline.R0,
                       "u"  => Springsteel.CubicBSpline.R1T1,
                       "v"  => Springsteel.CubicBSpline.R0,
                       "ub" => Springsteel.CubicBSpline.R1T1,
                       "vb" => Springsteel.CubicBSpline.R0,
                       "wb" => Springsteel.CubicBSpline.R0),
            vars = Dict(v => i for (i, v) in enumerate(vars)))

        model = ModelParameters(
            ts = 3.0,
            equation_set = "Twoway_PV_mixing",
            output_dir = mktempdir() * "/",
            grid_params = gp,
            physical_params = merge(Dict(:g => 9.81, :Ls_free => 500.0, :Ls_bl => 2000.0,
                                         :K_min_free => 0.0, :K_min_bl => 1000.0,
                                         :Cd => 2.4e-3, :Hfree => 2000.0, :Hb => 1000.0,
                                         :f => 5.0e-5, :S1 => 1.0e-5), extra_params),
            options = merge(Dict{Symbol,Any}(:semiimplicit => false,
                                             :exact_reference_state => false), options))

        patch = createGrid(model.grid_params)
        # A broad axisymmetric vortex, so the strain rates (and hence the Smagorinsky K) are
        # nonzero and every branch of the tendency does real work.
        gridpoints = Scythe.getGridpoints(patch)
        for i in 1:size(patch.physical, 1)
            r_m = gridpoints[i, 1]
            vmax = 30.0 * (r_m / 50.0e3) * exp(1.0 - (r_m / 50.0e3))
            patch.physical[i, 1, 1] = 100.0 * exp(-(r_m / 100.0e3)^2)   # h
            patch.physical[i, 3, 1] = vmax                               # v  (free atmosphere)
            patch.physical[i, 5, 1] = 0.8 * vmax                         # vb (boundary layer)
            patch.physical[i, 4, 1] = -0.1 * vmax                        # ub (inflow)
        end
        spectralTransform!(patch)
        gridTransform!(patch)

        return createModelTile(patch, patch, model, spzeros(1, 1)), size(patch.physical, 1)
    end

    pv_mtile, pv_N = build_pv_mixing_tile()

    @testset "RL tile is a single serial column" begin
        # This is the invariant that lets Twoway_PV_mixing share ONE scratch workspace per tile
        # rather than one per thread: `advanceTimestep` only reaches the `@threads :static` column
        # loop when num_columns > 0. For RL it is 0, so the `else` branch runs the equation set
        # exactly once per timestep, serially, over the whole tile. If this ever becomes nonzero,
        # the shared buffers become a data race and must go per-thread (as mc_scratch is).
        @test Springsteel.num_columns(pv_mtile.tile) == 0
    end

    @testset "Twoway_PV_mixing tendency is allocation-free" begin
        Twoway_PV_mixing(pv_mtile, 1, pv_N, 2)        # compile
        @test (@allocations Twoway_PV_mixing(pv_mtile, 1, pv_N, 2)) == 0
    end

    @testset "sw scratch slots are unique and correctly sized" begin
        @test length(unique(Scythe.SW_SCRATCH_SLOTS)) == length(Scythe.SW_SCRATCH_SLOTS)
        @test isconcretetype(typeof(pv_mtile.sw_scratch))
        # Full tile length, NOT kDim: this set has no vertical basis and treats the tile as one column.
        @test all(length(getfield(pv_mtile.sw_scratch, s)) == pv_N
                  for s in Scythe.SW_SCRATCH_SLOTS)
    end
end
