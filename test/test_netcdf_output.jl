using Test
using Scythe
using Springsteel
using SparseArrays
using NCDatasets

# Stage N1: the comprehensive NetCDF analysis writer (src/netcdf_output.jl).
#
# Three layers, tested at the layer they live in:
#
#   1. `mc_derived_fields` is a PURE function of arrays, so it is tested on a hand-built
#      3 x nk column set where every expected value can be written down independently --
#      no grid, no reference file, no NetCDF.
#   2. `regular_reference_profiles` is tested against the reference object it samples: at
#      the MISH levels it must reproduce the stored profile, because that is the whole
#      claim ("the model's own reference spline, evaluated somewhere else").
#   3. The writer is tested by an in-process round trip on a small `moist_compressible_XZ`
#      RiRk fixture (the test_mynn_io.jl idiom), with no workers and no distributed driver.
#
# Plus the option validation, which is the part a user meets first and is meant to fail
# loudly rather than silently write nothing.

@testset "Comprehensive NetCDF output (N1)" begin

    import Springsteel.Thermodynamics: Rd, Rv, Cpd, gravity, rho_v_sat, p_0

    # ── A hand-built reference and slot set ──────────────────────────────────
    # The reference profiles are deliberately ZERO except where a test needs them: with
    # `pbar = rho_dbar = ... = 0` the slot matrices ARE the totals, so every expected value
    # below is written directly and the "add the background back" step is checked separately
    # (testset 3, against a real reference).
    zeros_prof(nk) = zeros(Float64, nk)
    flat_ref(z; rho_cbar = nothing) = Scythe.RegularReferenceProfiles(
        collect(z), zeros_prof(length(z)), zeros_prof(length(z)), zeros_prof(length(z)),
        rho_cbar === nothing ? zeros_prof(length(z)) : rho_cbar,
        zeros_prof(length(z)), zeros_prof(length(z)), zeros_prof(length(z)),
        zeros_prof(length(z)))

    plain_cfg(; has_v = false, ice = false, itrans = :none, ctrans = :none, rtrans = :none,
                cond_floor = false, rain_moments = 1, N0 = 8.0e6, Nc = 100.0,
                cmu = 1.0e-7, rmu = 1.0e-7, imu = (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16)) =
        Scythe.DerivedConfig(ctrans, cmu, rtrans, rmu, :none, 1.0, itrans, imu,
                             rain_moments, ice, has_v, N0, Nc, "test", cond_floor)

    "A 3-column, nk-level dry-ish state with a plausible p/rho_d/E_t and no condensate."
    function synthetic_slots(z; rho_r = nothing, rho_v = nothing)
        n_i = 3; n_k = length(z)
        m(f) = [f(i, z[k]) for i in 1:n_i, k in 1:n_k]
        p     = m((i, zz) -> 1.0e5 * exp(-zz / 8000.0) + 100.0 * i)
        rho_d = m((i, zz) -> 1.15 * exp(-zz / 8500.0))
        rho_t = copy(rho_d)
        # A total energy that puts the retrieval in the troposphere: E_t ~ rho_d*Cvd*T + gz.
        E_t   = m((i, zz) -> 1.15 * exp(-zz / 8500.0) * (718.0 * 290.0 + gravity * zz))
        u     = m((i, zz) -> 2.0 * i)
        w     = m((i, zz) -> 0.1 * zz / 1000.0)
        rain  = rho_r === nothing ? zeros(n_i, n_k) : rho_r
        rv    = rho_v === nothing ? zeros(n_i, n_k) : rho_v
        return (; p, rho_d, rho_t, E_t, Q_ss = zeros(n_i, n_k), rho_v = rv,
                  u, w, rain, cloud = zeros(n_i, n_k))
    end

    # ══════════════════════════════════════════════════════════════════════
    @testset "mc_derived_fields on a hand-built column set" begin
        z = collect(0.0:250.0:1000.0)
        ref = flat_ref(z)
        cfg = plain_cfg()
        slots = synthetic_slots(z)
        d = Scythe.mc_derived_fields(z, slots, ref, cfg)

        @testset "temperature is the model's own retrieval, bitwise" begin
            # The 4-argument method delegates with rho_ice = 0.0 and that identity is
            # bitwise (see `retrieve_temperature`); with cond_floor off and ice off the
            # derived layer must land on exactly that number.
            T_ref = Scythe.retrieve_temperature.(d.M, d.rho_d, d.rho_t, d.rho_liq)
            @test all(d.T .=== T_ref)
            @test all(150.0 .< d.T .< 350.0)
            # ...and M is the available enthalpy the retrieval is defined against.
            M_ref = d.p .+ d.E_t .- d.rho_t .* (d.ke .+ gravity .* reshape(z, 1, :))
            @test all(d.M .≈ M_ref)
        end

        @testset "ke carries v only when the geometry has it" begin
            @test all(d.ke .≈ 0.5 .* (slots.u .^ 2 .+ slots.w .^ 2))
            sv = merge(slots, (; v = fill(7.0, size(slots.u))))
            dv = Scythe.mc_derived_fields(z, sv, ref, plain_cfg(has_v = true))
            @test all(dv.ke .≈ 0.5 .* (slots.u .^ 2 .+ 49.0 .+ slots.w .^ 2))
            # And a `v` slot present but `has_v = false` must NOT sneak into the kinetic
            # energy: the config, not the slot table, decides.
            dn = Scythe.mc_derived_fields(z, sv, ref, plain_cfg(has_v = false))
            @test all(dn.ke .≈ d.ke)
        end

        @testset "RH == 1 at saturation" begin
            # T does not depend on rho_v (the vapor is its own slot), so the saturation
            # density can be computed from the first pass and fed back.
            sat = rho_v_sat.(d.T, d.p ./ 100.0)
            ds = Scythe.mc_derived_fields(z, merge(slots, (; rho_v = sat)), ref, cfg)
            @test all(abs.(ds.RH .- 1.0) .< 1.0e-12)
            @test all(ds.rho_v .≈ sat)
        end

        @testset "theta is the Pa-form potential temperature" begin
            # The EOS temperature of a dry column, p/(Rd rho_d), raised by (1e5/p)^kappa.
            # This is the check that catches a hPa/Pa mix-up in the call.
            T_eos = d.p ./ (Rd .* d.rho_d)
            @test all(d.theta .≈ T_eos .* (100.0 * p_0 ./ d.p) .^ (Rd / Cpd))
        end

        @testset "rain rate is the sedimentation flux of the run's own fall speed" begin
            rr = fill(1.0e-3, 3, length(z))
            dr = Scythe.mc_derived_fields(z, merge(slots, (; rain = rr)), ref, cfg)
            Vt = Scythe.rain_terminal_velocity.(dr.rho_r, dr.rho_d, dr.T)
            @test all(dr.rain_rate .≈ .-dr.rho_r .* Vt .* 3600.0)
            @test all(dr.rain_rate .> 0.0)              # falling rain is a positive rate
            @test dr.precip_rate == dr.rain_rate[:, 1]  # surface level
        end

        @testset "reflectivity is monotone in rain and NaN with no hydrometeors" begin
            @test all(isnan, d.reflectivity)            # dry column: no echo anywhere
            dbz = [Scythe.reflectivity_dBZ(0.0, q, 8.0e6, 1.0e8)
                   for q in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)]
            @test all(isfinite, dbz)
            @test issorted(dbz)
            @test isnan(Scythe.reflectivity_dBZ(0.0, 0.0, 8.0e6, 1.0e8))
        end

        @testset "column integrals: constant vapor over a uniform grid" begin
            const_rv = 0.01
            dv = Scythe.mc_derived_fields(
                z, merge(slots, (; rho_v = fill(const_rv, 3, length(z)))), ref, cfg)
            @test all(abs.(dv.PW .- const_rv * (z[end] - z[1])) .< 1.0e-12)
            @test all(dv.column_cloud_water .== 0.0)
            @test all(dv.column_rain_water .== 0.0)
        end

        @testset "the cloud transform is inverted against its own reference" begin
            rho_cbar = fill(2.0e-4, length(z))
            rc = fill(5.0e-4, 3, length(z))
            slot = Scythe.condensate_slot.(rc, reshape(rho_cbar, 1, :), :bhyp, 1.0e-7)
            dc = Scythe.mc_derived_fields(
                z, merge(slots, (; cloud = slot)), flat_ref(z; rho_cbar = rho_cbar),
                plain_cfg(ctrans = :bhyp))
            @test all(abs.(dc.rho_c .- rc) .< 1.0e-12)
        end

        @testset "ice mass recovery inverts the transform" begin
            imu = (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16)
            vals = ntuple(j -> fill(1.0e-5 * j, 3, length(z)), 12)
            raw = ntuple(j -> Scythe.total_slot.(vals[j], :bhyp, imu[((j - 1) % 4) + 1]), 12)
            di = Scythe.mc_derived_fields(z, merge(slots, (; ice = raw)), ref,
                                          plain_cfg(ice = true, itrans = :bhyp, imu = imu))
            for j in 1:12
                expect = Scythe.recover_total.(raw[j], :bhyp, imu[((j - 1) % 4) + 1])
                @test all(di.ice[j] .=== expect)
                @test all(abs.(di.ice[j] .- vals[j]) .< 1.0e-9 * maximum(vals[j]))
            end
            @test all(di.rho_ice .≈ vals[1] .+ vals[5] .+ vals[9])
            @test haskey(di, :RH_ice)
            @test haskey(di, :column_ice_water)
        end

        @testset "the diagnostic condensate floor is read, and only by the retrieval" begin
            neg = fill(-1.0e-5, 3, length(z))
            df = Scythe.mc_derived_fields(z, merge(slots, (; cloud = neg)), ref,
                                          plain_cfg(cond_floor = true))
            dn = Scythe.mc_derived_fields(z, merge(slots, (; cloud = neg)), ref,
                                          plain_cfg(cond_floor = false))
            # State stays RAW under the floor -- the floor is a reader, not a clamp.
            @test all(df.rho_c .== neg)
            @test all(df.rho_liq .== dn.rho_liq)
            # ...and the temperature differs, because the retrieval read the floored value.
            @test all(df.T .!= dn.T)
        end
    end

    # ── A small moist_compressible_XZ RiRk fixture (test_mynn_io.jl's idiom) ──
    """Stably stratified dry column with an exact hydrostatic pressure for a linear
    temperature profile."""
    function stable_column(z; T0 = 300.0, lapse = 0.004, p0 = 100000.0)
        Tk = @. T0 - lapse * z
        p_Pa = @. p0 * (Tk / T0)^(gravity / (Rd * lapse))
        rho_d = p_Pa ./ (Rd .* Tk)
        n = length(z)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    """A small `moist_compressible_XZ` RiRk patch with a written exact reference and a
    non-trivial (fitted) perturbation state, plus the ModelParameters that go with it.
    `formats === nothing` leaves `:output_formats` unset, i.e. the DEFAULT."""
    function make_nc_fixture(tmpdir; formats = nothing, outname = "out",
                             num_cells_i = 3, num_cells_k = 12, kMax = 3.0e3)
        opts_names = Dict{Symbol,Any}()
        varlist = Scythe.mc_var_names(opts_names; cyl = false)
        rain_name = Scythe.rain_var_name(opts_names)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = scalar_bc
        bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
        top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 6.0e3, num_cells_i = num_cells_i,
            kMin = 0.0, kMax = kMax, num_cells_k = num_cells_k,
            BCL = side_bc, BCR = side_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
        ref_file = joinpath(tmpdir, "nc_$(outname).ref")
        options = Dict{Symbol,Any}(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => false)
        formats === nothing || (options[:output_formats] = formats)
        model = ModelParameters(
            ts = 1.0, integration_time = 4.0, output_interval = 2.0,
            equation_set = "moist_compressible_XZ",
            output_dir = joinpath(tmpdir, outname),
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict{Symbol,Any}(
                :Khdiff => 0.0, :Kvdiff => 0.0, :tau_qss => 10.0, :alpha => 0.0,
                :z_damp => 2.0e3, :f => 0.0),
            options = options)
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, end]
        col = stable_column(z)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        # A non-trivial perturbation state so the totals are not just the reference: a warm
        # (positive E_t') bubble and a little vapor.
        patch.physical .= 0.0
        for j in 1:size(gridpoints, 1)
            x = gridpoints[j, 1]; zz = gridpoints[j, end]
            bump = exp(-(((x - 3.0e3) / 1.5e3)^2 + ((zz - 1.0e3) / 6.0e2)^2))
            patch.physical[j, vars["E_t"], 1] = 5.0e3 * bump
            patch.physical[j, vars["rho_v"], 1] = 1.0e-3 * bump
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        return patch, model, gp, z
    end

    # ══════════════════════════════════════════════════════════════════════
    @testset "regular_reference_profiles reproduces the model reference" begin
        mktempdir() do tmpdir
            patch, model, gp, z_mish = make_nc_fixture(tmpdir; outname = "ref")
            col = Scythe.reference_column(patch, gp)
            ref = Scythe.build_reference_state(model, z_mish, col)
            @test ref isa Springsteel.PressureReferenceState

            # Evaluated at the MISH levels themselves, the sampled spline must reproduce
            # the stored profile: it IS that profile's fit, so anything else means the
            # wrong column, the wrong ordering, or a transform that is not the reference's
            # own.
            #
            # NOT to round-off, though, and the difference is real rather than a defect.
            # `Springsteel._profile` stores the RAW sounding values in the value column and
            # only the DERIVATIVE columns come from the fit, so the value column is not on
            # the spline. The comprehensive file lives on the regular grid, where the only
            # thing that can be evaluated IS the spline, so the profiles it writes carry
            # the fit residual -- the same residual every reference DERIVATIVE in the
            # equation set already carries, and the same one `tc/tc_postprocess.jl`
            # produced. It is a resolution-dependent interpolation error, so the tolerance
            # is loose here and PINNED BY CONVERGENCE below rather than by a magic number.
            prof = Scythe.regular_reference_profiles(ref, col, collect(z_mish))
            relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), eps()))
            e_coarse = (p = relerr(prof.pbar, Springsteel.ref_pressure(ref)[:, 1]),
                        d = relerr(prof.rho_dbar, Springsteel.ref_rho_d(ref)[:, 1]),
                        t = relerr(prof.rho_tbar, Springsteel.ref_rho_t(ref)[:, 1]),
                        E = relerr(prof.E_tbar, Springsteel.ref_total_energy(ref)[:, 1]),
                        T = relerr(prof.Tbar, Springsteel.reference_temperature(ref)))
            for e in e_coarse
                @test e < 1.0e-6
            end
            @test prof.z == collect(z_mish)

            # Convergence: doubling the vertical resolution must shrink the residual. That
            # is what makes the tolerance above an interpolation error rather than a bug --
            # a wrong column or a wrong ordering would not converge.
            patch2, model2, gp2, z2 = make_nc_fixture(tmpdir; outname = "ref2",
                                                      num_cells_k = 24)
            col2 = Scythe.reference_column(patch2, gp2)
            ref2 = Scythe.build_reference_state(model2, z2, col2)
            prof2 = Scythe.regular_reference_profiles(ref2, col2, collect(z2))
            @test relerr(prof2.pbar, Springsteel.ref_pressure(ref2)[:, 1]) < e_coarse.p
            @test relerr(prof2.rho_dbar, Springsteel.ref_rho_d(ref2)[:, 1]) < e_coarse.d
        end
    end

    # ══════════════════════════════════════════════════════════════════════
    @testset "comprehensive writer round trip" begin
        mktempdir() do tmpdir
            patch, model, gp, _ = make_nc_fixture(tmpdir; outname = "rt")
            ctx = Scythe.netcdf_output_context(patch, model)
            @test ctx.active                     # default formats are [:netcdf]

            Scythe.write_output(patch, model, 3600.0)
            path = joinpath(model.output_dir, "3600.0.nc")
            @test isfile(path)
            # The default writes NO CSV any more.
            @test !isfile(joinpath(model.output_dir, "3600.0_physical.csv"))

            NCDataset(path, "r") do ds
                @test ds.attrib["scythe_file_kind"] == "comprehensive"
                @test ds.attrib["netcdf_grid"] == "regular"
                @test ds.attrib["equation_set"] == "moist_compressible_XZ"
                # No BL, no radiation, no surface layer on this fixture.
                @test ds.attrib["physics_groups"] == "none (schemes not yet run)"
                @test ds.dim["time"] == 1
                @test ds.dim["x"] == gp.i_regular_out
                @test ds.dim["z"] == gp.k_regular_out
                # Read via `.var` so the raw stored seconds come back (a CF `units` of
                # "seconds since ..." would decode to a DateTime; ours is plain seconds).
                @test ds["time"].var[1] == 3600.0

                inventory = ["p_prime", "rho_d_prime", "rho_t_prime", "E_t_prime",
                             "Q_ss_prime", "rho_v_prime",
                             "p", "rho_d", "rho_t", "E_t", "Q_ss", "rho_v", "rho_c",
                             "rho_r", "u", "w",
                             "T", "theta", "theta_e", "RH", "q_v", "q_c", "q_r",
                             "reflectivity", "rain_rate", "precip_rate",
                             "PW", "column_cloud_water", "column_rain_water",
                             "pbar", "rho_dbar", "rho_vbar", "rho_cbar", "rho_tbar",
                             "E_tbar", "Q_ssbar", "Tbar"]
                for nm in inventory
                    @test haskey(ds, nm)
                end
                # This fixture has no v, no rain number, no ice and no TKE.
                for nm in ("v", "n_r", "rho_i1", "rho_e", "nu_c", "nu_r", "RH_ice")
                    @test !haskey(ds, nm)
                end
                for nm in inventory
                    @test haskey(ds[nm].attrib, "units")
                    @test haskey(ds[nm].attrib, "long_name")
                end
                @test ds["x"].attrib["units"] == "m"
                @test ds["z"].attrib["units"] == "m"

                T = Array(ds["T"])[1, :, :]
                @test all(isfinite, T)
                @test all(150.0 .< T .< 350.0)

                # The totals really are prime + background(z), and the background written
                # here really is the context's.
                pp = Array(ds["p_prime"])[1, :, :]
                p = Array(ds["p"])[1, :, :]
                pbar = Array(ds["pbar"])
                @test all(p .≈ pp .+ reshape(pbar, 1, :))
                @test pbar ≈ ctx.ref.pbar
                @test Array(ds["Tbar"]) ≈ ctx.ref.Tbar
                @test Array(ds["rho_cbar"]) ≈ ctx.ref.rho_cbar
                @test Array(ds["z"]) ≈ ctx.z_reg
                @test Array(ds["x"]) ≈ ctx.x_reg
            end
        end

        @testset "[:csv] writes the CSVs and no .nc" begin
            mktempdir() do tmpdir
                patch, model, _, _ = make_nc_fixture(tmpdir; formats = [:csv],
                                                     outname = "csvonly")
                Scythe.write_output(patch, model, 0.0)
                # RiRk `write_grid` writes the physical/spectral pair (the `_gridded.csv`
                # of the 1-D/RL writers has no 2-D k-active counterpart).
                for f in ("0.0_physical.csv", "0.0_spectral.csv")
                    @test isfile(joinpath(model.output_dir, f))
                end
                @test !isfile(joinpath(model.output_dir, "0.0.nc"))
                @test !isfile(joinpath(model.output_dir, "0.0_raw.nc"))
            end
        end

        @testset "[:netcdf_raw] writes the legacy prognostic file" begin
            mktempdir() do tmpdir
                patch, model, _, _ = make_nc_fixture(tmpdir; formats = [:netcdf_raw],
                                                    outname = "raw")
                Scythe.write_output(patch, model, 0.0)
                raw = joinpath(model.output_dir, "0.0_raw.nc")
                @test isfile(raw)
                @test !isfile(joinpath(model.output_dir, "0.0.nc"))
                NCDataset(raw, "r") do ds
                    # The Springsteel layout: prognostic slot names only, no derived set.
                    @test haskey(ds, "p")
                    @test haskey(ds, "rho_c")
                    @test !haskey(ds, "T")
                    @test !haskey(ds, "pbar")
                    @test !haskey(ds, "reflectivity")
                    @test ds["time"].var[1] == 0.0
                end
            end
        end

        @testset "[:csv, :netcdf] writes both" begin
            mktempdir() do tmpdir
                patch, model, _, _ = make_nc_fixture(tmpdir; formats = [:csv, :netcdf],
                                                     outname = "both")
                Scythe.write_output(patch, model, 0.0)
                @test isfile(joinpath(model.output_dir, "0.0_physical.csv"))
                @test isfile(joinpath(model.output_dir, "0.0.nc"))
                NCDataset(joinpath(model.output_dir, "0.0.nc"), "r") do ds
                    @test ds.attrib["scythe_file_kind"] == "comprehensive"
                end
            end
        end
    end

    # ══════════════════════════════════════════════════════════════════════
    # Stage N2: the physics groups (BL / radiation / surface).
    #
    # `physics_snapshot` runs on a WORKER and `assemble_physics` on the master; both are
    # pure functions of a `ModelTile` / a vector of snapshots, so a one-tile "gather"
    # built in process exercises exactly the code a distributed run does, minus the
    # serialization (which the distributed smoke covers separately).
    @testset "physics groups in the comprehensive file" begin

        """A `moist_compressible_XZ` RiRk tile with a live boundary layer and surface
        fluxes. `bl` is `:mynn`, `:louis` or `:none`; `rad` adds the artifact-free
        `:prescribed` radiation scheme. The MYNN half is test_mynn_io.jl's `make_io_tile`
        (warm SST under a cool column, `:mynn_init = :taper`, so ONE step gives nonzero
        K_m/K_h/pblh)."""
        function make_physics_tile(tmpdir; bl = :mynn, rad = false, outname = "phys")
            opts_names = Dict{Symbol,Any}()
            bl === :mynn && (opts_names[:mynn] = true)
            varlist = Scythe.mc_var_names(opts_names; cyl = false)
            rain_name = Scythe.rain_var_name(opts_names)
            vars = Dict(v => i for (i, v) in enumerate(varlist))
            scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
            bot_bc = merge(scalar_bc,
                           Dict("w" => DirichletBC(), rain_name => NaturalBC()))
            top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
            gp = GridParameters(geometry = "RiRk",
                iMin = 0.0, iMax = 8.0e3, num_cells_i = 2,
                kMin = 0.0, kMax = 5.0e3, num_cells_k = 30,
                BCL = scalar_bc, BCR = scalar_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
            ref_file = joinpath(tmpdir, "phys_$(outname).ref")
            options = Dict{Symbol,Any}(:semiimplicit => true,
                                       :exact_reference_state => true,
                                       :precipitation => false,
                                       :surface_fluxes => bl !== :none)
            if bl === :mynn
                options[:mynn] = true
                options[:mynn_init] = :taper
                options[:mynn_trace] = false
            elseif bl === :louis
                options[:louis_bl] = true
            end
            if rad
                options[:radiation] = :prescribed
                options[:radiation_trace] = false
            end
            model = ModelParameters(
                ts = 2.0, integration_time = 8.0, output_interval = 4.0,
                equation_set = "moist_compressible_XZ",
                output_dir = joinpath(tmpdir, outname),
                ref_state_file = ref_file, grid_params = gp,
                physical_params = Dict{Symbol,Any}(
                    :Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                    :Kvdiff_water => 0.0, :tau_qss => 10.0, :alpha => 0.0,
                    :z_damp => 4.0e3, :f => 0.0, :Cd => -1.0, :Ls => 0.0,
                    :Ck => 1.0e-3, :U_min => 1.0, :l_inf => 80.0, :SST => 302.0,
                    :radiation_prescribed_rate => -1.5),
                options = options)
            gp = model.grid_params
            patch = createGrid(gp)
            gridpoints = Scythe.getGridpoints(patch)
            kDim = gp.kDim
            zz = gridpoints[1:kDim, end]
            col = stable_column(zz)
            Scythe.write_exact_ref_mc(ref_file, zz, col.p_Pa, col.rho_d,
                                      col.rho_v, col.rho_c)
            patch.physical .= 0.0
            spectralTransform!(patch)
            gridTransform!(patch)
            hrm = sparse(Int64[], Int64[], Float64[],
                         size(patch.spectral, 1), size(patch.spectral, 2))
            mtile = createModelTile(patch, patch, model, hrm)
            return mtile, patch, model, gp
        end

        "Advance every column of `mtile` for `nsteps` real steps."
        function spin_up!(mtile, kDim, nsteps)
            ncols = div(size(mtile.tile.physical, 1), kDim)
            for t in 1:nsteps, c in 1:ncols
                Scythe.advance_column(mtile, c, t)
            end
            return ncols
        end

        "Write the comprehensive file for `mtile` with the physics of its own tile."
        function write_with_physics(mtile, patch, model, t)
            ctx = Scythe.netcdf_output_context(patch, model)
            physics = Scythe.assemble_physics([Scythe.physics_snapshot(mtile)])
            path = joinpath(model.output_dir, "$(string(round(t; digits = 2))).nc")
            isdir(model.output_dir) || mkpath(model.output_dir)
            Scythe.write_netcdf_comprehensive(path, patch, model, t, ctx, physics)
            return path, ctx, physics
        end

        @testset "MYNN + surface: fields, names and provenance" begin
            mktempdir() do tmpdir
                mtile, patch, model, gp = make_physics_tile(tmpdir; outname = "my")
                @test mtile.mynn.active
                @test mtile.surface.active
                # ...and the store is inert until a column actually runs.
                @test mtile.surface.n_calls[1] == 0
                @test Scythe.physics_snapshot(mtile).surface === nothing

                spin_up!(mtile, gp.kDim, 1)
                @test mtile.surface.n_calls[1] > 0
                @test any(!=(0.0), mtile.mynn.K_h)

                path, ctx, physics = write_with_physics(mtile, patch, model, 4.0)
                @test physics !== nothing
                @test physics.mynn !== nothing
                @test physics.surface !== nothing
                @test physics.radiation === nothing

                NCDataset(path, "r") do ds
                    @test ds.attrib["physics_groups"] == "mynn,surface"
                    @test ds.attrib["boundary_layer"] == "mynn"
                    @test ds.attrib["surface_fluxes"] == 1
                    @test ds.attrib["sfc_z0"] == "komori"

                    # Every documented MYNN variable, on the REGULAR grid.
                    for nm in ("K_m", "K_h", "mynn_e", "mynn_el", "mynn_sm", "mynn_sh",
                               "mynn_cldfra_bl", "mynn_qc_bl", "mynn_qi_bl", "mynn_vt",
                               "mynn_vq", "mynn_P_s", "mynn_P_s_mynn", "mynn_P_b",
                               "mynn_eps", "mynn_tke_transport", "mynn_s_aw")
                        @test haskey(ds, nm)
                        @test size(Array(ds[nm])) == (1, gp.i_regular_out,
                                                      gp.k_regular_out)
                        @test all(isfinite, Array(ds[nm])[1, :, :])
                    end
                    for nm in ("mynn_pblh", "mynn_kpbl", "mynn_ust", "mynn_inv_L",
                               "mynn_plume_ktop", "mynn_plume_ztop", "mynn_aw_max",
                               "mynn_bdry_E")
                        @test haskey(ds, nm)
                        @test size(Array(ds[nm])) == (1, gp.i_regular_out)
                        @test all(isfinite, Array(ds[nm])[1, :])
                    end
                    for nm in ("F_sh", "F_q", "tau_u", "tau_v", "ust", "inv_L", "U10",
                               "Cd", "Ch", "z0m")
                        @test haskey(ds, nm)
                        @test size(Array(ds[nm])) == (1, gp.i_regular_out)
                        @test all(isfinite, Array(ds[nm])[1, :])
                        @test haskey(ds[nm].attrib, "units")
                        @test haskey(ds[nm].attrib, "long_name")
                    end
                    # The surface fluxes are REAL: a 302 K SST under a ~300 K column
                    # drives an upward sensible-heat flux and a moisture flux.
                    @test all(>(0.0), Array(ds["F_sh"])[1, :])
                    @test all(>(0.0), Array(ds["F_q"])[1, :])
                    @test all(>(0.0), Array(ds["Cd"])[1, :])

                    # `mynn_pblh` IS `regrid1d` of the held column vector -- the claim the
                    # whole regrid path makes, checked against the pure helper.
                    wx = Scythe.interp_weights(physics.mynn.x, ctx.x_reg)
                    @test Array(ds["mynn_pblh"])[1, :] ==
                          Scythe.regrid1d(mtile.mynn.pblh, wx)
                    @test Array(ds["F_sh"])[1, :] ==
                          Scythe.regrid1d(mtile.surface.F_sh, wx)

                    # The MYNN configuration attributes, under the tc_postprocess names.
                    for nm in ("mynn_closure", "mynn_edmf", "mynn_init_mode",
                               "mynn_water_carry", "mynn_n_clamp_e", "mynn_n_cap_K",
                               "mynn_n_diffnum", "mynn_n_gate", "mynn_n_stall",
                               "mynn_n_plume", "mynn_interval_s")
                        @test haskey(ds.attrib, nm)
                    end
                    @test ds.attrib["mynn_init_mode"] == "taper"
                    # ...and no double prefix on the one name that already carries it.
                    @test !haskey(ds.attrib, "mynn_mynn_interval_s")
                    @test haskey(ds.attrib, "mynn_regridding")

                    # No radiation group.
                    @test !haskey(ds, "dT_net")
                    @test !haskey(ds, "olr")
                end

                # The weights are CACHED on the context after the first write.
                @test ctx.physics_weights isa Dict
                @test haskey(ctx.physics_weights, :mynn)
                @test haskey(ctx.physics_weights, :surface)
            end
        end

        @testset "radiation group" begin
            mktempdir() do tmpdir
                mtile, patch, model, gp = make_physics_tile(tmpdir; bl = :none,
                                                            rad = true, outname = "rad")
                @test mtile.radiation.active
                @test !mtile.surface.active          # no BL, no surface store
                Scythe.radiation_update!(mtile, 1)
                @test any(!=(0.0), mtile.radiation.q_lw)

                path, _, physics = write_with_physics(mtile, patch, model, 4.0)
                @test physics.radiation !== nothing
                @test physics.mynn === nothing
                @test physics.surface === nothing

                NCDataset(path, "r") do ds
                    @test ds.attrib["physics_groups"] == "radiation"
                    for nm in ("q_lw", "q_sw", "q_sw_applied", "dT_lw", "dT_sw", "dT_net")
                        @test haskey(ds, nm)
                        @test size(Array(ds[nm])) == (1, gp.i_regular_out,
                                                      gp.k_regular_out)
                    end
                    for nm in ("olr", "olr_model_top", "lw_sfc_dn", "lw_sfc_up",
                               "sw_sfc_dn", "sw_sfc_up", "sw_toa_dn", "sw_toa_up",
                               "lwp", "iwp", "cloudy")
                        @test haskey(ds, nm)
                        @test size(Array(ds[nm])) == (1, gp.i_regular_out)
                    end
                    # dT_net = dT_lw + sw_scale*dT_sw, and :prescribed has no shortwave.
                    @test Array(ds["dT_net"])[1, :, :] == Array(ds["dT_lw"])[1, :, :]
                    @test all(<(0.0), Array(ds["dT_lw"])[1, :, :])   # -1.5 K/day cooling
                    # The face fluxes stay in the sidecar, deliberately.
                    for nm in ("flux_lw_up", "flux_sw_dn", "zf")
                        @test !haskey(ds, nm)
                    end
                    for nm in ("radiation_scheme", "radiation_method", "radiation_solar",
                               "radiation_forcing", "radiation_z_max",
                               "radiation_interval_s", "radiation_n_clamp_tk",
                               "radiation_n_clamp_re_liq", "radiation_n_clamp_re_ice",
                               "radiation_n_neg_rho_v", "radiation_time",
                               "radiation_cos_zenith", "radiation_toa_flux",
                               "radiation_sw_scale")
                        @test haskey(ds.attrib, nm)
                    end
                    @test ds.attrib["radiation_scheme"] == "prescribed"
                    @test ds.attrib["radiation_time"] == 4.0
                    @test !haskey(ds.attrib, "radiation_radiation_interval_s")
                end
            end
        end

        @testset "a scheme that has not run yet writes NO group" begin
            mktempdir() do tmpdir
                mtile, patch, model, gp = make_physics_tile(tmpdir; rad = true,
                                                            outname = "fresh")
                # Fresh tile: MYNN, radiation and the surface store are all configured
                # and all EMPTY. Zero-filling them would be indistinguishable from a run
                # whose boundary layer genuinely did nothing.
                snap = Scythe.physics_snapshot(mtile)
                @test snap.mynn === nothing
                @test snap.radiation === nothing
                @test snap.surface === nothing
                @test Scythe.assemble_physics([snap]) === nothing

                path, _, physics = write_with_physics(mtile, patch, model, 0.0)
                @test physics === nothing
                NCDataset(path, "r") do ds
                    @test ds.attrib["physics_groups"] == "none (schemes not yet run)"
                    for nm in ("K_h", "mynn_pblh", "F_sh", "dT_net", "olr")
                        @test !haskey(ds, nm)
                    end
                    @test haskey(ds, "T")        # the prognostic/derived set is unaffected
                end
            end
        end

        @testset "Louis BL: the surface group alone" begin
            mktempdir() do tmpdir
                mtile, patch, model, gp = make_physics_tile(tmpdir; bl = :louis,
                                                            outname = "louis")
                @test !mtile.mynn.active
                @test mtile.surface.active
                spin_up!(mtile, gp.kDim, 1)
                @test mtile.surface.n_calls[1] > 0

                path, _, physics = write_with_physics(mtile, patch, model, 4.0)
                @test physics.surface !== nothing
                @test physics.mynn === nothing

                NCDataset(path, "r") do ds
                    @test ds.attrib["physics_groups"] == "surface"
                    @test ds.attrib["boundary_layer"] == "louis"
                    @test haskey(ds, "F_sh")
                    @test haskey(ds, "tau_u")
                    @test all(isfinite, Array(ds["U10"])[1, :])
                    @test !haskey(ds, "K_h")
                    @test !haskey(ds, "mynn_pblh")
                end
            end
        end

        @testset "assemble_physics stitches tiles and refuses a partial group" begin
            mktempdir() do tmpdir
                mtile, patch, model, gp = make_physics_tile(tmpdir; outname = "asm")
                spin_up!(mtile, gp.kDim, 1)
                snap = Scythe.physics_snapshot(mtile)

                # Two "tiles" (the same one twice, at different offsets) concatenate
                # along x, in ASCENDING offset order whatever order they arrive in.
                lo = merge(snap, (; offset = 0))
                hi = merge(snap, (; offset = 100))
                a = Scythe.assemble_physics([hi, lo])
                @test length(a.surface.F_sh) == 2 * length(snap.surface.F_sh)
                @test a.surface.F_sh == vcat(snap.surface.F_sh, snap.surface.F_sh)
                @test size(a.mynn.K_h, 1) == 2 * size(snap.mynn.K_h, 1)
                @test size(a.mynn.K_h, 2) == size(snap.mynn.K_h, 2)
                @test a.mynn.z == snap.mynn.z            # the vertical is shared
                # The counters are the DOMAIN-wide census: summed, not taken from tile 1.
                @test a.mynn.attrs["n_clamp_e"] == 2 * snap.mynn.attrs["n_clamp_e"]

                # A group present on one tile and not the other is an error, loudly:
                # the schemes are configured per MODEL, so a partial answer means a tile
                # failed to run one and the file would cover part of the domain.
                bare = merge(snap, (; offset = 200, surface = nothing))
                @test_throws ErrorException Scythe.assemble_physics([lo, bare])
                @test Scythe.assemble_physics(Any[]) === nothing
            end
        end
    end

    # ══════════════════════════════════════════════════════════════════════
    @testset "output option validation" begin
        mktempdir() do tmpdir
            bad(formats) = begin
                _, model, _, _ = make_nc_fixture(tmpdir; formats = formats,
                                                 outname = "v$(hash(formats))")
                model
            end
            @test_throws ErrorException Scythe.validate_output_options(bad([:parquet]))
            @test_throws ErrorException Scythe.validate_output_options(bad(:csv))
            @test_throws ErrorException Scythe.validate_output_options(bad(["csv"]))
            @test_throws ErrorException Scythe.validate_output_options(bad([:jld2]))
            # ...and the :jld2 message still points at restart_interval.
            err = try
                Scythe.validate_output_options(bad([:jld2])); ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("restart_interval", err)

            # A valid list passes, and so does an absent key (the default).
            @test Scythe.validate_output_options(bad([:csv, :netcdf, :netcdf_raw])) ===
                  nothing
            _, plain, _, _ = make_nc_fixture(tmpdir; outname = "vdefault")
            @test Scythe.validate_output_options(plain) === nothing

            # The :netcdf_* keys.
            with_opt(k, v) = begin
                _, m, _, _ = make_nc_fixture(tmpdir; outname = "o$(k)")
                m.options[k] = v
                m
            end
            @test_throws ErrorException Scythe.validate_output_options(
                with_opt(:netcdf_grid, :mish))
            @test_throws ErrorException Scythe.validate_output_options(
                with_opt(:netcdf_grid, :chebyshev))
            @test_throws ErrorException Scythe.validate_output_options(
                with_opt(:netcdf_typo, 1))
            @test_throws ErrorException Scythe.validate_output_options(
                with_opt(:netcdf_derivatives, 1))
            @test Scythe.validate_output_options(with_opt(:netcdf_grid, :regular)) ===
                  nothing
            @test Scythe.validate_output_options(with_opt(:netcdf_derivatives, true)) ===
                  nothing
        end
    end
end
