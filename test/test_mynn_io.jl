using Test
using Scythe
using Springsteel
using SparseArrays
using NCDatasets

# Stage S9: the MYNN-EDMF sidecar writer (`mynn_write!`/`mynn_write_final!`,
# src/mynn_io.jl) and reader (`read_mynn`/`mynn_snapshots`).
#
# The idiom is test_radiation_io.jl's: a small, self-contained fixture (reproduced here
# rather than reached across files) that drives `Scythe.advance_column` directly, the
# same idiom test_mynn_bl.jl uses for the closure itself -- an actual timestep still runs
# the closure and populates the held state, but nothing here needs the full distributed
# driver or `radiation_prepass!`'s call site.

@testset "MYNN sidecar I/O (S9)" begin

    import Springsteel.Thermodynamics: Rd, gravity

    """Stably stratified dry column with an exact hydrostatic pressure for a linear
    temperature profile (test_mynn_bl.jl's fixture, reproduced so this file has no
    cross-testset dependency)."""
    function stable_column(z; T0 = 300.0, lapse = 0.004, p0 = 100000.0)
        Tk = @. T0 - lapse * z
        p_Pa = @. p0 * (Tk / T0)^(gravity / (Rd * lapse))
        rho_d = p_Pa ./ (Rd .* Tk)
        n = length(z)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    """A small `moist_compressible_XZ` RiRk tile. `mynn = false` builds the closure-off
    control (used only to check the off-path writes nothing). `output` and `trace`
    forward straight to `:mynn_output`/`:mynn_trace`; every other MYNN option is left at
    its (now-live) default so the sidecar sees a realistic configuration. Warm SST
    against a cool near-surface column with `:surface_fluxes` on drives real turbulence
    (K_m, K_h, pblh all nonzero after a step), the same fixture idea test_mynn_bl.jl uses."""
    function make_io_tile(tmpdir; mynn = true, output = true, trace = false,
                          num_cells_i = 2, num_cells_k = 30, kMax = 5.0e3, ts = 2.0,
                          outname = "out")
        opts_names = Dict{Symbol,Any}()
        mynn && (opts_names[:mynn] = true)
        varlist = Scythe.mc_var_names(opts_names; cyl = false)
        rain_name = Scythe.rain_var_name(opts_names)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = scalar_bc
        bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
        top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 8.0e3, num_cells_i = num_cells_i,
            kMin = 0.0, kMax = kMax, num_cells_k = num_cells_k,
            BCL = side_bc, BCR = side_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
        ref_file = joinpath(tmpdir, "mynn_io_$(outname).ref")
        options = Dict{Symbol,Any}(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => false,
                                   :surface_fluxes => mynn)
        if mynn
            options[:mynn] = true
            # `:taper` (not `:zero`): a resting column started at TKE = 0 stays at
            # K_m = K_h = 0 through its very first closure call (test_mynn_bl.jl's
            # "resting column at equilibrium" pins exactly that), and this fixture wants
            # the sidecar to see NONZERO diffusivities after one spin-up step, from the
            # warm-SST forcing alone -- `:taper` seeds an initial TKE estimate from that
            # forcing rather than requiring several steps to grow one from zero.
            options[:mynn_init] = :taper
            options[:mynn_trace] = trace
            options[:mynn_output] = output
        end
        model = ModelParameters(
            ts = ts, integration_time = 4.0 * ts, output_interval = 2.0 * ts,
            equation_set = "moist_compressible_XZ",
            output_dir = joinpath(tmpdir, outname),
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict{Symbol,Any}(
                :Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                :Kvdiff_water => 0.0, :tau_qss => 10.0, :alpha => 0.0,
                :z_damp => 4.0e3, :f => 0.0, :Cd => -1.0, :Ls => 0.0,
                :Ck => 1.0e-3, :U_min => 1.0, :l_inf => 80.0, :SST => 302.0),
            options = options)
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, end]
        col = stable_column(z)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model, gp
    end

    "Advance every column of `mtile` once at step `t`."
    function step_all!(mtile, kDim, t)
        ncols = div(size(mtile.tile.physical, 1), kDim)
        for c in 1:ncols
            Scythe.advance_column(mtile, c, t)
        end
        return ncols
    end

    "Run `nsteps` real steps (closure included) so the held MYNN state is non-trivial."
    function spin_up!(mtile, kDim, nsteps)
        for t in 1:nsteps
            step_all!(mtile, kDim, t)
        end
        return nothing
    end

    @testset "mynn_write! fires on cadence and round-trips" begin
        mktempdir() do tmp
            mtile, patch, model, gp = make_io_tile(tmp)
            MY = mtile.mynn
            kDim = gp.kDim
            @test MY.active && MY.output

            spin_up!(mtile, kDim, 1)
            @test any(!=(0.0), MY.K_h)      # surface heat flux drove real mixing

            Scythe.mynn_write!(mtile, 1)
            tag = "0.0"
            path = joinpath(model.output_dir, "$(tag)_mynn_i0.nc")
            @test isfile(path)

            NCDataset(path, "r") do ds
                @test ds.dim["time"] == 1
                @test ds.dim["x"] == MY.ncol
                @test ds.dim["z"] == MY.kDim
                @test ds["time"][1] == 0.0
                @test collect(ds["z"]) == MY.z_lay

                for v in ("K_m", "K_h", "e", "el", "sm", "sh", "cldfra_bl", "qc_bl",
                          "qi_bl", "vt", "vq", "P_s", "P_s_mynn", "P_b", "eps",
                          "tke_transport", "s_aw", "edmf_a", "edmf_w")
                    @test haskey(ds, v)
                    @test size(Array(ds[v])) == (1, MY.ncol, MY.kDim)
                end
                for v in ("pblh", "kpbl", "ust", "inv_L", "plume_ktop", "plume_ztop",
                          "aw_max", "bdry_E")
                    @test haskey(ds, v)
                    @test size(Array(ds[v])) == (1, MY.ncol)
                end

                # K_m/K_h round-trip BITWISE: reshape/permutedims is exact arithmetic.
                K_h_file = Array(ds["K_h"])[1, :, :]
                K_h_expected = permutedims(reshape(MY.K_h, kDim, MY.ncol))
                @test K_h_file == K_h_expected

                # `e = rho_e/rho_t` reconstructed independently here and compared.
                phys = mtile.tile.physical
                rho_tbar = view(Springsteel.ref_rho_t(mtile.ref_state), :, 1)
                rho_e_slot = mtile.mc_slots.rho_e
                e_file = Array(ds["e"])[1, :, :]
                for c in 1:MY.ncol, k in 1:kDim
                    j = (c - 1) * kDim + k
                    expected = phys[j, rho_e_slot, 1] / (phys[j, 3, 1] + rho_tbar[k])
                    @test e_file[c, k] ≈ expected rtol=1.0e-12
                end

                # `edmf_a`/`edmf_w` are not separately held (edmf off here) -- honest zero.
                @test all(==(0.0), Array(ds["edmf_a"]))
                @test all(==(0.0), Array(ds["edmf_w"]))

                # Attributes
                @test ds.attrib["Conventions"] == "CF-1.12"
                @test ds.attrib["init_mode"] == "taper"
                @test ds.attrib["water_carry"] == "flux"
                @test ds.attrib["n_clamp_e"] == 0
            end

            # ── N2: the sidecar is written FROM `mynn_diagnostics`, and the
            # comprehensive <t>.nc writes the same NamedTuple regridded. Every variable
            # in the file must therefore equal the corresponding diagnostics array
            # EXACTLY -- if the two ever diverge, the two output files disagree about
            # what a MYNN field is, which is precisely the failure the shared extraction
            # exists to make impossible.
            d = Scythe.mynn_diagnostics(mtile)
            NCDataset(path, "r") do ds
                @test collect(ds["x"]) == d.x
                @test collect(ds["z"]) == d.z
                for (nm, _, _) in Scythe.MYNN_FIELDS_2D
                    @test Array(ds[nm])[1, :, :] == getfield(d, Symbol(nm))
                end
                for (nm, _, _) in Scythe.MYNN_FIELDS_1D
                    @test Array(ds[nm])[1, :] == getfield(d, Symbol(nm))
                end
                @test Array(ds["edmf_a"])[1, :, :] == d.edmf_a
                @test Array(ds["edmf_w"])[1, :, :] == d.edmf_w
                # ...and the shared attribute list is what the file carries.
                for (k, v) in Scythe.mynn_global_attrs(MY)
                    @test ds.attrib[k] == v
                end
                @test issubset(Scythe.MYNN_COUNTER_ATTRS, keys(d.attrs))
            end
        end
    end

    @testset "mynn_write! is a no-op off the output cadence" begin
        mktempdir() do tmp
            mtile, patch, model, gp = make_io_tile(tmp)
            kDim = gp.kDim
            out_int = max(1, round(Int, model.output_interval / model.ts))
            @test out_int > 1
            spin_up!(mtile, kDim, 1)
            non_firing_t = 2
            @test mod(non_firing_t - 1, out_int) != 0
            Scythe.mynn_write!(mtile, non_firing_t)
            @test !isdir(model.output_dir) ||
                  isempty(filter(f -> endswith(f, ".nc"), readdir(model.output_dir)))

            firing_t = 1 + out_int
            Scythe.mynn_write!(mtile, firing_t)
            expected_tag = string(round((firing_t - 1) * model.ts; digits = 2))
            @test isfile(joinpath(model.output_dir, "$(expected_tag)_mynn_i0.nc"))
        end
    end

    @testset "no sidecar when :mynn is off" begin
        mktempdir() do tmp
            mtile, patch, model, gp = make_io_tile(tmp; mynn = false, outname = "off")
            @test !mtile.mynn.active
            kDim = gp.kDim
            spin_up!(mtile, kDim, 1)
            Scythe.mynn_write!(mtile, 1)
            Scythe.mynn_write_final!(mtile)
            @test !isdir(model.output_dir) ||
                  isempty(filter(f -> endswith(f, ".nc"), readdir(model.output_dir)))
        end
    end

    @testset "no sidecar with :mynn_output = false" begin
        mktempdir() do tmp
            mtile, patch, model, gp = make_io_tile(tmp; output = false, outname = "noout")
            @test mtile.mynn.active && !mtile.mynn.output
            kDim = gp.kDim
            spin_up!(mtile, kDim, 1)
            Scythe.mynn_write!(mtile, 1)
            Scythe.mynn_write_final!(mtile)
            @test !isdir(model.output_dir) ||
                  isempty(filter(f -> endswith(f, ".nc"), readdir(model.output_dir)))
        end
    end

    @testset "mynn_write_final! writes the end tag and folds counters" begin
        mktempdir() do tmp
            mtile, patch, model, gp = make_io_tile(tmp)
            kDim = gp.kDim
            spin_up!(mtile, kDim, 4)
            Scythe.mynn_write_final!(mtile)
            tag = string(round(model.integration_time; digits = 2))
            path = joinpath(model.output_dir, "$(tag)_mynn_i0.nc")
            @test isfile(path)
            NCDataset(path, "r") do ds
                @test ds["time"][1] == model.integration_time
                @test ds.attrib["n_clamp_e"] == mtile.mynn.n_clamp_e
            end
        end
    end

    @testset "mynn_snapshots lists tags" begin
        mktempdir() do tmp
            mtile, patch, model, gp = make_io_tile(tmp)
            kDim = gp.kDim
            @test isempty(Scythe.mynn_snapshots(model.output_dir))
            spin_up!(mtile, kDim, 1)
            Scythe.mynn_write!(mtile, 1)
            Scythe.mynn_write_final!(mtile)
            tags = Scythe.mynn_snapshots(model.output_dir)
            @test tags == ["0.0", string(round(model.integration_time; digits = 2))]
        end
    end

    @testset "read_mynn reassembles two tiles in offset order" begin
        mktempdir() do tmp
            mtile_a, patch_a, model_a, gp_a =
                make_io_tile(tmp; num_cells_i = 2, outname = "tile_a")
            mtile_b, patch_b, model_b, gp_b =
                make_io_tile(tmp; num_cells_i = 3, outname = "tile_b")
            kDim = gp_a.kDim
            spin_up!(mtile_a, kDim, 1)
            spin_up!(mtile_b, kDim, 1)

            dir = joinpath(tmp, "combined")
            mkpath(dir)
            tag = "0.0"
            Scythe._mynn_write_file!(joinpath(dir, "$(tag)_mynn_i0.nc"), mtile_a, 0.0)
            Scythe._mynn_write_file!(joinpath(dir, "$(tag)_mynn_i5.nc"), mtile_b, 0.0)

            snap = Scythe.read_mynn(dir, tag)
            @test length(snap.x) == mtile_a.mynn.ncol + mtile_b.mynn.ncol
            # First ncol_a columns come from tile A (offset 0), the rest from tile B
            # (offset 5): reshape each tile's own K_h the same way the writer does and
            # compare directly rather than re-deriving an index expression.
            K_h_a = permutedims(reshape(mtile_a.mynn.K_h, kDim, mtile_a.mynn.ncol))
            K_h_b = permutedims(reshape(mtile_b.mynn.K_h, kDim, mtile_b.mynn.ncol))
            @test snap.K_h[1:mtile_a.mynn.ncol, :] == K_h_a
            @test snap.K_h[(mtile_a.mynn.ncol + 1):end, :] == K_h_b
            @test snap.n_clamp_e == mtile_a.mynn.n_clamp_e + mtile_b.mynn.n_clamp_e
        end
    end

    @testset "the per-column allocation gate is unaffected" begin
        # The sidecar writer runs OUTSIDE the column loop (called once per tile per
        # output step, from the same site as `radiation_write!`), so it never touches
        # the per-column allocation budget `advance_column` is measured against. This is
        # asserted structurally rather than re-running the allocation harness here:
        # `mynn_write!`/`mynn_write_final!` are not called from anywhere inside
        # `mc_driver!`/`advance_column`/`_mynn_apply_column!`.
        src = read(joinpath(@__DIR__, "..", "src", "moist_compressible.jl"), String)
        @test !occursin("mynn_write!", src)
        @test !occursin("mynn_write_final!", src)
        src2 = read(joinpath(@__DIR__, "..", "src", "mc_mynn_bl.jl"), String)
        @test !occursin("mynn_write!(", src2)
    end
end
