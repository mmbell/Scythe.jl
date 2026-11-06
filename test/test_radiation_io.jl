using Test
using Scythe
using Springsteel
using SparseArrays
using NCDatasets

# Stage S5: the radiation sidecar writer (`radiation_write!`/`radiation_write_final!`,
# src/radiation_io.jl) and reader (`read_radiation`/`radiation_snapshots`).
#
# Uses the SAME `:prescribed` O01-like fixture `test_radiation_driver.jl` builds
# (`make_o01_rad_mtile`), reproduced here rather than reached across files (its testset
# defines it in a local scope) -- no lookup artifacts, no network, no solver, so the
# writer's shape/round-trip can be tested without RRTMGP.

@testset "Radiation sidecar I/O (S5)" begin

"""A small `moist_compressible_XZ` RiRk tile with `:radiation => :prescribed`, the
artifact-free scheme -- see test_radiation_driver.jl's `make_o01_rad_mtile` for the full
rationale (reproduced here so this file has no cross-testset dependency)."""
function make_rad_io_mtile(tmpdir; num_cells_i = 2, num_cells_k = 100,
                           extra_opts = Dict{Symbol,Any}(), outname = "out")
    sounding = joinpath(@__DIR__, "..", "benchmarks", "reference_data",
                        "o01_rainfall", "dunion_MT_hum90.ref")
    isfile(sounding) || error("the O01 sounding is missing: $sounding")
    varlist = Scythe.mc_var_names(Dict{Symbol,Any}(); cyl = false)
    rain_name = Scythe.rain_var_name(Dict{Symbol,Any}())
    vars = Dict(v => i for (i, v) in enumerate(varlist))
    scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
    side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
    topbot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
    gp = GridParameters(geometry = "RiRk",
        iMin = 0.0, iMax = 150.0e3, num_cells_i = num_cells_i,
        kMin = 0.0, kMax = 25.0e3, num_cells_k = num_cells_k,
        BCL = side_bc, BCR = side_bc, BCB = topbot_bc, BCT = topbot_bc, vars = vars)
    ref_file = joinpath(tmpdir, "o01_rad_io_$(num_cells_i)x$(num_cells_k).ref")
    model = ModelParameters(
        ts = 0.3, integration_time = 1.0, output_interval = 1.0,
        equation_set = "moist_compressible_XZ",
        output_dir = joinpath(tmpdir, outname),
        ref_state_file = ref_file, grid_params = gp,
        physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Khdiff_heat => 0.0,
                               :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                               :tau_qss => 10.0, :alpha => 0.02, :z_damp => 17.0e3,
                               :radiation_prescribed_rate => -1.5),   # matches the default
        options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                         :exact_reference_state => true,
                                         :precipitation => true,
                                         :vertical_mixing => false,
                                         :radiation => :prescribed,
                                         :radiation_trace => false,
                                         # N2: the sidecar is opt-in (the fields ride in
                                         # the comprehensive <t>.nc now), and this file
                                         # is the sidecar's own test, so it asks. The
                                         # "noout" arm below overrides it back to false.
                                         :radiation_output => true),
                        extra_opts),
    )
    gp = model.grid_params
    patch = createGrid(gp)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = gp.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, gp)
    ref_phys = Springsteel.calculate_pressure_reference_state(sounding, z, column)
    Scythe.write_exact_ref_mc(ref_file, z,
                              Springsteel.ref_pressure(ref_phys)[:, 1],
                              Springsteel.ref_rho_d(ref_phys)[:, 1],
                              Springsteel.ref_rho_v(ref_phys)[:, 1],
                              zeros(kDim))
    patch.physical .= 0.0
    spectralTransform!(patch)
    gridTransform!(patch)
    hrm = sparse(Int64[], Int64[], Float64[],
                 size(patch.spectral, 1), size(patch.spectral, 2))
    mtile = createModelTile(patch, patch, model, hrm)
    return mtile, patch, model, gp, z
end

@testset "radiation_write! fires on cadence and round-trips" begin
    mktempdir() do tmp
        mtile, patch, model, gp, z = make_rad_io_mtile(tmp)
        rs = mtile.radiation
        kDim = gp.kDim
        @test rs.active

        # Fire the update once (t = 1, the forced first call) so q_lw/flux fields are
        # non-trivial, then write the sidecar for that same step.
        Scythe.radiation_update!(mtile, 1)
        @test any(!=(0.0), rs.q_lw)     # the -1.5 K/day prescribed rate is non-zero

        Scythe.radiation_write!(mtile, 1)
        tag = "0.0"
        path = joinpath(model.output_dir, "$(tag)_radiation_i0.nc")
        @test isfile(path)

        NCDataset(path, "r") do ds
            @test ds.dim["time"] == 1
            @test ds.dim["x"] == rs.ncol
            @test ds.dim["z"] == rs.nlay
            @test ds.dim["zf"] == rs.nlay + 1     # :prescribed carries no extension

            @test ds["time"][1] == 0.0
            @test collect(ds["z"]) == rs.z
            @test collect(ds["zf"]) == rs.z_face

            # q_lw round-trips BITWISE: reshape(kDim,ncol) then permutedims is exact
            # arithmetic (no floating-point operation), so this must be exact, not `≈`.
            q_lw_file = Array(ds["q_lw"])[1, :, :]           # (x, z)
            q_lw_expected = permutedims(reshape(rs.q_lw, kDim, rs.ncol))
            @test q_lw_file == q_lw_expected

            q_sw_file = Array(ds["q_sw"])[1, :, :]
            @test all(==(0.0), q_sw_file)                    # :prescribed sets q_sw = 0

            # dT_lw = q_lw / (rho * c_p) * 86400, the SAME `_rad_kday` helper the trace
            # uses. Recompute independently via `radiation_column_state!` and compare.
            work = Scythe.RadiationWork(kDim)
            dT_lw_file = Array(ds["dT_lw"])[1, :, :]
            for c in 1:rs.ncol
                cs = (c - 1) * kDim + 1
                Scythe.radiation_column_state!(work, mtile, cs, cs + kDim - 1)
                for k in 1:kDim
                    f = Scythe._rad_kday(work, k)
                    expected = rs.q_lw[cs + k - 1] * f
                    @test dT_lw_file[c, k] ≈ expected rtol=1.0e-12
                end
            end

            # Flux matrices: :prescribed never runs a solver, so they are identically
            # zero, and the sidecar must report that honestly rather than omit them.
            @test all(==(0.0), Array(ds["flux_lw_up"]))
            @test all(==(0.0), Array(ds["flux_sw_up"]))
            @test all(==(0.0), Array(ds["olr"]))
            @test all(==(0.0), Array(ds["lwp"]))
            @test all(==(0.0), Array(ds["iwp"]))
            @test all(==(0.0), Array(ds["cloudy"]))

            # Scalars/attrs
            @test ds.attrib["scheme"] == "prescribed"
            @test ds.attrib["forcing"] == "full"
            @test ds.attrib["Conventions"] == "CF-1.12"
            @test ds.attrib["sw_scale"] == rs.sw_scale
            @test ds.attrib["cos_zenith"] == rs.cos_zenith
            @test !haskey(ds, "q_lw_ref")     # :full forcing carries no reference profile
        end

        # ── N2: the sidecar is written FROM `radiation_diagnostics`, and the
        # comprehensive <t>.nc writes the same NamedTuple regridded. Every variable in
        # the file must equal the corresponding diagnostics array EXACTLY -- if they
        # ever diverge, the two output files disagree about what a radiation field is.
        d = Scythe.radiation_diagnostics(mtile)
        NCDataset(path, "r") do ds
            @test collect(ds["x"]) == d.x
            @test collect(ds["z"]) == d.z
            @test collect(ds["zf"]) == d.zf
            for (nm, _, _) in Scythe.RADIATION_FIELDS_2D
                @test Array(ds[nm])[1, :, :] == getfield(d, Symbol(nm))
            end
            for (nm, _, _) in Scythe.RADIATION_FIELDS_FACE
                @test Array(ds[nm])[1, :, :] == getfield(d, Symbol(nm))
            end
            for (nm, _, _) in Scythe.RADIATION_FIELDS_1D
                @test Array(ds[nm])[1, :] == getfield(d, Symbol(nm))
            end
            for (k, v) in Scythe.radiation_global_attrs(rs, model.ts)
                @test ds.attrib[k] == v || (v isa Number && isnan(v) &&
                                            ds.attrib[k] isa Number && isnan(ds.attrib[k]))
            end
            @test issubset(Scythe.RADIATION_COUNTER_ATTRS, keys(d.attrs))
        end
    end
end

@testset "radiation_write! is a no-op off the output cadence" begin
    mktempdir() do tmp
        mtile, patch, model, gp, z = make_rad_io_mtile(tmp)
        # output_interval = 1.0 s, ts = 0.3 s -> out_int = round(1.0/0.3) = 3 steps.
        out_int = max(1, round(Int, model.output_interval / model.ts))
        @test out_int > 1
        Scythe.radiation_update!(mtile, 1)
        # A step that is NOT a multiple-of-out_int offset from step 1 must write nothing.
        non_firing_t = 2                                  # mod(2-1, out_int) != 0
        @test mod(non_firing_t - 1, out_int) != 0
        Scythe.radiation_write!(mtile, non_firing_t)
        @test !isdir(model.output_dir) ||
              isempty(filter(f -> endswith(f, ".nc"), readdir(model.output_dir)))

        firing_t = 1 + out_int                             # mod(firing_t-1, out_int) == 0
        Scythe.radiation_write!(mtile, firing_t)
        expected_tag = string(round((firing_t - 1) * model.ts; digits = 2))
        @test isfile(joinpath(model.output_dir, "$(expected_tag)_radiation_i0.nc"))
    end
end

@testset "radiation_write! is a no-op when radiation is off or output is disabled" begin
    mktempdir() do tmp
        # Radiation OFF entirely.
        mtile_off, patch_off, model_off, _, _ = make_rad_io_mtile(tmp;
            extra_opts = Dict{Symbol,Any}(:radiation => :none), outname = "off")
        @test !mtile_off.radiation.active
        Scythe.radiation_write!(mtile_off, 1)
        Scythe.radiation_write_final!(mtile_off, model_off.integration_time)
        (!isdir(model_off.output_dir) ||
         isempty(filter(f -> endswith(f, ".nc"), readdir(model_off.output_dir)))) ||
            error("radiation-off tile wrote a sidecar")

        # Radiation ON but output disabled.
        mtile_noout, patch_noout, model_noout, _, _ = make_rad_io_mtile(tmp;
            extra_opts = Dict{Symbol,Any}(:radiation_output => false), outname = "noout")
        @test mtile_noout.radiation.active && !mtile_noout.radiation.output
        Scythe.radiation_update!(mtile_noout, 1)
        Scythe.radiation_write!(mtile_noout, 1)
        Scythe.radiation_write_final!(mtile_noout, model_noout.integration_time)
        (!isdir(model_noout.output_dir) ||
         isempty(filter(f -> endswith(f, ".nc"), readdir(model_noout.output_dir)))) ||
            error("radiation_output=false tile wrote a sidecar")
    end
end

@testset "radiation_write_final! writes the end tag" begin
    mktempdir() do tmp
        mtile, patch, model, gp, z = make_rad_io_mtile(tmp)
        Scythe.radiation_update!(mtile, 1)
        Scythe.radiation_write_final!(mtile, model.integration_time)
        tag = string(round(model.integration_time; digits = 2))
        path = joinpath(model.output_dir, "$(tag)_radiation_i0.nc")
        @test isfile(path)
        NCDataset(path, "r") do ds
            @test ds["time"][1] == model.integration_time
        end
    end
end

@testset ":anomaly forcing writes q_lw_ref/q_sw_ref" begin
    mktempdir() do tmp
        mtile, patch, model, gp, z = make_rad_io_mtile(tmp;
            extra_opts = Dict{Symbol,Any}(:radiation_forcing => :anomaly))
        rs = mtile.radiation
        @test rs.forcing === :anomaly
        Scythe.radiation_update!(mtile, 1)
        Scythe.radiation_write!(mtile, 1)
        path = joinpath(model.output_dir, "0.0_radiation_i0.nc")
        @test isfile(path)
        NCDataset(path, "r") do ds
            @test haskey(ds, "q_lw_ref")
            @test haskey(ds, "q_sw_ref")
            @test collect(ds["q_lw_ref"][1, :]) == rs.q_lw_ref
            @test collect(ds["q_sw_ref"][1, :]) == rs.q_sw_ref
            # The tile is at rest, so `radiation_store!` subtracts the RESTING REFERENCE
            # COLUMN's heating from the model column's own (bitwise identical on a
            # resting column), leaving q_lw exactly zero -- the far-field-zero property
            # S4 built the reference-column definition for. q_lw_ref itself is left
            # un-subtracted (it IS the reference, not the anomaly).
            @test all(iszero, rs.q_lw)
            @test any(!=(0.0), rs.q_lw_ref)   # the reference itself is a real profile
        end
    end
end

@testset "radiation_snapshots lists tags" begin
    mktempdir() do tmp
        mtile, patch, model, gp, z = make_rad_io_mtile(tmp)
        @test isempty(Scythe.radiation_snapshots(model.output_dir))
        Scythe.radiation_update!(mtile, 1)
        Scythe.radiation_write!(mtile, 1)
        Scythe.radiation_write_final!(mtile, model.integration_time)
        tags = Scythe.radiation_snapshots(model.output_dir)
        @test tags == ["0.0", string(round(model.integration_time; digits = 2))]
    end
end

@testset "read_radiation reassembles two tiles in offset order" begin
    mktempdir() do tmp
        # Two INDEPENDENT tiles standing in for two workers' slices of one patch: same
        # vertical grid (kDim), different horizontal extents/positions, written to the
        # SAME directory under the SAME tag with different `_i<offset>` suffixes via the
        # internal single-tile writer (there is no multi-worker run in a unit test).
        mtile_a, patch_a, model_a, gp_a, z =
            make_rad_io_mtile(tmp; num_cells_i = 2, outname = "tile_a")
        mtile_b, patch_b, model_b, gp_b, _ =
            make_rad_io_mtile(tmp; num_cells_i = 3, outname = "tile_b")
        rs_a = mtile_a.radiation; rs_b = mtile_b.radiation
        Scythe.radiation_update!(mtile_a, 1)
        Scythe.radiation_update!(mtile_b, 1)

        dir = joinpath(tmp, "combined")
        mkpath(dir)
        t_model = 0.0
        tag = "0.0"
        # Offsets 0 and 5 -- arbitrary but distinct and NOT already sorted by call order,
        # so a read that failed to sort by offset (e.g. sorted by readdir order) would be
        # caught by the monotonic-x assertion below.
        path_a = joinpath(dir, "$(tag)_radiation_i5.nc")
        path_b = joinpath(dir, "$(tag)_radiation_i0.nc")
        Scythe._radiation_write_file!(path_a, mtile_a, t_model)
        Scythe._radiation_write_file!(path_b, mtile_b, t_model)

        out = Scythe.read_radiation(dir, tag)
        @test length(out.x) == rs_a.ncol + rs_b.ncol
        # Tile at offset 0 (mtile_b, 3 cells) comes FIRST, tile at offset 5 (mtile_a, 2
        # cells) comes SECOND -- offset order, not write order or ncol order.
        @test length(out.x) == rs_b.ncol + rs_a.ncol
        @test out.x[1:rs_b.ncol] == [mtile_b.tilepoints[(c - 1) * gp_b.kDim + 1, 1]
                                      for c in 1:rs_b.ncol]
        @test out.x[(rs_b.ncol + 1):end] == [mtile_a.tilepoints[(c - 1) * gp_a.kDim + 1, 1]
                                              for c in 1:rs_a.ncol]
        @test size(out.q_lw) == (rs_a.ncol + rs_b.ncol, rs_b.nlay)
        @test out.q_lw[1:rs_b.ncol, :] ==
              permutedims(reshape(rs_b.q_lw, gp_b.kDim, rs_b.ncol))
        @test out.q_lw[(rs_b.ncol + 1):end, :] ==
              permutedims(reshape(rs_a.q_lw, gp_a.kDim, rs_a.ncol))
        @test out.z == rs_b.z
        @test out.scheme == "prescribed"
        @test out.n_clamp_tk == rs_a.n_clamp_tk + rs_b.n_clamp_tk
    end
end

@testset "read_radiation errors on a missing tag" begin
    mktempdir() do tmp
        @test_throws ErrorException Scythe.read_radiation(tmp, "999.0")
    end
end

end # @testset "Radiation sidecar I/O (S5)"
