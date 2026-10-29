using Test
using Scythe
using Springsteel
using SparseArrays
using SharedArrays

# Stream C2/R2c: the per-step tile exchange on the RRR (x, y, z) box. Every step a tile
# sends the 3-row right halo of each of its spectral i-blocks (`extract_halo_values`),
# writes its inner rows into the patch's shared B-vector (`write_tile_to_shared!`) and adds
# the left neighbour's halo at its receive map (`accumulate_at_map!`). The RRR layout is
# b_jDim*b_kDim i-blocks (block q = (z-1)*b_jDim + l), so the RiRk methods generalize with
# b_kDim -> b_jDim*b_kDim. Before R2c the RRR grid fell to the generic method, which treats
# any multi-block array as the RL wavenumber layout.
#
# The check needs no workers: the i-direction B-vector of a field is additive over tiles
# (each tile's SB transform integrates its own mish against the basis functions it
# shares), so the assembled shared array must equal the PATCH's own (unfiltered) forward
# transform to roundoff, whatever the tile count.

# The exchange needs Springsteel's RRR i-tiling maps (its rrr/halo branch, d74ea56); an
# older Springsteel falls to the generic first-block maps, the run's preflight refuses RRR
# there by the same method lookup (run_driver.jl), and this testset is skipped, loudly.
const RRR_TILING_AVAILABLE = hasmethod(Springsteel.tileTransform!,
    Tuple{SharedArray{Float64}, RRR_Grid, Array{Float64}, Array{Float64}})
RRR_TILING_AVAILABLE || @warn("test_rrr_tiling: $(pkgdir(Springsteel)) has no RRR tiling " *
    "methods (calcPatchMap/calcHaloMap/splineTransform!/tileTransform! on RRR_Grid); the " *
    "tile-exchange testset is skipped and run_simulation refuses RRR with this Springsteel")

RRR_TILING_AVAILABLE && @testset "RRR tile exchange (extract_halo_values / write_tile_to_shared!)" begin
    vars = Dict("u" => 1, "v" => 2, "w" => 3)
    bc = Dict(v => NeumannBC() for v in keys(vars))
    gp = GridParameters(geometry = "RRR",
        iMin = 0.0, iMax = 36.0e3, num_cells_i = 12,
        jMin = 0.0, jMax = 12.0e3, num_cells_j = 4,
        kMin = 0.0, kMax = 4.0e3, num_cells_k = 4,
        BCL = bc, BCR = bc, BCD = bc, BCU = bc, BCB = bc, BCT = bc, vars = vars)
    patch = createGrid(gp)
    pp = Scythe.getGridpoints(patch)
    f(x, y, z, v) = (1.0 + v) * exp(-((x - 15.0e3) / 7.0e3)^2) *
                    (1.0 + 0.4 * sin(2pi * y / 12.0e3)) * (0.5 + z / 4.0e3)
    for i in axes(pp, 1), v in 1:3
        patch.physical[i, v, 1] = f(pp[i, 1], pp[i, 2], pp[i, 3], v)
    end
    Bpatch = Springsteel.spectralTransform(patch, patch.physical, similar(patch.spectral))
    nblk = patch.params.b_jDim * patch.params.b_kDim

    # The method lookups the preflight reads (src/run_driver.jl): RRR has its own now.
    @test which(Scythe.extract_halo_values, Tuple{RRR_Grid}).sig !=
          Tuple{typeof(Scythe.extract_halo_values), Springsteel.AbstractGrid}

    for ntiles in (1, 2, 3)
        tiles = calcTileSizes(patch, ntiles)
        shared = SharedArray{Float64}(size(patch.spectral))
        shared .= 0.0
        halos = Vector{Vector{Float64}}(undef, ntiles)
        for (n, tile) in enumerate(tiles)
            tp = Scythe.getGridpoints(tile)
            for i in axes(tp, 1), v in 1:3
                tile.physical[i, v, 1] = f(tp[i, 1], tp[i, 2], tp[i, 3], v)
            end
            Springsteel.spectralTransform(tile, tile.physical, tile.spectral)
            halos[n] = Scythe.extract_halo_values(tile)
            @test length(halos[n]) == 3 * 3 * nblk
            Scythe.write_tile_to_shared!(shared, tile, patch.params.b_iDim)
        end
        # each tile's halo goes to the next one (the last one's to the master); the
        # receive map of tile n+1 is tile n's send map, calcHaloMap(patch, tile n)
        for n in 1:ntiles
            Scythe.accumulate_at_map!(shared, calcHaloMap(patch, tiles[n]), halos[n])
        end
        err = maximum(abs.(shared .- Bpatch)) / maximum(abs, Bpatch)
        @info "RRR tile exchange, $ntiles tile(s): max |assembled - patch B| / max|B| = $err"
        @test err < 1.0e-13
    end
end

# Stream C2/R2e: the per-variable vertical work columns. Springsteel's RRR k-basis is an
# (iDim, jDim, nvars) array of per-column splines (every other geometry carries ONE k spline
# per variable), so the linear index `kbasis.data[v]` the scratch/flux-column allocation used
# picked column (v, 1) of variable 1 -- every vertical SI / diffusion solve on the box ran
# with p's boundary conditions (w lost its Dirichlet ground and lid). Found by the 40-step
# slab == XZ run (test_3d_rrr_timestepping.jl), not by the single-tendency tests, which never
# touch these columns.
@testset "RRR vertical work columns carry each variable's own k-basis" begin
    vars = Dict("p" => 1, "u" => 2, "w" => 3)
    bc = Dict(v => NeumannBC() for v in keys(vars))
    wbc = merge(bc, Dict("w" => DirichletBC()))
    for (geom, extra) in (("RRR", (; jMin = 0.0, jMax = 12.0e3, num_cells_j = 4, BCD = bc, BCU = bc)),
                          ("RiRk", (;)))
        gp = GridParameters(; geometry = geom, iMin = 0.0, iMax = 36.0e3, num_cells_i = 12,
            kMin = 0.0, kMax = 4.0e3, num_cells_k = 4, extra...,
            BCL = bc, BCR = bc, BCB = wbc, BCT = wbc, vars = vars)
        tile = createGrid(gp)
        cols = Scythe._allocate_scratch_columns(tile.kbasis, tile)
        flux = Scythe._allocate_flux_columns(tile.kbasis, tile, cols)
        for (name, v) in vars
            own = Scythe._kbasis_column(tile.kbasis, v)
            @test own === (geom == "RRR" ? tile.kbasis.data[1, 1, v] : tile.kbasis.data[v])
            @test cols[1, v].params == own.params
            @test flux[1, v].params.BCL == own.params.BCL           # the ground is the slot's
        end
        @test cols[1, vars["w"]].params.BCL != cols[1, vars["p"]].params.BCL
    end
end
