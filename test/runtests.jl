using Test
using Scythe

@testset "Scythe.jl" begin
    include("test_thermodynamics.jl")
    include("test_microphysics.jl")
    include("test_bf02_restoration.jl")
    include("test_partial_density.jl")
    include("test_idealized_init.jl")
    include("test_spectralGrid.jl")
    include("test_reference_state.jl")
    include("test_reference_migration.jl")
    include("test_api_migration.jl")
    include("test_generic_transforms.jl")
    include("test_linear_advection_integration.jl")
    include("test_distributed_linear_advection.jl")
    include("test_distributed_rz.jl")
    include("test_oneway_sw_slab.jl")
    include("test_pv_mixing_floor.jl")
    include("test_benchmark_smoke.jl")
end
