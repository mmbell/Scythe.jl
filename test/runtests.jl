using Test
using Scythe

@testset "Scythe.jl" begin
    include("test_thermodynamics.jl")
    include("test_microphysics.jl")
    include("test_spectralGrid.jl")
    include("test_reference_state.jl")
    include("test_api_migration.jl")
    include("test_linear_advection_integration.jl")
    include("test_distributed_linear_advection.jl")
    include("test_oneway_sw_slab.jl")
end
