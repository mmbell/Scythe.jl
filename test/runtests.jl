using Test
using Scythe

@testset "Scythe.jl" begin
    include("test_thermodynamics.jl")
    include("test_microphysics.jl")
    include("test_spectralGrid.jl")
    include("test_reference_state.jl")
end
