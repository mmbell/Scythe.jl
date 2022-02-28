push!(LOAD_PATH, pwd())
using Integrator
using CubicBSpline
using SpectralGrid
using NumericalModels

nodes = 100
lpoints = 0
blpoints = 0
for r = 1:(nodes*3)
    global lpoints += 4 + 4*r
    global blpoints += 1 + 2*r
end
model = ModelParameters(
    ts = 3.0,
    integration_time = 10800.0,
    output_interval = 900.0,
    equation_set = "Oneway_ShallowWater_Slab",
    initial_conditions = "./SWslab_rankine_ics.csv",
    output_dir = "./SWslab_test/",
    grid_params = GridParameters(xmin = 0.0,
        xmax = 3.0e5,
        num_nodes = nodes,
        rDim = nodes*3,
        b_rDim = nodes+3,
        BCL = Dict(
            "h" => CubicBSpline.R1T1,
            "u" => CubicBSpline.R1T0,
            "v" => CubicBSpline.R1T0,
            "ub" => CubicBSpline.R1T0,
            "vb" => CubicBSpline.R1T0,
            "wb" => CubicBSpline.R1T1),
        BCR = Dict(
            "h" => CubicBSpline.R0,
            "u" => CubicBSpline.R0,
            "v" => CubicBSpline.R0,
            "ub" => CubicBSpline.R0,
            "vb" => CubicBSpline.R0,
            "wb" => CubicBSpline.R0),
        lDim = lpoints,
        b_lDim = blpoints,
        vars = Dict(
            "h" => 1,
            "u" => 2,
            "v" => 3,
            "ub" => 4,
            "vb" => 5,
            "wb" => 6)))

grid = initialize_model(model);
@time run_model(grid, model)
finalize_model(grid,model)
