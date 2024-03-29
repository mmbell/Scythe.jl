model = ModelParameters(
    ts = 0.05,
    integration_time = 100.0,
    output_interval = 100.0,
    equation_set = "LinearAdvection1D",
    initial_conditions = "1d_linear_advection_test_ics.csv",
    output_dir = "./linearAdvection1D_distributed/",
    grid_params = GridParameters(
        geometry = "R",
        xmin = -50.0,
        xmax = 50.0,
        num_cells = 100,
        BCL = Dict(
            "u" => CubicBSpline.PERIODIC),
        BCR = Dict(
            "u" => CubicBSpline.PERIODIC),
        vars = Dict(
            "u" => 1)),
    physical_params = Dict(
        :c_0 => 1.0,
        :K => 0.0))