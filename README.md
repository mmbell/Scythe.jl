# Scythe

## Semi-spectral Cylindrical Transform Hurricane Emulator

Scythe is a numerical weather prediction model written in Julia. The model is based on the spectral transform method (Orszag 1970) and is unique in its ability to solve the equations of motion in cylindrical coordinates.

The model uses a mixture of cubic B-spline (radial), Fourier (azimuthal), and Chebyshev (vertical) basis functions to represent the meteorological variables and their spatial derivatives. Scythe currently supports 1-D radius (R), 2-D radius-azimuth (RL, where L denotes azimuthal angle lamba), 2-D radius-height (RZ), and 3-D radius-azimuth-height (RLZ) grids. Future versions will support different geometries that can be constructed from the underlying basis functions (e.g. Cartesian).

To run a simulation, you first need to specify a `model` ModelParameters structure that contains the relevant information to construct a grid, a set of physical equations, time step, and initial conditions. Some examples are provided, and more will be added in the future.

After creating a model, you can integrate it with `integrate_model(model)`. The diagnostics will be directed to `scythe_out.log` and `scythe_err.log`, and the output will be directed to the specified directory in the model parameters. The current I/O is CSV format, which will be changed to CF-compliant NetCDF in future releases.

Scythe has parallel capabilities using the Julia Distributed package, and limited support for multi-threading. Grids are decomposed in the radial direction according to the number of processors specified. Currently only shared memory parallelization is supported, so run it on a single node.

Stay tuned for more functionality!
