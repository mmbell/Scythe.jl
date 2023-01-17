# Scythe

## Semi-spectral Cylindrical Transform Hurricane Emulator

Scythe is a numerical weather prediction model written in Julia. The model is based on the spectral transform method (Orszag 1970) and can solve the equations of motion in cylindrical coordinates.

The model uses a mixture of cubic B-spline (radial), Fourier (azimuthal), and Chebyshev (vertical) basis functions to represent the meteorological variables and their spatial derivatives. Scythe currently supports 1-D radius (R), 2-D radius-azimuth (RL, where L denotes azimuthal angle lamba), 2-D radius-height (RZ), and 3-D radius-azimuth-height (RLZ) grids. Future versions will support different geometries that can be constructed from the underlying basis functions (e.g. Cartesian).

To run a simulation, you first need to specify a `model` ModelParameters structure that contains the relevant information to construct a grid, a set of physical equations, time step, and initial conditions. Some examples are provided, and more will be added in the future.

After creating a model, you can integrate it with `integrate_model(model)`. The diagnostics will be directed to `scythe_out.log` and `scythe_err.log`, and the output will be directed to the specified directory in the model parameters. The current I/O is CSV format, which will be changed to CF-compliant NetCDF in future releases.

Scythe has parallel capabilities using the Julia Distributed package, and limited support for multi-threading. Grids are decomposed in the radial direction according to the number of processors specified. Currently only shared memory parallelization is supported, so run it on a single node.

### Installation

After cloning this repository, start Julia using Scythe.jl as the project directory. This can be done on the command line using `julia --project` or set the JULIA_PROJECT environmental variable:

`export JULIA_PROJECT=/path/to/Scythe.jl`

In the REPL, go into Package mode by pressing `]`.

You will see the REPL change color and indicate `pkg` mode. Run `activate` to activate the package environment.

Then, run `instantiate` to download the necessary dependencies. You should see the dependencies being installed, and then the Scythe package will be precompiled. Exit Package mode with ctrl-C.

Test to make sure the precompilation was successful by running `using Scythe` in the REPL. If everything is successful then you should get no errors and it will just move to a new line.

### Running Scythe

A helper script called `run_Scythe.jl` is provided. It supports running multiple processors on a local node or laptop, or can spawn processes on a SGE cluster with the `--sge` flag. The number of worker processes can be specified with the optional `-w` flag, with a default of one worker. If you want multiple threads, then you must indicate that when you launch Julia with the `--threads` flag *before* the `run_Scythe.jl` script name and arguments.

The run script requires an argument of a Julia filename containing the model information. The provided Julia file must contain a ModelParameters structure at a minimum, but is included as Julia code so you could add extra functionality as desired.

Since the distributed mode launches the workers directly in code, there is no need to call `qsub`. That is handled directly by the ClusterManager object. As such, if you run from the command line it will wait until the job is done. A preferred option is to submit the script using `nohup` and a trailing & so that the job can run asynchronously.

Email of job completion is supported with the `--email` flag, but currently it sends an email for every worker instead of a summary. This will be improved in a future version.

An example of the above commands would be:
```
nohup time  julia --threads 2 run_Scythe.jl -w 8 --sge /path/to/model_file.jl &> scythe_run.log &
```

The above command includes `time` to get an overall wall clock time, and pipes the overarching output and errors to scythe_run.log. Note that the model diagnostics themselves are put in the directory containing model output. This log is for Julia errors and some limited status information.

That's it! Good luck and stay tuned for more functionality!
