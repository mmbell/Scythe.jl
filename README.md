# Scythe.jl

Scythe is a numerical weather prediction model written in Julia based on the spectral transform method. The model is semi-spectral as it uses a mixture of cubic B-spline (radial), Fourier (azimuthal), and Chebyshev (vertical) basis functions to represent physical variables and their spatial derivatives. Scythe currently supports 1-D radius (R), 2-D polar radius-azimuth (RL, where L denotes azimuthal angle lamba), 2-D radius-height (RZ), and 3-D cylindrical radius-azimuth-height (RLZ) grids. 

The current version 1.0.1 is primarily a dynamical core with only limited physics options. Three example models are provided, along with two Jupyter notebooks. For a quick start after installation, check out `notebooks/LinearAdvection_example.ipynb` which provides a very simple model to test that everything is working. The other well-tested models are two-layer shallow water/slab boundary layer models with one-way and two-way feedback options. Other models are still experimental.

Scythe has some limited multi-processing capabilities using both the Julia Distributed package and multi-threading. The current code supports only shared memory parallelization on a single compute node. The grid is represented as a "patch" with a shared spectral array, which can be decomposed in the radial direction into "tiles" according to the number of workers specified. Within each tile, individual threads are utilized for concurrency. 

### Installation

To install Scythe, in the Julia REPL, go into Package mode by pressing `]`. You will see the REPL change color and indicate `pkg` mode. 

If you would like to just install a static version of the latest release, you can get it directly from this repository with `add https://github.com/mmbell/Scythe.jl`

Alternatively, clone the repository and run `add /path/to/Scythe.jl` to add the package to your local registry. With both of these options you will need to manually update to new releases via `update Scythe` in the package manager.

If you are actively developing or modifying Scythe then you can install after cloning by using `dev /path/to/Scythe.jl` in `pkg` mode. This will update the module as changes are made to the code. You should see the dependencies being installed, and then the Scythe package will be precompiled. Everytime you make local changes to the code the package will update in this mode.

Exit Package mode with ctrl-C and test to make sure the precompilation was successful by running `using Scythe` in the REPL. If everything is successful then you should get no errors and it will just move to a new line.

### Running Scythe

To run a simulation, you first need to specify a `model` ModelParameters structure that contains the relevant information to construct a grid, a set of physical equations, time step, and initial conditions. After creating a model, you can integrate it with `integrate_model(model)`. The diagnostics will be directed to `scythe_out.log` and `scythe_err.log`, and the output will be directed to the specified directory in the model parameters. The current I/O support is only in CSV format.

A helper script called `run_Scythe.jl` is provided. It supports running multiple processors on a local node or laptop, or can spawn processes on a SGE cluster with the `--sge` flag. The number of worker processes can be specified with the optional `-w` flag, with a default of one worker. If you want multiple threads, then you must indicate that *before* you launch Julia with the `JULIA_NUM_THREADS` environmental variable, or with the `--threads` flag in front of the `run_Scythe.jl` script name and arguments.

The run script requires an argument of a Julia filename containing the model information. The provided Julia file must contain a ModelParameters structure at a minimum, but is included as Julia code so you could add extra functionality as desired.

Since the distributed mode launches the workers directly in code, there is no need to invoke `qsub` directly. That is handled directly by the ClusterManager object. As such, if you run from the command line it will wait until the job is done. A preferred option is to submit the script using `nohup` and a trailing & so that the job can run asynchronously. 

Email of job completion is supported with the `--email` flag, but currently it sends an email for every worker instead of a summary. This will be improved in a future version.

An example of the above commands would be:
```
nohup time  julia --threads 2 run_Scythe.jl -w 8 /path/to/model_file.jl &> scythe_run.log &
```

The above command includes `time` to get an overall wall clock time, and pipes the overarching output and errors to scythe_run.log. Note that the model diagnostics themselves are put in the directory containing model output. This log is for Julia errors and some limited status information.

### Future plans
Support for CF-compliant NetCDF input and output will be added in the near future. Future releases will include fully distributed parallelization for use on multi-node clusters. Future versions will also support different geometries that can be constructed from the underlying basis functions (e.g. Cartesian and spherical). Support for grid nesting using the cubic B-splines will be added in future versions. Future releases will also improve the cluster options for HPC use. Interested users are welcome to contribute to improve the model. Stay tuned for more functionality!

### Publications using Scythe

Cha, T.-Y. and Bell, M. M.: Tropical Cyclone Asymmetric Eyewall Evolution and Intensification in a Two-Layer Model, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-505, 2024.