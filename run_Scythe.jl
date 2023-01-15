using Distributed
using ClusterManagers
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--workers", "-w"
        help = "Number of worker processes"
        arg_type = Int
        default = 1
    "--sge"
        help = "Run on Sun Grid Engine"
        arg_type = Bool
        default = false
    "model"
        help = "Name of model parameters file"
	    required = true
end
parsed_args = parse_args(ARGS, s)
num_workers = parsed_args["workers"]
run_sge = parsed_args["sge"]
modelfile = parsed_args["model"]
num_threads = Threads.nthreads()

println("Initializing with $(num_workers) workers and $(num_threads) threads")
if (run_sge)
    println("Using SGE for worker distribution")
    ClusterManagers.addprocs_sge(num_workers; qsub_flags=`-q all.q -pe mpi $(num_threads)`)
else
    println("Using local node for worker distribution")
    addprocs(num_workers)
end

@everywhere using Scythe
include(modelfile)
integrate_model(model)
rmprocs(workers())

