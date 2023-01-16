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
	action = :store_true
    "--email"
        help = "Email address for SGE"
        arg_type = String
        default = "none"
    "model"
        help = "Name of model parameters file"
	required = true
end
parsed_args = parse_args(ARGS, s)
num_workers = parsed_args["workers"]
run_sge = parsed_args["sge"]
email_address = parsed_args["email"]
modelfile = parsed_args["model"]
num_threads = Threads.nthreads()

email_flags = "n"
if (email_address != "none")
    email_flags = "eas"
end
println("Initializing with $(num_workers) workers and $(num_threads) threads")

if (run_sge)
    println("Using SGE for worker distribution")
    ClusterManagers.addprocs_sge(num_workers; qsub_flags=`-q all.q -pe mpi $(num_threads) -m $(email_flags) -M $(email_address)`)
else
    println("Using local node for worker distribution")
    addprocs(num_workers)
end

@everywhere using Scythe
include(modelfile)
integrate_model(model)
rmprocs(workers())

