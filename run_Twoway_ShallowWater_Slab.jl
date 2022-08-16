push!(LOAD_PATH, pwd())
using ArgParse
using Integrator
s = ArgParseSettings()
@add_arg_table s begin
    "icfile"
        help = "Name of initial condition csv file"
	required = true
end
parsed_args = parse_args(ARGS, s)
filename = parsed_args["icfile"]
integrate_Twoway_ShallowWater_Slab(filename)
