__precompile__()
module Scythe

# Infrastructure for model grid representation
using Springsteel

# Re-export Springsteel's basis-agnostic BoundaryConditions type system
export BoundaryConditions, NaturalBC, DirichletBC, NeumannBC, SecondDerivativeBC
export RobinBC, PeriodicBC, CauchyBC, ExponentialBC, SymmetricBC, AntisymmetricBC

"""
    ModelParameters

Main configuration struct for Scythe model runs. Uses `Base.@kwdef` for keyword construction.

# Fields
- `ts::Float64`: model timestep [s] (default: `0.0`)
- `integration_time::Float64`: total integration duration [s] (default: `1.0`)
- `output_interval::Float64`: time between analysis output writes [s] (default: `1.0`)
- `restart_interval::Float64`: time between JLD2 restart-checkpoint writes [s] (default: `0.0` = no checkpoints). Written by [`write_restart`](@ref); independent of `output_interval` so checkpoints can be less frequent than analysis output. A resulting `<t>.jld2` can be used as `initial_conditions` to restart. The restart is WARM, not bit-identical: the checkpoint stores the grid state exactly but not the AB3 integrator's tendency history (see [`write_restart`](@ref)).
- `equation_set::String`: name of the equation set to solve (default: `"LinearAdvection1D"`)
- `initial_conditions::String`: path to the initial conditions file (default: `"ic.csv"`)
- `output_dir::String`: path to the output directory (default: `"./output/"`)
- `ref_state_file::String`: path to the reference state sounding file (default: `""`)
- `grid_params::SpringsteelGridParameters`: Springsteel grid configuration (required, no default)
- `physical_params::Dict{Symbol,Float64}`: physical parameters for the equation set (default: empty)
- `options::Dict{Symbol,Any}`: solver and output options (default: `Dict(:semiimplicit => false, :exact_reference_state => false)`). `:semiimplicit` is an opt-in for the LEGACY equation sets only; the moist_compressible (pressure-reference) sets are ALWAYS semi-implicit — the key may be omitted or `true`, and an explicit `false` is an error (the explicit acoustic mode was removed). Output-format keys read by [`write_output`](@ref):
    - `:output_formats::Vector{Symbol}` (default `[:netcdf]`) — which ANALYSIS formats to write each `output_interval`. `:netcdf` writes ONE comprehensive `<t>.nc` per output time (primes, totals, recovered hydrometeors, retrieved thermodynamics, radar/precipitation products, column integrals and the reference profiles) for the moist-compressible sets, and the prognostic-slot layout for everything else; `:csv` writes the `<t>_spectral.csv`/`<t>_physical.csv`/`<t>_gridded.csv` trio (Springsteel `write_grid`); `:netcdf_raw` writes the LEGACY prognostic-only gridded file to `<t>_raw.nc` (Springsteel `write_netcdf`). List several to write several, e.g. `[:csv, :netcdf]`. JLD2 is not listed here — it is the restart format, written at `restart_interval` (see `write_restart`). NOTE: the ICs reader, regression references, and benchmark harnesses parse the CSVs, so a run that must feed one of those has to list `:csv` explicitly — every benchmark under `benchmarks/` pins it.
    - `:netcdf_grid::Symbol` (default `:regular`) — the grid the NetCDF output is written on. `:mish` is reserved and not implemented.
    - `:netcdf_derivatives::Bool` (default `false`) — whether `:netcdf_raw` writes derivative slots alongside field values. Does not apply to `:netcdf`, whose derived products have no derivative slots.

`grid_params` is passed through Springsteel's `compute_derived_params` on construction, so a
cubic B-spline axis may be sized by *either* its cell count (`num_cells_i`/`num_cells_k`, the
canonical form) or its gridpoint count (`iDim`/`kDim`), and both fields are populated and
mutually consistent afterwards. Without this, supplying only `num_cells_k` would leave
`grid_params.kDim == 0` for every consumer that reads it (the grid factory resolves the counts
onto the grid it returns, not onto the caller's parameter struct).
"""
Base.@kwdef struct ModelParameters
    ts::Float64 = 0.0
    integration_time::Float64 = 1.0
    output_interval::Float64 = 1.0
    # JLD2 restart-checkpoint cadence [s]. 0.0 = no checkpoints. Independent of
    # output_interval so checkpoints (for restart) can be less frequent
    # than analysis output (CSV/NetCDF); see write_restart.
    restart_interval::Float64 = 0.0
    equation_set::String = "LinearAdvection1D"
    initial_conditions::String = "ic.csv"
    output_dir::String = "./output/"
    ref_state_file::String = ""
    grid_params::SpringsteelGridParameters
    # Concretely typed so that `model.physical_params[:Khdiff]` inside a per-column function
    # returns a `Float64` instead of boxing an `Any`. The inner constructor's `new` converts,
    # so a caller passing an integer value (`:K => 75`) still works.
    physical_params::Dict{Symbol,Float64} = Dict{Symbol,Float64}()
    # Deliberately left with an `Any` value type: options are not all Bool (e.g. a numeric
    # :cfl_interval). The handful of option reads that sit inside per-column functions carry
    # a `::Bool` assertion at the read site instead, which is what type-stability needs.
    options::Dict{Symbol,Any} = Dict{Symbol,Any}(
        :semiimplicit => false,
        :exact_reference_state => false)

    function ModelParameters(ts, integration_time, output_interval, restart_interval,
                             equation_set, initial_conditions, output_dir, ref_state_file,
                             grid_params, physical_params, options)
        # Eddy diffusivities are specified directly per quantity, not as molecular-style
        # ratios of the momentum coefficient: :Khdiff/:Kvdiff (momentum),
        # :Khdiff_heat/:Kvdiff_heat (heat, default = momentum values), and
        # :Khdiff_water/:Kvdiff_water (water species, default 0).
        for bad in (:Prandtl, :Schmidt)
            haskey(physical_params, bad) && error(
                "physical_params[:$bad] was removed: eddy mixing coefficients are not " *
                "molecular ratios. Set :Khdiff_heat/:Kvdiff_heat (heat) and " *
                ":Khdiff_water/:Kvdiff_water (water species) directly.")
        end
        new(ts, integration_time, output_interval, restart_interval, equation_set,
            initial_conditions, output_dir, ref_state_file,
            compute_derived_params(grid_params), physical_params, options)
    end
end

# Files for model integration
include("thermodynamics.jl")
include("reference_state.jl")
# The ISHMAEL TABLE layer comes before semiimplicit.jl because `ModelTile` carries an
# `IshmaelTables` field CONCRETELY (see `mc_ishmael_tables`): the struct definition has to
# exist before the one that names it. ishmael_tables.jl is a leaf — SpecialFunctions, JLD2
# and its own literals — so it can sit anywhere above its first user.
include("ishmael_tables.jl")
# MYNN-EDMF closure (plan S2): pure column functions on plain vectors, no ModelTile, no
# scratch, no options -- only MYNNConstants (built from Springsteel) -- so they stay testable
# against the Fortran reference driver with no model. The coupling (mc_mynn_bl.jl, S5) comes
# after mc_boundary_layer.jl.
include("mynn_constants.jl")
include("mynn_closure.jl")
include("mynn_edmf.jl")        # DMP_mf mass flux (S6); EDMFWork is defined here, after the closure
# The MYNN STATE layer, for the same reason radiation_state.jl sits where it does:
# `ModelTile` carries a `MYNNState` field CONCRETELY (see `EMPTY_MYNN`), so the struct has
# to be defined before the one that names it. It is a leaf over mynn_closure.jl and names
# no closure routine -- the coupling (mc_mynn_bl.jl, S5) comes after mc_boundary_layer.jl.
include("mynn_state.jl")
# The RADIATION STATE layer comes before semiimplicit.jl for the same reason: `ModelTile`
# carries a `RadiationState` field CONCRETELY (see `EMPTY_RADIATION`), so the struct has to
# be defined before the one that names it. radiation_state.jl is a leaf — the Springsteel
# thermodynamic constants imported by thermodynamics.jl and nothing else — and in
# particular it names no radiative-transfer library: every RRTMGP reference is confined to
# src/radiation.jl, which is included after moist_compressible.jl.
include("radiation_state.jl")
include("semiimplicit.jl")
include("testModels.jl")
include("shallowWaterModels.jl")
include("tcblModels.jl")
include("primitive_equations.jl")
include("io.jl")
# The ISHMAEL process-rate layer comes BEFORE microphysics.jl: the two-moment warm-rain rate
# functions there build top-level constants out of the ISHMAEL parameter block
# (ISHMAEL_RHOW, ISHMAEL_AR, ...), and a `const` initializer is evaluated at include
# time, not at first call. Neither ishmael file references anything from microphysics.jl,
# so the swap is inert in the other direction.
include("ishmael.jl")
include("microphysics.jl")
include("mc_geometry.jl")
# The SHARED SURFACE-EXCHANGE layer comes before mc_boundary_layer.jl because the Louis BL
# calls `surface_exchange` and carries a `SurfaceLayerParams` through its argument list; the
# MYNN-EDMF closure will call the same function. It is a leaf apart from `komori_cd`, which
# now lives there too — the bulk air-sea formulas are in ONE file, not two.
include("mc_surface_layer.jl")
include("mc_boundary_layer.jl")
# The MYNN-EDMF APPLY (S5). After mc_boundary_layer.jl because it reuses that
# file's tangential-wind trait accessors and its surface-delivery convention, and
# after mynn_closure.jl/mynn_state.jl because it calls the closure and reads the
# held state. It names ModelTile, the scratch pool and the geometry traits, which is
# exactly why it is NOT one of the three leaf mynn_* files (D11).
include("mc_mynn_bl.jl")
include("moist_compressible.jl")
# The microphysics -> cloud-optics conversion (Stage S3a) needs `_ice_effective`
# (moist_compressible.jl) and ISHMAEL_NU (ishmael.jl), and radiation.jl's driver calls
# it directly, so it sits between the two.
include("radiation_cloud_optics.jl")
# The RADIATION DRIVER comes after moist_compressible.jl because it reuses the driver's own
# thermodynamic helpers (`recover_rho_c`, `recover_total`, `retrieve_temperature`, the
# transform-mode accessors) to reconstruct a column — a radiation column that disagreed with
# `mc_driver!` about what the cloud IS would put that disagreement into the heating field.
# radiation.jl names no radiative-transfer library at all; radiation_rrtmgp.jl is the ONE
# file that names RRTMGP/ClimaComms, and it is included after it so the driver's interface
# calls resolve. radiation_io.jl (S5, the sidecar NetCDF writer/reader) follows immediately:
# it is the SECOND of the two files that name NCDatasets (radiation_rrtmgp.jl names it only
# to trigger RRTMGP's NCDatasets extension; radiation_io.jl is the one that actually reads
# and writes with it) and defines `radiation_write!`/`radiation_write_final!`, which
# `radiation_prepass!` above calls by name only at RUNTIME, so the include order relative to
# radiation.jl does not matter for those two — it matters for RRTMGP/ClimaComms above.
include("radiation.jl")
include("radiation_rrtmgp.jl")
include("radiation_io.jl")
include("mynn_io.jl")
# The comprehensive NetCDF analysis writer (Stage N1). After mynn_io.jl because it is
# the third and last file that names NCDatasets, and after moist_compressible.jl /
# mc_geometry.jl / microphysics.jl because every derived product it forms goes through
# the equation set's OWN helpers (retrieve_temperature, recover_rho_c, the transform-mode
# accessors, the rain fall speeds) -- a writer that re-derived any of them would be free
# to disagree with the run about what its own state means.
include("netcdf_output.jl")
include("horizontal_si.jl")
include("exact_si.jl")
include("exact_si_rlr.jl")
include("idealized.jl")
include("nesting.jl")

# Export the primary driver function and the ModelParameters
export integrate_model
export ModelParameters

"""
    integrate_model(model::ModelParameters)

Main entry point for running a Scythe simulation. Initializes worker processes,
redirects stdout/stderr to log files in the output directory, then runs the model
through its initialize, run, and finalize stages.

Requires at least one additional Julia worker process (added via `addprocs`).

# Arguments
- `model::ModelParameters`: the model configuration specifying equation set, grid, timing, and output options.

# Throws
- `ErrorException` if no worker processes are available.
"""
function integrate_model(model::ModelParameters)

    if workers()[1] == 1
        throw(ErrorException("Need to add at least 1 worker process"))
    end
    
    println("Starting model...")

    # Advisory startup check (master console, before the worker stdout redirect):
    # catch a timestep that is too large for the vertical resolution (e.g. raising
    # kDim without lowering ts). Warn-only; never aborts.
    warn_timestep_stability(model.grid_params, model.ts; equation_set=model.equation_set,
                            options=model.options)

    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end
    outfile = model.output_dir * "/scythe_out.log"
    errfile = model.output_dir * "/scythe_err.log"
    wait(save_at(workers()[1], :out, :(open($(outfile),"w"))))
    wait(save_at(workers()[1], :err, :(open($(errfile),"w"))))
    wait(get_from(workers()[1], :(redirect_stdout(out))))
    wait(get_from(workers()[1], :(redirect_stderr(err))))
    
    wait(save_at(workers()[1], :patch, :(initialize_model($(model),workers()))))
    # The comprehensive-NetCDF output context (src/netcdf_output.jl): the reference state
    # and the regular output coordinates the writer adds back. Built ONCE, here, on the
    # master worker that owns `patch`, and threaded into every write of the run -- it
    # constructs the reference state (reads and fits the run's .ref file), which must not
    # happen once per output interval. Inert (`active = false`, nothing built) unless the
    # run asks for :netcdf on a pressure-reference equation set.
    wait(save_at(workers()[1], :nc_ctx, :(Scythe.netcdf_output_context(patch, model))))
    wait(get_from(workers()[1], :(@time run_model(patch, model, workers(); ctx = nc_ctx))))
    wait(get_from(workers()[1], :(finalize_model(patch, model; ctx = nc_ctx,
                                                workerids = workers()))))
    
    wait(get_from(workers()[1], :(close(out))))
    wait(get_from(workers()[1], :(close(err))))
    println("All done!")
end

# Module end
end