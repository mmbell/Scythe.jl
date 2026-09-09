# test/test_groups.jl -- the suite's file table, in canonical order, tagged by group.
#
# `julia --project=. test/runtests.jl` with no GROUP runs every file below in this order
# (the historical serial suite, ~35 min). `GROUP=<name>[,<name>...]` runs only those groups,
# so the suite can be split across shells locally or across a CI job matrix. The groups are
# cut by COST, not by count: `allocations` is one file that alone takes ~11 min (it compiles
# every equation-set body), `nesting` spawns worker processes, `moist` is one large file, and
# the parity files are thousands of fast comparisons against the Fortran references.
#
# Every test file is self-contained (its own `using`), so any subset runs on its own.

const TEST_FILES = [
    ("test_thermodynamics.jl",              :core),
    ("test_microphysics.jl",                :core),
    ("test_ishmael_tables.jl",              :parity),
    ("test_ishmael.jl",                     :parity),
    ("test_mynn_closure.jl",                :parity),
    ("test_mynn_edmf.jl",                   :parity),
    ("test_bf02_restoration.jl",            :core),
    ("test_partial_density.jl",             :core),
    ("test_sigma_entropy.jl",               :core),
    ("test_etd_relaxation.jl",              :core),
    ("test_ice_anchor_reconcile.jl",        :core),
    ("test_moist_compressible.jl",          :moist),
    ("test_radiation.jl",                   :physics),
    ("test_radiation_driver.jl",            :physics),
    ("test_radiation_rrtmgp.jl",            :physics),
    ("test_radiation_cloud_optics.jl",      :physics),
    ("test_radiation_io.jl",                :io),
    ("test_louis_bl.jl",                    :physics),
    ("test_surface_fluxes.jl",              :physics),
    ("test_surface_layer.jl",               :physics),
    ("test_mynn_driver.jl",                 :physics),
    ("test_mynn_bl.jl",                     :physics),
    ("test_mynn_fidelity.jl",               :physics),
    ("test_mynn_edmf_live.jl",              :physics),
    ("test_mynn_ice.jl",                    :physics),
    ("test_mynn_io.jl",                     :io),
    ("test_idealized_init.jl",              :core),
    ("test_spectralGrid.jl",                :core),
    ("test_output_formats.jl",              :io),
    ("test_netcdf_output.jl",               :io),
    ("test_reference_state.jl",             :core),
    ("test_reference_migration.jl",         :core),
    ("test_api_migration.jl",               :core),
    ("test_generic_transforms.jl",          :core),
    ("test_linear_advection_integration.jl", :nesting),
    ("test_nesting_config.jl",              :nesting),
    ("test_nested_advection_oneway.jl",     :nesting),
    ("test_nested_advection_twoway.jl",     :nesting),
    ("test_nested_advection_subcycle.jl",   :nesting),
    ("test_distributed_nesting.jl",         :nesting),
    ("test_distributed_linear_advection.jl", :nesting),
    ("test_distributed_rz.jl",              :nesting),
    ("test_oneway_sw_slab.jl",              :nesting),
    ("test_pv_mixing_floor.jl",             :core),
    ("test_pv_mixing_stochastic.jl",        :core),
    ("test_spline_factor.jl",               :core),
    ("test_allocations.jl",                 :allocations),
    ("test_benchmark_smoke.jl",             :io),
]

const TEST_GROUPS = (:core, :parity, :moist, :physics, :io, :nesting, :allocations)

"""
    test_files_for(group::AbstractString) -> Vector{String}

The files to run for `GROUP`: `"all"` (or empty) is every file in canonical order; otherwise
a comma-separated list of names from `TEST_GROUPS`, kept in canonical order. An unknown
name errors before anything is loaded.
"""
function test_files_for(group::AbstractString)
    g = strip(group)
    (isempty(g) || g == "all") && return first.(TEST_FILES)
    wanted = Symbol.(strip.(split(g, ',')))
    for w in wanted
        w in TEST_GROUPS || error("GROUP=$(g): \"$(w)\" is not a test group. " *
                                  "Use all or a comma-separated subset of " *
                                  join(string.(TEST_GROUPS), ", ") * ".")
    end
    return [f for (f, tag) in TEST_FILES if tag in wanted]
end
