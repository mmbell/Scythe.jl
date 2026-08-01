# Shared preflight for the TC sbatch wrappers. Sourced, not executed.
# Expects $JULIA and $SCYTHE to be set. Exits the caller on failure.
#
# WHY THIS EXISTS. Springsteel is a REGISTERED package, so `Pkg.instantiate` on a fresh
# checkout will happily resolve the released v1.0.0 -- which is 59 commits behind what the TC
# configuration needs and contains none of `_hydrostatic_pressure_profile`, `CubicBSpline.R1T1X`
# or the `hydrostatic` keyword of `calculate_pressure_reference_state`. tc_init.jl sets
# options[:hydrostatic_reference] = true, so that resolution dies -- but on an unknown-keyword
# MethodError, twenty minutes into a queued job, with nothing in the message saying "wrong
# Springsteel". origin/main is 8 commits behind and even origin/feature/rlr-tiling is 5 behind,
# so "it resolved" is not evidence of anything.
#
# Project.toml's [sources] entry points at a sibling checkout to make the right thing happen;
# this check is what catches the case where that sibling checkout is on the wrong branch.
tc_preflight() {
  "$JULIA" --project="$SCYTHE" -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()
using Springsteel
println("  Springsteel : ", pkgdir(Springsteel))
miss = String[]
isdefined(Springsteel, :_hydrostatic_pressure_profile) || push!(miss, "_hydrostatic_pressure_profile")
isdefined(Springsteel.CubicBSpline, :R1T1X)            || push!(miss, "CubicBSpline.R1T1X")
any(m -> :hydrostatic in Base.kwarg_decl(m),
    methods(Springsteel.calculate_pressure_reference_state)) ||
    push!(miss, "calculate_pressure_reference_state(; hydrostatic)")
if !isempty(miss)
    error("""
        This Springsteel is too old for the TC configuration. Missing: $(join(miss, ", ")).
        Needed: the feature/rlr-tiling work -- a749f27 (reference state / hydrostatic sweep),
        f988eff (R3X ahat reload, the nesting fix), db46e4d (R1T1X), f074acc (positivity fit).
        Registered v1.0.0 is 59 commits behind; origin/main 8; origin/feature/rlr-tiling 5.
        Check the branch of the Springsteel.jl checked out beside Scythe.jl.""")
end
println("  Springsteel : required symbols present")
' || { echo "PREFLIGHT FAILED - not starting the run." >&2; exit 3; }
}
