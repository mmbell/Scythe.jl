# Shared diagnostic helpers for benchmark verification.
#
# These reconstruct thermodynamic fields from a model output CSV plus the same
# reference state the model used, mirroring the analysis cells of the original
# benchmark notebooks.

using CSV
using DataFrames

"""
    rebuild_reference(model) -> (ref, z, kDim)

Rebuild the reference state exactly as `createModelTile` does, from the model
configuration. Used to reconstruct total fields from perturbation output.
"""
function rebuild_reference(model)
    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = deepcopy(patch.kbasis.data[1])
    if model.options[:exact_reference_state]
        ref = Scythe.exact_reference_state(model, z, column)
    else
        ref = Scythe.calculate_reference_state(model, z, column)
    end
    return ref, z, kDim
end

"""
    read_final_output(model) -> DataFrame

Read the physical output CSV at the final integration time.
"""
function read_final_output(model)
    tag = string(round(model.integration_time; digits=2))
    path = joinpath(model.output_dir, "$(tag)_physical.csv")
    return CSV.read(path, DataFrame)
end

"""
    theta_perturbation(df, ref, kDim)

Compute the potential temperature perturbation field from output entropy and
log-density perturbations plus the reference profile. Returns (theta_p, ncols)
where theta_p is a (kDim, ncols) matrix with z varying fastest, matching the
output ordering.
"""
function theta_perturbation(df::DataFrame, ref, kDim::Int)
    npts = nrow(df)
    ncols = div(npts, kDim)
    sbar = repeat(ref.sbar[:, 1], ncols)
    xibar = repeat(ref.xibar[:, 1], ncols)
    mubar = repeat(ref.mubar[:, 1], ncols)
    q_v = Scythe.inv_mu_transform.(df.mu .+ mubar)
    q_bar = Scythe.inv_mu_transform.(mubar)
    theta = Scythe.potential_temperature.(df.s .+ sbar, df.xi .+ xibar, q_v)
    theta0 = Scythe.potential_temperature.(sbar, xibar, q_bar)
    theta_p = reshape(theta .- theta0, kDim, ncols)
    return theta_p, ncols
end

"""
    front_location(r, row; threshold=-1.0)

Locate a density current front: the largest radius where `row` crosses
`threshold`, linearly interpolated. `r` and `row` are values along one height
level, ordered by increasing r. Returns NaN if the threshold is never reached.
"""
function front_location(r::AbstractVector, row::AbstractVector; threshold=-1.0)
    front = NaN
    for i in 1:(length(r) - 1)
        below = row[i] <= threshold
        above = row[i+1] > threshold
        if below && above
            frac = (threshold - row[i]) / (row[i+1] - row[i])
            front = r[i] + frac * (r[i+1] - r[i])
        end
    end
    # Threshold reached at the last point: front is at or beyond the boundary
    if isnan(front) && !isempty(row) && row[end] <= threshold
        front = r[end]
    end
    return front
end
