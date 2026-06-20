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
    column = Scythe.reference_column(patch, model.grid_params)
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
    theta = Scythe.potential_temperature.(df.s .+ sbar, df.xi .+ xibar, df.mu .+ mubar)
    theta0 = Scythe.potential_temperature.(sbar, xibar, mubar)
    theta_p = reshape(theta .- theta0, kDim, ncols)
    return theta_p, ncols
end

"""
    domain_integral(field, model) -> Float64

Spectrally integrate a `(kDim, ncols)` field over the 2-D domain: a vertical
integral in z for each column (Chebyshev for the RZ grid, cubic B-spline for the
RiRk grid — matching the model's vertical basis), then a cubic B-spline integral
across columns with the antiderivative evaluated at the domain edge.
"""
function domain_integral(field::AbstractMatrix, model)
    gp = model.grid_params
    spline_vertical = String(gp.geometry) == "RiRk"
    if spline_vertical
        zcol = Spline1D(SplineParameters(
            xmin = gp.kMin, xmax = gp.kMax,
            num_cells = gp.kDim ÷ gp.mubar, mubar = gp.mubar,
            quadrature = gp.quadrature,
            BCL = CubicBSpline.R0, BCR = CubicBSpline.R0))
    else
        zcol = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin, zmax = gp.kMax,
            zDim = gp.kDim, bDim = gp.b_kDim,
            BCB = Chebyshev.R0, BCT = Chebyshev.R0))
    end
    ncols = size(field, 2)
    colints = zeros(ncols)
    out = zeros(1)
    for c in 1:ncols
        if spline_vertical
            # Definite integral 0→kMax: spline antiderivative evaluated at the top.
            CubicBSpline.SIIntcoefficients!(zcol, field[:, c])
            CubicBSpline.SItransform(zcol.params, zcol.a, [gp.kMax], out, 0)
            colints[c] = out[1]
        else
            zcol.uMish .= field[:, c]
            Btransform!(zcol)
            Atransform!(zcol)
            colints[c] = IInttransform(zcol, 0.0)[end]
        end
    end
    sp = SplineParameters(xmin = gp.iMin, xmax = gp.iMax, num_cells = gp.num_cells,
                          BCL = CubicBSpline.R0, BCR = CubicBSpline.R0)
    spline = Spline1D(sp)
    CubicBSpline.SIIntcoefficients!(spline, colints)
    CubicBSpline.SItransform(spline.params, spline.a, [gp.iMax], out, 0)
    return out[1]
end

"""
    conservation_drift(model, ref; liquid_var=nothing) -> Dict

Percent drift of the domain-integrated total mass, total energy, and total
entropy between the initial and final output times (cf. Bryan & Fritsch 2002,
eqs. 28-29, who report ~1e-4 % drift for their benchmark).

The prognostic entropy is dry air + vapor only, so condensation acts as a
source/sink on it; the total entropy integrated here adds the condensate
entropy `q_l*Cl*log(T/T_0)` and should be conserved even in the moist case.
Set `liquid_vars` to the liquid water variable names (e.g. `["mu_l"]` for the
legacy BF02 set or `["mu_c", "mu_r"]` for the primitive equations); an empty
list treats the run as dry. The linear mu transform makes the sum of
transformed variables equal the transform of the summed mixing ratios.
"""
function conservation_drift(model, ref; liquid_vars::Vector{String}=String[])
    kDim = model.grid_params.kDim

    function integrals(tag)
        df = CSV.read(joinpath(model.output_dir, "$(tag)_physical.csv"), DataFrame)
        ncols = div(nrow(df), kDim)
        sbar = repeat(ref.sbar[:, 1], ncols)
        xibar = repeat(ref.xibar[:, 1], ncols)
        mubar = repeat(ref.mubar[:, 1], ncols)
        s = df.s .+ sbar
        xi = df.xi .+ xibar
        mu = df.mu .+ mubar
        thermo = Scythe.thermodynamic_tuple.(s, xi, mu)
        q_v = [x[1] for x in thermo]
        rho_d = [x[2] for x in thermo]
        Tk = [x[3] for x in thermo]
        q_l = zero(q_v)
        for lv in liquid_vars
            q_l = q_l .+ Scythe.inv_mu_transform.(df[!, lv])
        end
        q_t = q_v .+ q_l
        ke = 0.5 .* (df.u .^ 2 .+ df.w .^ 2)

        mass = rho_d .* (1.0 .+ q_t)
        energy = rho_d .* ((Scythe.Cvd .* Tk) .+ (q_v .* Scythe.Cvv .* Tk) .+
                           (q_l .* Scythe.Cl .* Tk) .- (Scythe.L_v.(Tk) .* q_l) .+
                           ((1.0 .+ q_t) .* ke) .+
                           ((1.0 .+ q_t) .* Scythe.gravity .* df.z))
        total_entropy = rho_d .* (s .+ (q_l .* Scythe.Cl .* log.(Tk ./ Scythe.T_0)))

        return (mass = domain_integral(reshape(mass, kDim, ncols), model),
                energy = domain_integral(reshape(energy, kDim, ncols), model),
                entropy = domain_integral(reshape(total_entropy, kDim, ncols), model))
    end

    init = integrals("0.0")
    final = integrals(string(round(model.integration_time; digits=2)))
    return Dict(
        "mass_drift_pct" => 100.0 * (final.mass - init.mass) / abs(init.mass),
        "energy_drift_pct" => 100.0 * (final.energy - init.energy) / abs(init.energy),
        "entropy_drift_pct" => 100.0 * (final.entropy - init.entropy) / abs(init.entropy),
    )
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
