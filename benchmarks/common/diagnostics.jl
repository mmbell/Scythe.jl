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
    gauss_cell_weights(npts, ncells, length, mubar, quadrature) -> Vector

Physical Gauss-quadrature weights for a field sampled on `npts = ncells*mubar`
mish points (cell-by-cell Gauss nodes) spanning `length`. The dot product of
these weights with the mish values is the exact integral of the cubic-spline
representation — the same quadrature the model's Galerkin solver uses.
"""
function gauss_cell_weights(npts::Int, ncells::Int, length::Float64,
                            mubar::Int, quadrature::Symbol)
    @assert npts == ncells * mubar "mish count $npts ≠ ncells*mubar $(ncells*mubar)"
    DX = length / ncells
    _, qw = CubicBSpline._quadrature_rule(mubar, quadrature)
    return repeat(qw .* DX, outer = ncells)              # length npts
end

"""
    domain_integral(field, model) -> Float64

Integrate a `(kDim, ncols)` field over the 2-D domain: a vertical integral in z
for each column, then a horizontal integral across columns.

The mish points are Gauss quadrature nodes in **both** directions, so the
integral uses the **Gauss-weight quadrature on those nodes** directly — exact for
the model's spline representation and consistent with its Galerkin solver.
Refitting the data to a non-interpolating spline (`b_kDim = num_cells+3 < kDim/
ncols` coefficients) and integrating its antiderivative, as before, introduced a
shape-dependent ~0.5% error that oscillated with the field and masqueraded as a
mass/energy conservation drift; the Gauss-weight integral removes it (apparent
RiRk drift ~30× smaller). The RZ (Chebyshev) vertical integral is spectrally
exact and unchanged; both grids share the spline horizontal direction, so the
horizontal Gauss-weight integral tightens RZ as well.
"""
function domain_integral(field::AbstractMatrix, model)
    gp = model.grid_params
    spline_vertical = String(gp.geometry) == "RiRk"
    ncols = size(field, 2)
    colints = zeros(ncols)

    # Vertical integral per column
    if spline_vertical
        Wv = gauss_cell_weights(gp.kDim, gp.kDim ÷ gp.mubar, gp.kMax - gp.kMin,
                                gp.mubar, gp.quadrature)
        for c in 1:ncols
            colints[c] = sum(Wv .* @view(field[:, c]))
        end
    else
        zcol = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin, zmax = gp.kMax,
            zDim = gp.kDim, bDim = gp.b_kDim,
            BCB = Chebyshev.R0, BCT = Chebyshev.R0))
        for c in 1:ncols
            zcol.uMish .= field[:, c]
            Btransform!(zcol)
            Atransform!(zcol)
            colints[c] = IInttransform(zcol, 0.0)[end]
        end
    end

    # Horizontal integral across columns (spline-i for both RZ and RiRk): Gauss
    # weights on the i mish points (mubar_i inferred from the output column count).
    Wh = gauss_cell_weights(ncols, gp.num_cells, gp.iMax - gp.iMin,
                            ncols ÷ gp.num_cells, gp.quadrature)
    return sum(Wh .* colints)
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
