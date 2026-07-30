# Shared diagnostic helpers for benchmark verification.
#
# These reconstruct thermodynamic fields from a model output CSV plus the same
# reference state the model used, mirroring the analysis cells of the original
# benchmark notebooks.

using CSV
using DataFrames

# Internal-energy latent reference constant (Bryan & Fritsch 2002). The total-energy budget
# of this compressible EOS uses each species' own sensible heat (Cvd/Cvv/Cl) plus a CONSTANT
# latent offset on the vapor density: the temperature dependence of L_v(T) is carried by the
# Cvv vs Cl heat-capacity difference, so the conserved internal energy is
#   ρ_d·Cvd·T + ρ_v·Cvv·T + ρ_c·Cl·T + ρ_v·L_const ,   L_const = L_v0 − (Cpv − Cl)·T_0 .
# Using the enthalpy form −L_v(T)·q_l instead double-counts that dependence and drifts
# spuriously (positive, condensation-driven) as condensate forms — see the single-column
# closure test in test/test_partial_density.jl.
const L_const = Scythe.L_v0 - (Scythe.Cpv - Scythe.Cl) * Scythe.T_0

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
    if Scythe.uses_pressure_reference(model.equation_set)
        # Total-energy stage: pressure-based reference (p, densities, E_t, Q_ss)
        ref = Springsteel.exact_pressure_reference_state(model.ref_state_file, z, column)
    elseif Scythe.uses_physical_reference(model.equation_set)
        # Partial-density / entropy-density stage: physical condensate-bearing reference
        ref = Springsteel.exact_reference_state(model.ref_state_file, z, column)
    elseif model.options[:exact_reference_state]
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
    mc_state(df, ref, kDim, ncols)

Reconstruct the diagnostic thermodynamic state of the total-energy set
(moist_compressible) from a perturbation output DataFrame: the closed-form
temperature retrieval from the PROGNOSTIC condensate, then the residual vapor.
Returns `(Tk, p, rho_d, rho_v, rho_c, rho_t)` as flat vectors (z fastest), with p
in Pa. `transform` must be the run's `options[:condensate_transform]`: slot 9 then holds a
control variable rather than a density, and reading it raw would be silently wrong by a
factor of two in the linear regime.
"""
function mc_state(df, ref, kDim, ncols; transform::Symbol = :none, mu = 1.0e-7)
    pbar = Springsteel.ref_pressure(ref)[:, 1]
    rho_dbar = Springsteel.ref_rho_d(ref)[:, 1]
    rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
    rho_cbar = Springsteel.ref_rho_c(ref)[:, 1]
    E_tbar = Springsteel.ref_total_energy(ref)[:, 1]
    p = df.p .+ repeat(pbar, ncols)
    rho_d = df.rho_d .+ repeat(rho_dbar, ncols)
    rho_t = df.rho_t .+ repeat(rho_tbar, ncols)
    E_t = df.E_t .+ repeat(E_tbar, ncols)
    # Slot 9 is not necessarily a density: under `options[:condensate_transform]` the output
    # column holds the CONTROL variable and the density has to be recovered, exactly as the
    # kernel does. `:none` is the plain add, bit for bit.
    rho_c = Scythe.recover_rho_c.(df.rho_c, repeat(rho_cbar, ncols), transform, mu)
    ke = 0.5 .* (df.u .^ 2 .+ df.w .^ 2)
    M = p .+ E_t .- (rho_t .* (ke .+ Scythe.gravity .* df.z))
    rho_liq = rho_c .+ df.rho_r
    Tk = Scythe.retrieve_temperature.(M, rho_d, rho_t, rho_liq)
    # Vapor is the residual of the prognostic water masses
    rho_v = rho_t .- rho_d .- rho_liq
    return Tk, p, rho_d, rho_v, rho_c, rho_t
end

"""
    theta_perturbation(df, ref, kDim)

Compute the potential temperature perturbation field from output entropy and
log-density perturbations plus the reference profile. Returns (theta_p, ncols)
where theta_p is a (kDim, ncols) matrix with z varying fastest, matching the
output ordering.
"""
function theta_perturbation(df::DataFrame, ref, kDim::Int;
                            transform::Symbol = :none, mu = 1.0e-7)
    npts = nrow(df)
    ncols = div(npts, kDim)

    # Total-energy set (moist_compressible): retrieve T from the prognostic
    # (p, E_t, Q_ss, densities), then θ = T·(p_0/p)^(Rd/Cpd) directly.
    if "E_t" in names(df)
        Tk, p, _, _, _, _ = mc_state(df, ref, kDim, ncols;
                                     transform = transform, mu = mu)
        theta = Tk .* ((Scythe.p_0 .* 100.0) ./ p) .^ (Scythe.Rd / Scythe.Cpd)
        Tbar = Springsteel.reference_temperature(ref)
        pbar = Springsteel.ref_pressure(ref)[:, 1]
        theta0 = Tbar .* ((Scythe.p_0 .* 100.0) ./ pbar) .^ (Scythe.Rd / Scythe.Cpd)
        return reshape(theta .- repeat(theta0, ncols), kDim, ncols), ncols
    end

    # Physical-density sets (pd / sigma) store rho_v and use a Springsteel
    # reference; reconstruct θ = T·(p_0/p)^(Rd/Cpd) from the densities and (s or σ).
    if "rho_v" in names(df)
        rho_dbar = Springsteel.ref_rho_d(ref)[:, 1]
        rho_d = df.rho_d .+ repeat(rho_dbar, ncols)
        if "sigma" in names(df)
            sigmabar = rho_dbar .* Springsteel.ref_entropy(ref)[:, 1]
            s = (df.sigma .+ repeat(sigmabar, ncols)) ./ rho_d
        else
            s = df.s .+ repeat(Springsteel.ref_entropy(ref)[:, 1], ncols)
        end
        rvbar = Springsteel.ref_rho_v(ref)
        rho_vbar = rvbar === 0.0 ? zeros(kDim) : rvbar[:, 1]
        q_v = (df.rho_v .+ repeat(rho_vbar, ncols)) ./ rho_d
        θ(s_, rd_, qv_) = Scythe.temperature(s_, rd_, qv_) *
            (Scythe.p_0 / Scythe.pressure(s_, rd_, qv_))^(Scythe.Rd / Scythe.Cpd)
        theta = θ.(s, rho_d, q_v)
        theta0 = θ.(repeat(Springsteel.ref_entropy(ref)[:, 1], ncols),
                    repeat(rho_dbar, ncols), repeat(rho_vbar ./ rho_dbar, ncols))
        return reshape(theta .- theta0, kDim, ncols), ncols
    end

    sbar = repeat(ref.sbar[:, 1], ncols)
    xibar = repeat(ref.xibar[:, 1], ncols)
    mubar = repeat(ref.mubar[:, 1], ncols)
    # The PE rho_d stage carries linear rho_d'; reconstruct xi = ln(rho_d/rho_0)
    xi = "rho_d" in names(df) ?
        Scythe.log_dry_density.(df.rho_d .+ repeat(ref.rhobar[:, 1], ncols)) :
        df.xi .+ xibar
    theta = Scythe.potential_temperature.(df.s .+ sbar, xi, df.mu .+ mubar)
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
        Wv = gauss_cell_weights(gp.kDim, gp.num_cells_k, gp.kMax - gp.kMin,
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
    domain_integral(field, model, col_mask) -> Float64

Column-masked variant for nested runs: only columns with `col_mask[c] == true`
contribute to the horizontal integral. Used to restrict a nest patch to its
NOMINAL region (its collar cells duplicate the child's territory and must be
excluded so abutting patches partition the domain exactly; collars occupy
whole cells, so a column mask loses no accuracy).
"""
function domain_integral(field::AbstractMatrix, model, col_mask::AbstractVector{Bool})
    gp = model.grid_params
    ncols = size(field, 2)
    length(col_mask) == ncols || error("col_mask length $(length(col_mask)) ≠ ncols $ncols")
    masked = field .* reshape(Float64.(col_mask), 1, ncols)
    return domain_integral(masked, model)
end

"""
    nominal_col_mask(model, xlo, xhi) -> Vector{Bool}

Column mask selecting the mish columns of `model`'s grid whose x lies in the
nominal region `[xlo, xhi]` (excluding this patch's collar cells).
"""
function nominal_col_mask(model, xlo::Float64, xhi::Float64)
    patch = createGrid(model.grid_params)
    pts = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    x = kDim > 0 && String(model.grid_params.geometry) != "R" ?
        pts[1:kDim:end, 1] : vec(pts)
    return [xlo - 1e-9 <= xi <= xhi + 1e-9 for xi in x]
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

    # Partial-density runs store moisture as the densities rho_v/rho_c/rho_r and use
    # a physical (CondensateReferenceState) reference accessed through the generic
    # Springsteel accessors rather than the legacy sbar/mubar/rhobar fields. The
    # sigma (entropy-density) set additionally stores slot 1 as sigma = rho_d*s.
    pd = Scythe.uses_physical_reference(model.equation_set)
    # The total-energy set (moist_compressible) integrates its extensive prognostics
    # directly: mass from rho_d/rho_t and energy from E_t itself, which IS the
    # corrected total-energy diagnostic E_t = ρ_d e_i + ρ_t(ke + gz) of
    # reference/Scythe_moist_compressible.tex.
    mc = Scythe.uses_pressure_reference(model.equation_set)

    function integrals(tag)
        df = CSV.read(joinpath(model.output_dir, "$(tag)_physical.csv"), DataFrame)
        ncols = div(nrow(df), kDim)

        if mc
            Tk, p, rho_d, rho_v, rho_c, rho_t = mc_state(df, ref, kDim, ncols;
                transform = Scythe.condensate_transform_mode(model.options),
                mu = get(model.physical_params, :condensate_mu, 1.0e-7))
            E_t = df.E_t .+ repeat(Springsteel.ref_total_energy(ref)[:, 1], ncols)
            # Clamp for the entropy diagnostic: in dry air the residual vapor sits at
            # 0 ± roundoff, and entropy() takes log(q_v).
            q_v = max.(rho_v, 0.0) ./ rho_d
            q_l = (max.(rho_c, 0.0) .+ df.rho_r) ./ rho_d
            water_mass = rho_t .- rho_d
            # Total entropy (informational; NOT conserved under finite-τ condensation —
            # the second-law production ∫ρ_d R_v ln(H) q̇_cond ≥ 0 makes it rise while
            # E_t stays flat; see the entropy budget subsection of the TeX).
            s = Scythe.entropy.(Tk, rho_d, q_v)
            total_entropy = rho_d .* (s .+ (q_l .* Scythe.Cl .* log.(Tk ./ Scythe.T_0)))
            return (dry_mass = domain_integral(reshape(rho_d, kDim, ncols), model),
                    water_mass = domain_integral(reshape(water_mass, kDim, ncols), model),
                    mass = domain_integral(reshape(rho_t, kDim, ncols), model),
                    energy = domain_integral(reshape(E_t, kDim, ncols), model),
                    entropy = domain_integral(reshape(total_entropy, kDim, ncols), model))
        end

        if pd
            rho_d = df.rho_d .+ repeat(Springsteel.ref_rho_d(ref)[:, 1], ncols)
            # Recover specific entropy s: directly (s set) or from the entropy density
            # sigma = rho_d*s carried by the moist_compressible set (s = (sigma'+sigmabar)/rho_d).
            if "sigma" in names(df)
                sigmabar = Springsteel.ref_rho_d(ref)[:, 1] .* Springsteel.ref_entropy(ref)[:, 1]
                s = (df.sigma .+ repeat(sigmabar, ncols)) ./ rho_d
            else
                s = df.s .+ repeat(Springsteel.ref_entropy(ref)[:, 1], ncols)
            end
            rho_v = df.rho_v .+ repeat(Springsteel.ref_rho_v(ref)[:, 1], ncols)
            rcbar = Springsteel.ref_rho_c(ref)
            rho_cbar = rcbar === 0.0 ? zeros(kDim) : rcbar[:, 1]
            rho_c = df.rho_c .+ repeat(rho_cbar, ncols)
            rho_r = df.rho_r                         # rho_rbar = 0
            q_v = rho_v ./ rho_d
            q_l = (rho_c .+ rho_r) ./ rho_d
            Tk = Scythe.temperature.(s, rho_d, q_v)
            q_t = q_v .+ q_l
            ke = 0.5 .* (df.u .^ 2 .+ df.w .^ 2)
            # Water mass is the integral of the partial densities directly — the
            # quantity the pd formulation conserves exactly under spline smoothing.
            water_mass = rho_v .+ rho_c .+ rho_r
            dry_mass = rho_d
            mass = rho_d .+ water_mass
            energy = rho_d .* ((Scythe.Cvd .* Tk) .+ (q_v .* Scythe.Cvv .* Tk) .+
                               (q_l .* Scythe.Cl .* Tk) .+ (q_v .* L_const) .+
                               ((1.0 .+ q_t) .* ke) .+
                               ((1.0 .+ q_t) .* Scythe.gravity .* df.z))
            total_entropy = rho_d .* (s .+ (q_l .* Scythe.Cl .* log.(Tk ./ Scythe.T_0)))
            return (dry_mass = domain_integral(reshape(dry_mass, kDim, ncols), model),
                    water_mass = domain_integral(reshape(water_mass, kDim, ncols), model),
                    mass = domain_integral(reshape(mass, kDim, ncols), model),
                    energy = domain_integral(reshape(energy, kDim, ncols), model),
                    entropy = domain_integral(reshape(total_entropy, kDim, ncols), model))
        end

        sbar = repeat(ref.sbar[:, 1], ncols)
        mubar = repeat(ref.mubar[:, 1], ncols)
        s = df.s .+ sbar
        mu = df.mu .+ mubar
        # Recover total dry density from whichever control variable the run stored:
        # the linear "rho_d" (rho_d') or the log-density "xi".
        if "rho_d" in names(df)
            rho_d = df.rho_d .+ repeat(ref.rhobar[:, 1], ncols)
        else
            rho_d = Scythe.dry_density.(df.xi .+ repeat(ref.xibar[:, 1], ncols))
        end
        thermo = Scythe.thermodynamic_tuple_rhod.(s, rho_d, mu)
        q_v = [x[1] for x in thermo]
        Tk = [x[3] for x in thermo]
        q_l = zero(q_v)
        for lv in liquid_vars
            q_l = q_l .+ Scythe.inv_mu_transform.(df[!, lv])
        end
        q_t = q_v .+ q_l
        ke = 0.5 .* (df.u .^ 2 .+ df.w .^ 2)
        dry_mass = rho_d
        water_mass = rho_d .* q_t
        mass = rho_d .* (1.0 .+ q_t)
        energy = rho_d .* ((Scythe.Cvd .* Tk) .+ (q_v .* Scythe.Cvv .* Tk) .+
                           (q_l .* Scythe.Cl .* Tk) .- (Scythe.L_v.(Tk) .* q_l) .+
                           ((1.0 .+ q_t) .* ke) .+
                           ((1.0 .+ q_t) .* Scythe.gravity .* df.z))
        total_entropy = rho_d .* (s .+ (q_l .* Scythe.Cl .* log.(Tk ./ Scythe.T_0)))

        return (dry_mass = domain_integral(reshape(dry_mass, kDim, ncols), model),
                water_mass = domain_integral(reshape(water_mass, kDim, ncols), model),
                mass = domain_integral(reshape(mass, kDim, ncols), model),
                energy = domain_integral(reshape(energy, kDim, ncols), model),
                entropy = domain_integral(reshape(total_entropy, kDim, ncols), model))
    end

    init = integrals("0.0")
    final = integrals(string(round(model.integration_time; digits=2)))
    pct_change = Dict(
        "dry_mass_drift_pct" => 100.0 * (final.dry_mass - init.dry_mass) / abs(init.dry_mass),
        "mass_drift_pct" => 100.0 * (final.mass - init.mass) / abs(init.mass),
        "energy_drift_pct" => 100.0 * (final.energy - init.energy) / abs(init.energy),
        "entropy_drift_pct" => 100.0 * (final.entropy - init.entropy) / abs(init.entropy),
    )
    # Check if any water mass exists before dividing by zero
    if abs(init.water_mass) > 0.0
        pct_change["water_mass_drift_pct"] = 100.0 * (final.water_mass - init.water_mass) / abs(init.water_mass)
    else
        pct_change["water_mass_drift_pct"] = 0.0
    end
    if mc
        sprod, n_neg_v = mc_entropy_production(model, ref)
        pct_change["entropy_prod_rate"] = sprod
        # The companion the rate cannot be read without: `entropy_prod_rate` is now summed
        # only over rho_v > 0, so a run with many negative-vapor points reports a rate for a
        # shrinking fraction of the domain. See mc_entropy_production.
        pct_change["neg_rho_v_points"] = float(n_neg_v)
    end
    return pct_change
end

"""
    mc_entropy_production(model, ref) -> (rate, n_neg_rho_v)

Second-law diagnostic for the total-energy set: the domain-integrated
instantaneous entropy production rate ∫ρ_d R_v ln(H) q̇_cond dV [J/(K s)] at the
final output time, summed ONLY over the points where the vapor is positive and the
quantity therefore exists. Positive-definite for irreversible phase change
(supersaturated condensation or subsaturated evaporation); zero for a
saturation-adjustment scheme. The domain total entropy should rise at roughly this
rate while the total energy stays constant.

Also returns the number of points EXCLUDED (rho_v <= 0). Read the two together: a
falling rate that comes with a rising exclusion count is not an improvement in
irreversibility, it is a shrinking domain.
"""
function mc_entropy_production(model, ref)
    kDim = model.grid_params.kDim
    tag = string(round(model.integration_time; digits=2))
    df = CSV.read(joinpath(model.output_dir, "$(tag)_physical.csv"), DataFrame)
    ncols = div(nrow(df), kDim)
    Tk, p, rho_d, rho_v, rho_c, rho_t = mc_state(df, ref, kDim, ncols;
        transform = Scythe.condensate_transform_mode(model.options),
        mu = get(model.physical_params, :condensate_mu, 1.0e-7))
    rho_vs = Springsteel.Thermodynamics.rho_v_sat.(Tk, p ./ 100.0)
    q_v = rho_v ./ rho_d
    q_l = (max.(rho_c, 0.0) .+ df.rho_r) ./ rho_d
    Q_ssbar = Springsteel.ref_qss(ref)[:, 1]
    Q_ss = df.Q_ss .+ repeat(Q_ssbar, ncols)
    H = max.(rho_v, 1.0e-12) ./ rho_vs
    Q_s = Scythe.Q_s_energy.(Tk, p, rho_d, q_v, q_l)
    Qdot = Scythe.qss_condensation_rate.(Q_ss, rho_v, max.(rho_c, 0.0), rho_d,
                                         Tk, p ./ 100.0, Q_s, model.ts)
    sprod = Scythe.Rv .* log.(H) .* Qdot
    # RESTRICT TO THE POINTS WHERE THE QUANTITY EXISTS. `ln(H)` is undefined for rho_v <= 0,
    # and the `max(rho_v, 1e-12)` above turns each such point into ln(1e-12/rho_vs) ~ -30 --
    # a large, entirely artificial contribution. Measured 2026-07-30 on the o01 quick runs:
    # the clamped points supply 0 % of the total on the healthy configurations but 31 % under
    # POSITIVITY=ck, 63 % under POSITIVITY=1, and 94-95 % on the NOPRECIP storm. Reporting
    # the unrestricted sum made a run's entropy production track its NEGATIVE-VAPOR COUNT
    # rather than its irreversibility, and two conclusions were drawn from it before the
    # artifact was found (the corrected factors are in
    # reference/FINDINGS_CONDENSATE_STAGE1.md §4).
    #
    # The excluded points are not a physical loss: no phase change there has a defined
    # entropy production. Their COUNT is the diagnostic that matters, and it is reported
    # separately as `neg_rho_v_points`.
    ok = rho_v .> 0.0
    sprod[.!ok] .= 0.0
    return domain_integral(reshape(sprod, kDim, ncols), model), count(.!ok)
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
