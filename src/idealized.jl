# Idealized initialization utilities for benchmark cases and test setups.
#
# These build reference soundings and analytic perturbations (cold/warm
# bubbles) on a model patch, in the perturbation form the equation sets
# expect (deviations from a hydrostatic ReferenceState). All moisture
# variables use the linear mu_transform convention.

"""
    write_dry_sounding(path; sfc_p_hPa=1000.0, theta=300.0, q_v_gkg=0.0, zmax, dz=500.0)

Write a constant-theta sounding file in the format read by
[`calculate_reference_state`](@ref): a surface line `p_sfc theta q_v` followed
by `z theta q_v` levels. `zmax` must exceed the model domain top so every model
level can be interpolated. Water vapor is in g/kg.
"""
function write_dry_sounding(path::String; sfc_p_hPa=1000.0, theta=300.0,
                            q_v_gkg=0.0, zmax::Float64, dz=500.0)
    open(path, "w") do f
        println(f, "$(sfc_p_hPa)\t$(theta)\t$(q_v_gkg)")
        for z in dz:dz:zmax
            println(f, "$(z)\t$(theta)\t$(q_v_gkg)")
        end
    end
    return path
end

"""
    reference_profiles(ref::ReferenceState)

Return a NamedTuple of physical profiles `(q_v, rho_d, Tk, p, theta)` derived
from a reference state, for use in initial condition construction.
"""
function reference_profiles(ref::ReferenceState)
    thermo = thermodynamic_tuple.(ref.sbar[:, 1], ref.xibar[:, 1], ref.mubar[:, 1])
    q_v = [x[1] for x in thermo]
    rho_d = [x[2] for x in thermo]
    Tk = [x[3] for x in thermo]
    p = [x[4] for x in thermo]
    theta = potential_temperature.(ref.sbar[:, 1], ref.xibar[:, 1], ref.mubar[:, 1])
    return (; q_v, rho_d, Tk, p, theta)
end

"""
    temperature_bubble!(patch, gridpoints, ref; xc, xr, zc, zr, dT_max)

Add an elliptical cosine temperature perturbation at constant pressure:
`dT = dT_max * (cos(pi*L) + 1)/2` inside the unit ellipse
`L = sqrt(((x-xc)/xr)^2 + ((z-zc)/zr)^2) <= 1`. Density is recomputed from the
perturbed temperature at the unperturbed pressure, and the entropy (`s`, var 1)
and log-density (`xi`, var 2) perturbations are written in place. This is the
Straka et al. (1993) cold bubble for `dT_max = -15`.
"""
function temperature_bubble!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                             ref::ReferenceState;
                             xc=0.0, xr=4000.0, zc=3000.0, zr=2000.0, dT_max=-15.0)
    prof = reference_profiles(ref)
    kDim = patch.params.kDim
    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dT = L <= 1.0 ? dT_max * (cos(pi * L) + 1.0) / 2.0 : 0.0
            new_T = prof.Tk[k] + dT
            new_rho_d = prof.p[k] * 100.0 / (Rd * new_T)
            patch.physical[i, 1, 1] = entropy(new_T, new_rho_d, prof.q_v[k]) - ref.sbar[k, 1]
            patch.physical[i, 2, 1] = log_dry_density(new_rho_d) - ref.xibar[k, 1]
            i += 1
        end
    end
    return patch
end

"""
    theta_bubble!(patch, gridpoints, ref; xc, xr, zc, zr, dtheta_max)

Add a circular squared-cosine potential temperature perturbation at constant
pressure: `dtheta = dtheta_max * cos^2(pi*L/2)` inside the unit ellipse.
Temperature and density are recomputed from the perturbed theta at the
unperturbed pressure, and the entropy (`s`, var 1) and log-density (`xi`,
var 2) perturbations are written in place. This is the Bryan & Fritsch (2002)
warm bubble for `dtheta_max = 2`.
"""
function theta_bubble!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                       ref::ReferenceState;
                       xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0)
    prof = reference_profiles(ref)
    kDim = patch.params.kDim
    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dtheta = L <= 1.0 ? dtheta_max * (cos(pi * L / 2.0))^2 : 0.0
            theta = prof.theta[k] + dtheta
            Tk = theta / (p_0 / prof.p[k])^(Rd / Cpd)
            rho_d = prof.p[k] * 100.0 / (Rd * Tk)
            patch.physical[i, 1, 1] = entropy(Tk, rho_d, prof.q_v[k]) - ref.sbar[k, 1]
            patch.physical[i, 2, 1] = log_dry_density(rho_d) - ref.xibar[k, 1]
            i += 1
        end
    end
    return patch
end

"""
    saturated_surface_state(; q_t=0.02, theta_e=320.0, sfc_p_hPa=1000.0, T_guess=290.0, tol=1.0e-10)

Find the saturated surface state at pressure `sfc_p_hPa` with constant total
water `q_t` whose reversible equivalent potential temperature equals `theta_e`
(Newton iteration on temperature). This anchors the Bryan & Fritsch (2002)
moist neutral base state.

Returns a NamedTuple `(T, p, q_v, q_l, rho_d, s, xi, s_rev)` where `s_rev` is
the reversible entropy `s + q_l*Cl*log(T/T_0)` that is constant throughout the
moist neutral atmosphere.
"""
function saturated_surface_state(; q_t=0.02, theta_e=320.0, sfc_p_hPa=1000.0,
                                 T_guess=290.0, tol=1.0e-10)
    p = sfc_p_hPa

    function theta_e_residual(T)
        q_v = q_sat_liquid(T, p)
        e = vapor_pressure(p, q_v)
        rho_d = 100.0 * (p - e) / (Rd * T)
        s = entropy(T, rho_d, q_v)
        xi = log_dry_density(rho_d)
        te = reversible_theta_e(s, xi, mu_transform(q_v), mu_transform(q_t - q_v))
        return te - theta_e
    end

    T = T_guess
    incr = 1.0e-4
    for _ in 1:100
        f = theta_e_residual(T)
        abs(f) < tol && break
        dfdT = (theta_e_residual(T + incr) - theta_e_residual(T - incr)) / (2.0 * incr)
        T -= f / dfdT
    end

    q_v = q_sat_liquid(T, p)
    e = vapor_pressure(p, q_v)
    rho_d = 100.0 * (p - e) / (Rd * T)
    s = entropy(T, rho_d, q_v)
    xi = log_dry_density(rho_d)
    q_l = q_t - q_v
    s_rev = s + q_l * Cl * log(T / T_0)
    return (; T, p, q_v, q_l, rho_d, s, xi, s_rev)
end

"""
    write_moist_neutral_sounding(path; q_t=0.02, theta_e=320.0, sfc_p_hPa=1000.0,
                                 zmax, dz=100.0)

Write a first-guess sounding for the Bryan & Fritsch (2002) moist neutral
atmosphere by finite-difference upward integration of hydrostatic balance at
constant reversible entropy with exact saturation. The result seeds
`calculate_reference_state`; the spectral iteration in
[`saturated_hydrostatic_profile`](@ref) then refines it.
"""
function write_moist_neutral_sounding(path::String; q_t=0.02, theta_e=320.0,
                                      sfc_p_hPa=1000.0, zmax::Float64, dz=100.0)
    sfc = saturated_surface_state(; q_t, theta_e, sfc_p_hPa)
    s_rev_const = sfc.s_rev

    # Newton solve for T at pressure p such that the reversible entropy of the
    # exactly saturated state matches s_rev_const
    function solve_T(p, T_guess)
        function residual(T)
            q_v = q_sat_liquid(T, p)
            e = vapor_pressure(p, q_v)
            rho_d = 100.0 * (p - e) / (Rd * T)
            s = entropy(T, rho_d, q_v)
            return s + (q_t - q_v) * Cl * log(T / T_0) - s_rev_const
        end
        T = T_guess
        incr = 1.0e-4
        for _ in 1:50
            f = residual(T)
            abs(f) < 1.0e-10 && break
            dfdT = (residual(T + incr) - residual(T - incr)) / (2.0 * incr)
            T -= f / dfdT
        end
        return T
    end

    levels = Float64[]
    thetas = Float64[]
    qvs = Float64[]
    T = sfc.T
    p = sfc.p
    for z in 0.0:dz:zmax
        T = solve_T(p, T)
        q_v = q_sat_liquid(T, p)
        e = vapor_pressure(p, q_v)
        rho_d = 100.0 * (p - e) / (Rd * T)
        rho_t = rho_d * (1.0 + q_t)
        theta = T * (1000.0 / p)^(Rd / Cpd)
        push!(levels, z)
        push!(thetas, theta)
        push!(qvs, q_v * 1000.0)
        # Hydrostatic step to the next level
        p -= rho_t * gravity * dz / 100.0
    end

    open(path, "w") do f
        println(f, "$(sfc_p_hPa)\t$(thetas[1])\t$(qvs[1])")
        for i in 2:length(levels)
            println(f, "$(levels[i])\t$(thetas[i])\t$(qvs[i])")
        end
    end
    return path
end

"""
    saturated_hydrostatic_profile(z, column, ref; q_t=0.02, theta_e=320.0,
                                  sfc_p_hPa=1000.0, n_outer=5, n_inner=10)

Construct the Bryan & Fritsch (2002) saturated, exactly neutrally stable
(constant reversible entropy) hydrostatic base state on the model levels `z`.

`ref` is a hydrostatically balanced [`ReferenceState`](@ref) close to the
target profile (e.g., from `calculate_reference_state` on a first-guess
sounding); the construction iterates the perturbation hydrostatic equation
against it, so its accuracy only affects convergence rate. `column` is a
natural-BC vertical basis (see [`reference_column`](@ref)) used for the
perturbation derivative and integral transforms.

Each outer iteration solves the hydrostatic equation for the log-density
profile with entropy slaved to the constant reversible entropy, then
re-converges exact saturation level-by-level with `saturation_adjustment`.

Returns a NamedTuple of converged profiles
`(s, xi, mu, mu_l, q_v, q_l, Tk, p, rho_d, theta_rho, theta_e, residual)`
where `residual` is the hydrostatic imbalance [m/s²] at each level.
"""
function saturated_hydrostatic_profile(z::Vector{Float64}, column, ref::ReferenceState;
                                       q_t=0.02, theta_e=320.0, sfc_p_hPa=1000.0,
                                       n_outer=5, n_inner=10)
    sfc = saturated_surface_state(; q_t, theta_e, sfc_p_hPa)
    s_rev_const = sfc.s_rev
    sfc_xiprime = sfc.xi - ref.xibar[1, 1]

    nz = length(z)
    prof = reference_profiles(ref)
    rho_bar = prof.rho_d .* (1.0 .+ prof.q_v)

    # First guess: saturation at the reference temperature and pressure
    Tk = copy(prof.Tk)
    p = copy(prof.p)
    q_v = q_sat_liquid.(Tk, p)
    q_l = q_t .- q_v
    rho_d = copy(prof.rho_d)
    rho_t = rho_d .* (1.0 .+ q_t)
    rho_p = rho_t .- rho_bar
    s = s_rev_const .- (q_l .* Cl .* log.(Tk ./ T_0))
    xi = log_dry_density.(rho_d)
    mu = mu_transform.(q_v)
    mu_l = mu_transform.(q_l)

    # Rough temperature guess: integrate total density hydrostatically and
    # invert the entropy definition at constant reversible entropy
    for _ in 1:4
        column.uMish[:] .= -gravity .* rho_t
        Btransform!(column)
        Atransform!(column)
        p = IInttransform(column, sfc_p_hPa * 100.0) ./ 100.0
        e = vapor_pressure.(p, q_v)
        rho_d = 100.0 .* (p .- e) ./ (Tk .* Rd)
        rho_t = rho_d .* (1.0 .+ q_t)
        rho_p = rho_t .- rho_bar
        qfactor = q_v .* (Rv .* log.(q_v .* rho_d ./ rho_v0) .- (L_v(T_0) / T_0))
        Cfactor = Cvd .+ (q_v .* Cvv)
        Tk = T_0 .* exp.((s_rev_const .+ (Rd .* log.(rho_d ./ rho_d0)) .+ qfactor) ./
                         (Cfactor .+ (q_l .* Cl)))
    end

    # Anchor the surface values
    s[1] = sfc.s; q_v[1] = sfc.q_v; q_l[1] = sfc.q_l
    Tk[1] = sfc.T; p[1] = sfc.p; rho_d[1] = sfc.rho_d
    mu[1] = mu_transform(sfc.q_v); mu_l[1] = mu_transform(sfc.q_l)
    xi[1] = sfc.xi
    rho_t[1] = rho_d[1] * (1.0 + q_t)
    rho_p[1] = rho_t[1] - rho_bar[1]

    s_z = zeros(nz)
    xi_z = zeros(nz)
    qvp_z = zeros(nz)
    for _ in 1:n_outer
        # Solve the perturbation hydrostatic equation with entropy slaved to
        # the constant reversible entropy
        for _ in 1:n_inner
            Ps = P_s.(Tk, rho_d, q_v)
            Pxi = P_xi.(Tk, rho_d, q_v)
            Pqv = P_qv.(Tk, rho_d, q_v)

            mu = mu_transform.(q_v)
            mu_l = mu_transform.(q_l)
            column.uMish[:] .= mu .- ref.mubar[:, 1]
            Btransform!(column)
            Atransform!(column)
            mu_z = Ixtransform(column)
            qvp_z = mu_z ./ dmudq.(mu, q_v)

            s = s_rev_const .- (q_l .* Cl .* log.(Tk ./ T_0))
            column.uMish[:] .= s .- ref.sbar[:, 1]
            Btransform!(column)
            Atransform!(column)
            s_z = Ixtransform(column)

            xi_z = ((-gravity .* rho_p) .- (Ps .* s_z) .- (Pqv .* qvp_z)) ./ Pxi
            column.uMish[:] .= xi_z
            Btransform!(column)
            Atransform!(column)
            xi_prime = IInttransform(column, sfc_xiprime)
            xi = xi_prime .+ ref.xibar[:, 1]
            rho_d = dry_density.(xi)
            rho_t = rho_d .* (1.0 .+ q_t)
            rho_p = rho_t .- rho_bar
        end

        # Re-converge exact saturation level by level
        for k in 1:nz
            rho_d[k] = dry_density(xi[k])
            rho_t[k] = rho_d[k] * (1.0 + q_t)
            rho_p[k] = rho_t[k] - rho_bar[k]
            Tk[k] = temperature(s[k], rho_d[k], q_v[k])
            dq, dT = saturation_adjustment(s[k], xi[k], mu[k], mu_l[k], eps())
            s[k] += s_condensation_relaxation(-dq, Tk[k], rho_d[k], q_v[k], q_l[k], p[k])
            q_v[k] += dq
            q_l[k] = q_t - q_v[k]
            mu[k] = mu_transform(q_v[k])
            mu_l[k] = mu_transform(q_l[k])
            Tk[k] = temperature(s[k], rho_d[k], q_v[k])
            p[k] = pressure(s[k], rho_d[k], q_v[k])
        end
    end

    theta = potential_temperature.(s, xi, mu)
    theta_rho = theta .* (1.0 .+ (q_v ./ Eps)) ./ (1.0 .+ q_t)
    theta_e_prof = reversible_theta_e.(s, xi, mu, mu_l)
    residual = ((-pressure_gradient.(Tk, rho_d, q_v, s_z, xi_z, qvp_z)) .+
                (-gravity .* rho_p)) ./ rho_t

    return (; s, xi, mu, mu_l, q_v, q_l, Tk, p, rho_d, theta_rho,
            theta_e=theta_e_prof, residual)
end

"""
    interpolate_base_state(base, src_column::Chebyshev1D, z_target) -> NamedTuple

Spectrally interpolate every field of a base-state NamedTuple (from
[`saturated_hydrostatic_profile`](@ref)) off the source Chebyshev column's levels
onto `z_target`. The saturated-neutral base-state iteration converges on a
Chebyshev column but not on a low-DOF cubic B-spline column (the iteration is not
a true root finder and the saturation vapor pressure aloft is hypersensitive), so
the moist RiRk case builds the base state on Chebyshev and transfers it to the
spline model levels with this near-analytic spectral interpolation. The
interpolated profile is a stable initial condition; it is not re-balanced on the
target grid (a small hydrostatic residual from interpolation is acceptable for a
perturbation run).
"""
function interpolate_base_state(base::NamedTuple, src_column::Chebyshev1D,
                                z_target::Vector{Float64})
    Cmat = Chebyshev.CItransform_matrix(src_column, z_target, 0)
    interp = function (vals)
        src_column.uMish[:] .= vals
        Btransform!(src_column)
        Atransform!(src_column)
        return Cmat * src_column.a
    end
    return NamedTuple{keys(base)}(map(interp, values(base)))
end

"""
    write_exact_ref(path, z, s, xi, mu)

Write an exact reference state file (`z s xi mu` per line) in the format read
by [`exact_reference_state`](@ref). The `z` values are written with `string()`
so they match the model gridpoints exactly when generated in-process.
"""
function write_exact_ref(path::String, z::Vector{Float64}, s::Vector{Float64},
                         xi::Vector{Float64}, mu::Vector{Float64})
    open(path, "w") do f
        for i in 1:length(z)
            println(f, "$(z[i]) $(s[i]) $(xi[i]) $(mu[i])")
        end
    end
    return path
end

"""
    moist_buoyancy_bubble!(patch, gridpoints, base, ref; q_t=0.02,
                           xc, xr, zc, zr, amp=2.0/300.0, liquid_var="mu_l")

Add the Bryan & Fritsch (2002) moist warm bubble to a saturated base state
(from [`saturated_hydrostatic_profile`](@ref)). Inside the unit ellipse the
density potential temperature is increased by the factor
`1 + amp*cos^2(pi*L/2)` at constant pressure, vapor is reset to exact
saturation at the perturbed temperature, and the state is re-converged with
`saturation_adjustment`, mirroring eq. (36) of the paper.

Writes perturbations against `ref` for `s`, `xi`, and `mu`; the liquid water
variable (`liquid_var`) is written as a full field since the reference liquid
water is zero. All other variables are left untouched.
"""
function moist_buoyancy_bubble!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                                base, ref::ReferenceState; q_t=0.02,
                                xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0,
                                amp=2.0/300.0, liquid_var="mu_l", control::Symbol=:xi)
    vars = patch.params.vars
    s_i = vars["s"]; mu_i = vars["mu"]
    # Density control variable: log-density "xi" (default) or linear "rho_d"
    dens_i = control === :rhod ? vars["rho_d"] : vars["xi"]
    ql_i = vars[liquid_var]
    kDim = patch.params.kDim
    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            b_incr = L <= 1.0 ? amp * (cos(pi * L / 2.0))^2 : 0.0

            new_s = base.s[k]
            new_xi = base.xi[k]
            new_rho_d = base.rho_d[k]
            new_mu = base.mu[k]
            new_mu_l = base.mu_l[k]
            if b_incr > 0.0
                p = base.p[k]
                new_theta = base.theta_rho[k] * (1.0 + b_incr) * (1.0 + q_t) /
                            (1.0 + (base.q_v[k] / Eps))
                new_T = new_theta / (p_0 / p)^(Rd / Cpd)
                new_q_v = q_sat_liquid(new_T, p)
                new_q_l = q_t - new_q_v
                new_T_rho = new_T * (1.0 + new_q_v / Eps) / (1.0 + q_t)
                new_rho_t = (p * 100.0) / (new_T_rho * Rd)
                new_rho_d = new_rho_t / (1.0 + q_t)
                new_xi = log_dry_density(new_rho_d)
                new_mu = mu_transform(new_q_v)
                new_mu_l = mu_transform(new_q_l)
                new_s = entropy(new_T, new_rho_d, new_q_v)
                dq, _ = saturation_adjustment(new_s, new_xi, new_mu, new_mu_l, eps())
                new_s += s_condensation_relaxation(-dq, new_T, new_rho_d, new_q_v, new_q_l, p)
                new_q_v += dq
                new_q_l = q_t - new_q_v
                new_mu = mu_transform(new_q_v)
                new_mu_l = mu_transform(new_q_l)
            end

            patch.physical[i, s_i, 1] = new_s - ref.sbar[k, 1]
            patch.physical[i, dens_i, 1] = control === :rhod ?
                (new_rho_d - ref.rhobar[k, 1]) : (new_xi - ref.xibar[k, 1])
            patch.physical[i, mu_i, 1] = new_mu - ref.mubar[k, 1]
            patch.physical[i, ql_i, 1] = new_mu_l
            i += 1
        end
    end
    return patch
end

"""
    write_ics_csv(path, patch, gridpoints)

Write the value slice of `patch.physical` as an initial conditions CSV with
coordinate columns (`r`, `z` for 2-D grids) and one column per variable, in the
format read by `read_physical_grid`.
"""
function write_ics_csv(path::String, patch::AbstractGrid, gridpoints::Matrix{Float64})
    vars = sort(collect(patch.params.vars), by = last)
    ndims_grid = size(gridpoints, 2)
    coords = ndims_grid == 1 ? ["r"] : ndims_grid == 2 ? ["r", "z"] : ["r", "l", "z"]
    open(path, "w") do f
        println(f, join(vcat(coords, first.(vars)), ","))
        for i in 1:size(patch.physical, 1)
            coord_vals = [string(gridpoints[i, d]) for d in 1:ndims_grid]
            var_vals = [string(patch.physical[i, vi, 1]) for (_, vi) in vars]
            println(f, join(vcat(coord_vals, var_vals), ","))
        end
    end
    return path
end
