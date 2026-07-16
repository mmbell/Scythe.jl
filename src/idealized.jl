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
    thermo = thermodynamic_tuple.(ref_entropy(ref)[:, 1], ref_xi(ref)[:, 1], ref_mu(ref)[:, 1])
    q_v = [x[1] for x in thermo]
    rho_d = [x[2] for x in thermo]
    Tk = [x[3] for x in thermo]
    p = [x[4] for x in thermo]
    theta = potential_temperature.(ref_entropy(ref)[:, 1], ref_xi(ref)[:, 1], ref_mu(ref)[:, 1])
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
                             xc=0.0, xr=4000.0, zc=3000.0, zr=2000.0, dT_max=-15.0,
                             control::Symbol=:xi)
    prof = reference_profiles(ref)
    kDim = patch.params.kDim
    # Density control variable: log-density "xi" (default) or linear "rho_d" (slot 2)
    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dT = L <= 1.0 ? dT_max * (cos(pi * L) + 1.0) / 2.0 : 0.0
            new_T = prof.Tk[k] + dT
            new_rho_d = prof.p[k] * 100.0 / (Rd * new_T)
            patch.physical[i, 1, 1] = entropy(new_T, new_rho_d, prof.q_v[k]) - ref_entropy(ref)[k, 1]
            patch.physical[i, 2, 1] = control === :rhod ?
                (new_rho_d - ref_rho_d(ref)[k, 1]) : (log_dry_density(new_rho_d) - ref_xi(ref)[k, 1])
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
                       xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0,
                       control::Symbol=:xi)
    prof = reference_profiles(ref)
    kDim = patch.params.kDim
    # Density control variable: log-density "xi" (default) or linear "rho_d" (slot 2)
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
            patch.physical[i, 1, 1] = entropy(Tk, rho_d, prof.q_v[k]) - ref_entropy(ref)[k, 1]
            patch.physical[i, 2, 1] = control === :rhod ?
                (rho_d - ref_rho_d(ref)[k, 1]) : (log_dry_density(rho_d) - ref_xi(ref)[k, 1])
            i += 1
        end
    end
    return patch
end

"""
    theta_bubble_pd!(patch, gridpoints, ref; xc, xr, zc, zr, dtheta_max)

Dry warm-bubble initial condition for the physical-density equation sets
(`primitive_equation_XZ_rhod_pd`, `primitive_equation_XZ_sigma`) on a `Springsteel` reference state.
The constant-pressure θ perturbation mirrors [`theta_bubble!`](@ref) but writes the
physical-density slots: slot 1 is the intensive entropy `s'` (pd) or the entropy density
`σ' = ρ_d·s − σ̂` (when a `"sigma"` slot is present), slot 2 is `ρ_d'`, and the moisture slots
(`rho_v`, `rho_c`, `rho_r`) are zero (a dry run of the moist set, `q_v = 0`).
"""
function theta_bubble_pd!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                          ref::Springsteel.AbstractReferenceState;
                          xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0)
    vars = patch.params.vars
    sigma_mode = haskey(vars, "sigma")
    s1_i = sigma_mode ? vars["sigma"] : vars["s"]
    rho_d_i = vars["rho_d"]; rho_v_i = vars["rho_v"]
    rho_c_i = vars["rho_c"]; rho_r_i = vars["rho_r"]
    kDim = patch.params.kDim
    sbar = ref_entropy(ref); rho_dbar = ref_rho_d(ref)

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dtheta = L <= 1.0 ? dtheta_max * (cos(pi * L / 2.0))^2 : 0.0
            s_ref = sbar[k, 1]; rho_dref = rho_dbar[k, 1]
            T_ref = temperature(s_ref, rho_dref, 0.0)
            p_ref = pressure(s_ref, rho_dref, 0.0)          # hPa
            exner = (p_0 / p_ref)^(Rd / Cpd)
            theta = (T_ref * exner) + dtheta
            Tk = theta / exner
            rho_d = p_ref * 100.0 / (Rd * Tk)               # constant-pressure perturbation
            s = entropy(Tk, rho_d, 0.0)
            patch.physical[i, s1_i, 1] = sigma_mode ? (rho_d * s) - (rho_dref * s_ref) :
                                                      s - s_ref
            patch.physical[i, rho_d_i, 1] = rho_d - rho_dref
            patch.physical[i, rho_v_i, 1] = 0.0
            patch.physical[i, rho_c_i, 1] = 0.0
            patch.physical[i, rho_r_i, 1] = 0.0
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
    sfc_xiprime = sfc.xi - ref_xi(ref)[1, 1]

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
            column.uMish[:] .= mu .- ref_mu(ref)[:, 1]
            Btransform!(column)
            Atransform!(column)
            mu_z = Ixtransform(column)
            qvp_z = mu_z ./ dmudq.(mu, q_v)

            s = s_rev_const .- (q_l .* Cl .* log.(Tk ./ T_0))
            column.uMish[:] .= s .- ref_entropy(ref)[:, 1]
            Btransform!(column)
            Atransform!(column)
            s_z = Ixtransform(column)

            xi_z = ((-gravity .* rho_p) .- (Ps .* s_z) .- (Pqv .* qvp_z)) ./ Pxi
            column.uMish[:] .= xi_z
            Btransform!(column)
            Atransform!(column)
            xi_prime = IInttransform(column, sfc_xiprime)
            xi = xi_prime .+ ref_xi(ref)[:, 1]
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
    write_exact_ref_pd(path, z, s, rho_d, rho_v, rho_c)

Write a physical-density exact reference state file (`z s rho_d rho_v rho_c` per line)
in the format read by `Springsteel.exact_reference_state`, which returns a
`CondensateReferenceState`. The partial densities carry the condensate in the base so a
saturated cloudy profile (Bryan & Fritsch 2002) is neutrally buoyant. `z` values are
written with `string()` so they match the model gridpoints exactly when generated
in-process. Used by the partial-density (`*_pd`) equation sets in place of the
transformed `z s xi mu` format of [`write_exact_ref`](@ref).
"""
function write_exact_ref_pd(path::String, z::Vector{Float64}, s::Vector{Float64},
                            rho_d::Vector{Float64}, rho_v::Vector{Float64},
                            rho_c::Vector{Float64})
    open(path, "w") do f
        for i in 1:length(z)
            println(f, "$(z[i]) $(s[i]) $(rho_d[i]) $(rho_v[i]) $(rho_c[i])")
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

            patch.physical[i, s_i, 1] = new_s - ref_entropy(ref)[k, 1]
            patch.physical[i, dens_i, 1] = control === :rhod ?
                (new_rho_d - ref_rho_d(ref)[k, 1]) : (new_xi - ref_xi(ref)[k, 1])
            patch.physical[i, mu_i, 1] = new_mu - ref_mu(ref)[k, 1]
            patch.physical[i, ql_i, 1] = new_mu_l
            i += 1
        end
    end
    return patch
end

"""
    moist_buoyancy_bubble_pd!(patch, gridpoints, base, ref; q_t=0.02,
                              xc, xr, zc, zr, amp=2.0/300.0)

Partial-density variant of [`moist_buoyancy_bubble!`](@ref) for the
`primitive_equation_XZ_rhod_pd` set. Identical Bryan & Fritsch (2002) warm-bubble
construction, but writes the moisture state as *partial densities*: the perturbations
`s'`, `rho_d'`, `rho_v' = rho_d·q_v - ref_rho_v`, and `rho_c = rho_d·q_l - ref_rho_c`
(all liquid is cloud, so `rho_r = 0`). `ref` is the physical reference state
(`CondensateReferenceState` for the saturated cloudy base), whose `ref_rho_c` carries the
base cloud so the perturbation is (near) zero outside the bubble. Outside the bubble the
perturbations are at the spectral-smoothing level (raw base minus the smoothed reference),
matching the transformed-variable bubble.
"""
function moist_buoyancy_bubble_pd!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                                   base, ref::Springsteel.AbstractReferenceState; q_t=0.02,
                                   xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0,
                                   amp=2.0/300.0)
    vars = patch.params.vars
    rho_d_i = vars["rho_d"]; rho_v_i = vars["rho_v"]
    rho_c_i = vars["rho_c"]; rho_r_i = vars["rho_r"]
    kDim = patch.params.kDim

    # Slot 1 is either the intensive entropy s' (rhod_pd set) or the entropy density
    # sigma' = rho_d*s - sigmabar (moist_compressible set). Detect which this grid carries.
    sigma_mode = haskey(vars, "sigma")
    s1_i = sigma_mode ? vars["sigma"] : vars["s"]

    rho_vbar = ref_rho_v(ref)
    rcbar = ref_rho_c(ref)
    rho_cbar = rcbar === 0.0 ? zeros(Float64, kDim) : rcbar[:, 1]

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            b_incr = L <= 1.0 ? amp * (cos(pi * L / 2.0))^2 : 0.0

            new_s = base.s[k]
            new_rho_d = base.rho_d[k]
            new_q_v = base.q_v[k]
            new_q_l = base.q_l[k]
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
            end

            patch.physical[i, s1_i, 1] = sigma_mode ?
                (new_rho_d * new_s) - (ref_rho_d(ref)[k, 1] * ref_entropy(ref)[k, 1]) :
                new_s - ref_entropy(ref)[k, 1]
            patch.physical[i, rho_d_i, 1] = new_rho_d - ref_rho_d(ref)[k, 1]
            patch.physical[i, rho_v_i, 1] = (new_rho_d * new_q_v) - rho_vbar[k, 1]
            patch.physical[i, rho_c_i, 1] = (new_rho_d * new_q_l) - rho_cbar[k]
            patch.physical[i, rho_r_i, 1] = 0.0
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

# ── Balanced vortex initialization (tropical cyclone spin-up) ───────────────────

"""
    modified_rankine_v(r, z; Vmax=15.0, RMW=50.0e3, alpha=0.3, v_top=15.0e3)

Modified Rankine vortex tangential wind [m/s]: linear inside the radius of
maximum wind, `Vmax (RMW/r)^alpha` outside, decaying linearly with height to
zero at `v_top` (`v *= max(0, (v_top - z)/v_top)`).
"""
function modified_rankine_v(r, z; Vmax=15.0, RMW=50.0e3, alpha=0.3, v_top=15.0e3)

    vr = r <= RMW ? Vmax * (r / RMW) : Vmax * (RMW / r)^alpha
    return vr * max(0.0, (v_top - z) / v_top)
end

"Centered first derivative on a (possibly nonuniform) axis; one-sided at the ends."
function _ddz_nonuniform!(out::AbstractVector, f::AbstractVector, z::AbstractVector)

    n = length(z)
    out[1] = (f[2] - f[1]) / (z[2] - z[1])
    out[n] = (f[n] - f[n-1]) / (z[n] - z[n-1])
    @inbounds for k in 2:n-1
        h1 = z[k] - z[k-1]
        h2 = z[k+1] - z[k]
        out[k] = ((f[k+1] * h1 * h1) - (f[k-1] * h2 * h2) +
                  (f[k] * ((h2 * h2) - (h1 * h1)))) / (h1 * h2 * (h1 + h2))
    end
    return out
end

"""
    thermal_wind_lnrho(r_axis, z, C, lnrho_bar) -> Matrix (length(z) × length(r_axis))

Solve the thermal-wind equation for the log total density,

    ∂_r ln ρ + (C/g) ∂_z ln ρ = -(1/g) ∂_z C,      C = v²/r + f v,

marching INWARD from the outer edge (`ln ρ = ln ρ̄(z)` at `r_axis[end]`) with a
trapezoidal predictor–corrector in r and centered ∂_z on the (possibly
nonuniform) `z` axis. With `C = 0` (no vortex) the result is `ln ρ̄` exactly.
`C` is a `(length(z), length(r_axis))` matrix.
"""
function thermal_wind_lnrho(r_axis::AbstractVector, z::AbstractVector,
                            C::AbstractMatrix, lnrho_bar::AbstractVector)

    nz = length(z)
    nr = length(r_axis)
    lnrho = zeros(nz, nr)
    dCdz = zeros(nz, nr)
    for j in 1:nr
        _ddz_nonuniform!(view(dCdz, :, j), view(C, :, j), z)
    end
    dlnrho_dz = zeros(nz)
    rhs = zeros(nz)
    pred = zeros(nz)
    rhs_p = zeros(nz)

    lnrho[:, nr] .= lnrho_bar
    for j in (nr-1):-1:1
        dr = r_axis[j+1] - r_axis[j]
        # RHS at the outer column (known)
        _ddz_nonuniform!(dlnrho_dz, view(lnrho, :, j+1), z)
        @. rhs = (-dCdz[:, j+1] / gravity) - ((C[:, j+1] / gravity) * dlnrho_dz)
        @. pred = lnrho[:, j+1] - (dr * rhs)
        # Corrector at the inner column with the predicted profile
        _ddz_nonuniform!(dlnrho_dz, pred, z)
        @. rhs_p = (-dCdz[:, j] / gravity) - ((C[:, j] / gravity) * dlnrho_dz)
        @. lnrho[:, j] = lnrho[:, j+1] - (dr * 0.5 * (rhs + rhs_p))
    end
    return lnrho
end

"""
    balanced_vortex_fields(r_axis, z, pbar, rho_dbar, rho_vbar;
                           Vmax, RMW, alpha, v_top, fcor)
        -> (; v, rho_t, p, rho_d, rho_v, Tk, residual, n_supersat)

Gradient-wind and hydrostatically balanced modified-Rankine vortex on the
`(z, r_axis)` work grid, anchored on the reference profiles at the domain top
and outer edge:

1. `v(r,z)` from [`modified_rankine_v`](@ref), `C = v²/r + f v`.
2. `ρ_t` from the thermal-wind march [`thermal_wind_lnrho`](@ref).
3. `p` by downward hydrostatic integration from `p(r, z_top) = p̄(z_top)`
   (the vortex vanishes above `v_top`, so `ρ = ρ̄` there and the anchor is
   consistent).
4. Moisture: the mixing ratio holds its ambient profile, `q_v(r,z) = q̄_v(z)`,
   so `ρ_d = ρ_t/(1 + q̄_v)`, `ρ_v = ρ_t − ρ_d`, and `T` from the moist EOS.

`residual` is the max relative gradient-wind imbalance `|∂_r p − ρ_t C|` over
the interior (scaled by the max `|∂_r p|`); `n_supersat` counts grid points
pushed past saturation (should be 0 for a warm-core vortex on a subsaturated
sounding).
"""
function balanced_vortex_fields(r_axis::AbstractVector, z::AbstractVector,
                                pbar::AbstractVector, rho_dbar::AbstractVector,
                                rho_vbar::AbstractVector;
                                Vmax=15.0, RMW=50.0e3, alpha=0.3, v_top=15.0e3,
                                fcor=3.775e-5)

    nz = length(z)
    nr = length(r_axis)
    q_vbar = rho_vbar ./ rho_dbar
    rho_tbar = rho_dbar .+ rho_vbar

    v = zeros(nz, nr)
    C = zeros(nz, nr)
    for j in 1:nr, k in 1:nz
        r = r_axis[j]
        vv = modified_rankine_v(r, z[k]; Vmax, RMW, alpha, v_top)
        v[k, j] = vv
        C[k, j] = r > 0.0 ? ((vv * vv) / r) + (fcor * vv) : 0.0
    end

    lnrho = thermal_wind_lnrho(r_axis, z, C, log.(rho_tbar))
    rho_t = exp.(lnrho)

    # Hydrostatic pressure, downward from the reference top
    p = zeros(nz, nr)
    for j in 1:nr
        p[nz, j] = pbar[nz]
        for k in (nz-1):-1:1
            p[k, j] = p[k+1, j] + (0.5 * (rho_t[k, j] + rho_t[k+1, j]) *
                                   gravity * (z[k+1] - z[k]))
        end
    end

    rho_d = rho_t ./ (1.0 .+ q_vbar)     # broadcast q̄_v(z) down the columns
    rho_v = rho_t .- rho_d
    Tk = p ./ ((rho_d .* Rd) .+ (rho_v .* Rv))

    n_supersat = count(rho_v .> rho_v_sat.(Tk, p ./ 100.0))
    n_supersat == 0 ||
        @warn "balanced_vortex_fields: $(n_supersat) points pushed past saturation"

    # Gradient-wind imbalance diagnostic (interior columns)
    residual = 0.0
    dpdr_max = 1.0e-300
    for j in 2:nr-1, k in 1:nz
        dpdr = (p[k, j+1] - p[k, j-1]) / (r_axis[j+1] - r_axis[j-1])
        res = abs(dpdr - (rho_t[k, j] * C[k, j]))
        residual = max(residual, res)
        dpdr_max = max(dpdr_max, abs(dpdr))
    end
    residual /= dpdr_max

    return (; v, rho_t, p, rho_d, rho_v, Tk, residual, n_supersat)
end

"""
    balanced_vortex_mc!(patch, gridpoints, ref, flds, r_axis; zcol=2)

Write the [`balanced_vortex_fields`](@ref) result to the 9-variable
moist_compressible slots of `patch` as perturbations from the
`PressureReferenceState ref`: p′, ρ_d′, ρ_t′, v (prognostic, v̄ ≡ 0), Q_ss′,
and E_t′ from the exact retrieval identity
`E_t = ρ_d e_int(T, q_v) + ρ_t (gz + v²/2)`; u = w = ρ_r = 0. The work grid's
`z` axis must be the patch's own vertical mish (no vertical interpolation);
fields are linearly interpolated in radius from `r_axis`.
"""
function balanced_vortex_mc!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                             ref::Springsteel.PressureReferenceState,
                             flds, r_axis::AbstractVector; zcol=2)

    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    u_i = vars["u"]; w_i = vars["w"]; et_i = vars["E_t"]
    qss_i = vars["Q_ss"]; rho_r_i = vars["rho_r"]; v_i = vars["v"]
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
    E_tbar = ref_total_energy(ref); Q_ssbar = ref_qss(ref)

    nr = length(r_axis)
    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            r = gridpoints[i, 1]
            z = gridpoints[i, zcol]
            # Linear interpolation in radius (clamped to the work-grid extent)
            j = clamp(searchsortedlast(r_axis, r), 1, nr - 1)
            wgt = clamp((r - r_axis[j]) / (r_axis[j+1] - r_axis[j]), 0.0, 1.0)
            interp(F) = ((1.0 - wgt) * F[k, j]) + (wgt * F[k, j+1])
            v = interp(flds.v)
            p = interp(flds.p)
            rho_d = interp(flds.rho_d)
            rho_v = interp(flds.rho_v)
            rho_t = rho_d + rho_v
            Tk = p / ((rho_d * Rd) + (rho_v * Rv))
            q_v = rho_v / rho_d
            E_t = (rho_d * internal_energy_bf02(Tk, q_v, 0.0)) +
                  (rho_t * ((gravity * z) + (0.5 * v * v)))
            Q_ss = rho_v - rho_v_sat(Tk, p / 100.0)
            patch.physical[i, p_i, 1] = p - pbar[k, 1]
            patch.physical[i, rho_d_i, 1] = rho_d - rho_dbar[k, 1]
            patch.physical[i, rho_t_i, 1] = rho_t - rho_tbar[k, 1]
            patch.physical[i, u_i, 1] = 0.0
            patch.physical[i, w_i, 1] = 0.0
            patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
            patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
            patch.physical[i, rho_r_i, 1] = 0.0
            patch.physical[i, v_i, 1] = v
            i += 1
        end
    end
    return patch
end
