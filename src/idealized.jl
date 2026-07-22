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
    modified_rankine_v(r, z; Vmax=15.0, RMW=50.0e3, alpha=0.3, v_top=15.0e3, z_bt=0.0)

Modified Rankine vortex tangential wind [m/s]: linear inside the radius of
maximum wind, `Vmax (RMW/r)^alpha` outside, decaying linearly with height to
zero at `v_top`.

`z_bt` makes the vortex BAROTROPIC (height-independent) below that level, with
the linear decay to `v_top` starting from there:

    v(z) = v(r)                              for z <= z_bt
    v(z) = v(r)·(v_top − z)/(v_top − z_bt)   above

`z_bt = 0` (default) recovers the plain linear profile exactly.

Why it matters: `∂v/∂z = 0` implies `∂C/∂z = 0` (`C = v²/r + fv`), and the
thermal-wind characteristic form solved by [`thermal_wind_lnrho`](@ref) is
`d(ln ρ)/dr = −(1/g)∂C/∂z` along `dz/dr = C/g` — so a barotropic layer generates
NO density anomaly at all. Without it the balanced vortex carries a low-level
warm core peaking near 1 km (+3.9 K at the axis, +3.1 K at the surface for
Vmax = 30), which pushes the surface air ABOVE the SST and reverses the sign of
the surface enthalpy and moisture fluxes — the ocean then cools and dries the
boundary layer instead of powering it, and the vortex simply spins down. Real
TCs keep the warm core aloft and the boundary layer coupled to the sea surface.
A barotropic (or slightly increasing) BL is also what the Louis scheme drives
the flow toward anyway, with the supergradient jet near the BL top.
"""
function modified_rankine_v(r, z; Vmax=15.0, RMW=50.0e3, alpha=0.3, v_top=15.0e3,
                            z_bt=0.0)

    vr = r <= RMW ? Vmax * (r / RMW) : Vmax * (RMW / r)^alpha
    z_bt <= 0.0 && return vr * max(0.0, (v_top - z) / v_top)   # legacy linear, bitwise
    z <= z_bt && return vr
    z >= v_top && return 0.0
    # SMOOTHERSTEP taper above z_bt: S(t) = 6t^5 - 15t^4 + 10t^3 has S = S' = S''
    # = 0 at t = 0 and S = 1, S' = S'' = 0 at t = 1, so v is C2 at BOTH z_bt and
    # v_top -- no kink and no curvature jump anywhere.
    #
    # This matters more than it looks. The thermal-wind march solves
    # d(ln rho)/dr = -(1/g) dC/dz along dz/dr = C/g with C = v^2/r + f v, so any
    # discontinuity in dv/dz lands directly in the density field, and a jump in
    # d2v/dz2 lands in its vertical gradient. A piecewise-LINEAR taper (dv/dz
    # jumping 0 -> -vr/(v_top-z_bt) at z_bt) produced a d(dT)/dz spike of
    # +0.695 K/km against ~-0.05 K/km either side, with a local T maximum just
    # above z_bt, and seeded an exponentially growing deep mode centred there:
    # max|u| in nest2 went 2.6 -> 8.7 -> 17.4 -> 93 m/s over hours 3-6 (e-folding
    # < 1 h) while the otherwise identical z_bt = 0 run stayed bounded near
    # 1 m/s. Preserved in tc/output/tc_500m_barotropic_KINK_unstable/. A cosine
    # taper fixes dv/dz but still jumps d2v/dz2 and left a 0.16 -> 0.88 K/km step.
    t = (z - z_bt) / (v_top - z_bt)
    return vr * (1.0 - (t * t * t * (10.0 + (t * ((6.0 * t) - 15.0)))))
end

"""
    re87_v(r, z; v_m=15.0, r_m=82.5e3, r_0=412.5e3, fcor=5.0e-5, z_sponge=15.0e3)

Rotunno & Emanuel (1987, JAS 44, 542-561) eq. (37) initial tangential wind:

    v(r,z) = (z_s - z)/z_s * { [ v_m^2 (r/r_m)^2 ( (2 r_m/(r+r_m))^3
                                 - (2 r_m/(r_0+r_m))^3 ) + f^2 r^2/4 ]^(1/2) - f r/2 }

`r_0` is the outer radius beyond which `v = 0` (the subtraction makes it vanish
there EXACTLY), and `v_m`, `r_m` are approximately the maximum wind and its radius
(exactly so as `r_0/r_m` and `v_m/(f r_m)` become large). Intensity decays linearly
with height to zero at `z_sponge`, and `v = 0` above.

WHY THIS PROFILE. The cubic `(2r_m/(r+r_m))^3` falloff plus the exact zero at `r_0`
makes the vortex genuinely COMPACT, which is what keeps the thermal-wind warm
anomaly small: RE87 report "a temperature adjustment of at most ~0.6 K at the
vortex center" for their control (r_0 = 412.5 km, r_m = 82.5 km, v_m = 15 giving
v_max ~ 12 m/s, T_surf = 26.3 C). The modified-Rankine profile with alpha = 0.3
used here previously decays as r^-0.3 and was still 12 m/s at r = 1050 km, giving
a +6.9 K surface warm anomaly -- which put the surface air ABOVE the SST and
reversed the air-sea enthalpy and moisture fluxes, so the vortex could only decay.
See reference/HANDOFF_2026-07-19.md.

# `z_round` — rounding off the kink at `z_sponge`

`(z_sponge − z)/z_sponge` truncated at zero leaves `dv/dz` DISCONTINUOUS at
`z_sponge`, and that jump lands in the balanced density exactly the way
[`modified_rankine_v`](@ref) documents for a piecewise-linear `z_bt` taper. It is
measurable: with the native initialization the hydrostatic residual in the outer
nests peaks at z = 15.25 km — the first mish point above `v_top = 15 km` — and
once the wall condition is relaxed it becomes the binding floor over the whole
domain (`model_tests/tc_balance_floor.jl`).

`z_round ∈ (0, 1)` replaces the linear decay ABOVE `z_round·z_sponge` with the
quintic Hermite that matches value and slope there and has zero value, slope AND
curvature at `z_sponge`, so `v` is C² at both joins. The decay below is untouched,
which is what keeps this from reintroducing the `z_bt` disease: a taper that
flattened `dv/dz` near the GROUND would recreate the barotropic layer that drove
dry Ertel PV negative (`tc/tc_params.jl`, `Z_BAROTROPIC` MUST stay 0).

`z_round = 1.0` (the default) is the untouched linear profile, bitwise.
"""
function re87_v(r, z; v_m=15.0, r_m=82.5e3, r_0=412.5e3, fcor=5.0e-5,
                z_sponge=15.0e3, z_round=1.0)

    (z >= z_sponge || r >= r_0) && return 0.0
    a = (2.0 * r_m / (r + r_m))^3
    b = (2.0 * r_m / (r_0 + r_m))^3
    inner = (v_m * v_m * (r / r_m)^2 * (a - b)) + (0.25 * fcor * fcor * r * r)
    v = sqrt(max(inner, 0.0)) - (0.5 * fcor * r)
    t = z / z_sponge
    (z_round >= 1.0 || t <= z_round) && return max(v, 0.0) * (1.0 - t)
    # Quintic Hermite on [z_round, 1] in t: value 1-z_round and slope -1 at the
    # left join (matching the linear decay), value = slope = curvature = 0 at t = 1.
    s = (t - z_round) / (1.0 - z_round)
    s2 = s * s; s3 = s2 * s; s4 = s3 * s; s5 = s4 * s
    A0 = 1.0 - (10.0 * s3) + (15.0 * s4) - (6.0 * s5)      # value basis
    A1 = s - (6.0 * s3) + (8.0 * s4) - (3.0 * s5)          # slope basis
    return max(v, 0.0) * (1.0 - z_round) * (A0 - A1)
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
4. Moisture: by default the mixing ratio holds its ambient profile,
   `q_v(r,z) = q̄_v(z)`, so `ρ_d = ρ_t/(1 + q̄_v)`, `ρ_v = ρ_t − ρ_d`, and `T`
   from the moist EOS. Passing `RH_core` instead moistens the inner core toward
   a target relative humidity (see below).

# Inner-core moisture (`RH_core`)

Holding `q̄_v` fixed in radius makes the core the *driest* column in RH terms —
the warm core raises T at fixed vapor — which is backwards for a developing TC
and loads the sounding with CIN that must be overcome by an explosive release.
With `RH_core` set, the target humidity is blended smoothly from the environment
to the core,

    RH(r,z) = RH̄(z) + W(r,z)·(RH_core − RH̄(z)),   W = exp(−(r/r_moist)²)·τ(z)

with `τ` a cosine taper to zero at `z_moist`, and the blend applied only where it
*moistens* (`RH_core > RH̄`). The repartition holds ρ_t and p FIXED and moves
water between the dry and vapor components, which is **exactly balance
preserving**: gradient-wind and hydrostatic balance constrain only ρ_t and p, so
moisture is a free knob here. `ρ_v = RH·ρ_vs(T,p)` is solved by fixed-point
iteration because `T = p/((ρ_t−ρ_v)R_d + ρ_v R_v)` depends on the partition
(more vapor ⇒ larger R_m ⇒ lower T ⇒ lower ρ_vs, a contracting feedback).

`residual` is the max relative gradient-wind imbalance `|∂_r p − ρ_t C|` over
the interior (scaled by the max `|∂_r p|`); `n_supersat` counts grid points
pushed past saturation (should be 0 for a warm-core vortex on a subsaturated
sounding).
"""
function balanced_vortex_fields(r_axis::AbstractVector, z::AbstractVector,
                                pbar::AbstractVector, rho_dbar::AbstractVector,
                                rho_vbar::AbstractVector;
                                Vmax=15.0, RMW=50.0e3, alpha=0.3, v_top=15.0e3,
                                fcor=3.775e-5, RH_core=nothing, r_moist=150.0e3,
                                z_moist=8.0e3, RH_max=0.98, RH_bl=nothing,
                                z_bl=1.5e3, moist_profile=:gaussian, z_bt=0.0,
                                vortex_profile=:rankine, v_m=15.0, r_m=82.5e3,
                                r_0=412.5e3)

    nz = length(z)
    nr = length(r_axis)
    q_vbar = rho_vbar ./ rho_dbar
    rho_tbar = rho_dbar .+ rho_vbar

    v = zeros(nz, nr)
    C = zeros(nz, nr)
    for j in 1:nr, k in 1:nz
        r = r_axis[j]
        vv = vortex_profile === :re87 ?
             re87_v(r, z[k]; v_m, r_m, r_0, fcor, z_sponge = v_top) :
             modified_rankine_v(r, z[k]; Vmax, RMW, alpha, v_top, z_bt)
        v[k, j] = vv
        C[k, j] = r > 0.0 ? ((vv * vv) / r) + (fcor * vv) : 0.0
    end

    lnrho = thermal_wind_lnrho(r_axis, z, C, log.(rho_tbar))
    rho_t = exp.(lnrho)

    # Hydrostatic pressure in PERTURBATION form: dp'/dz = -rho_t' g integrated
    # downward from p'(z_top) = 0, with p = pbar + p'. Integrating the FULL
    # profile (trapezoid on rho_t g) instead does NOT reproduce the reference
    # state's own spline hydrostatic solve, and the quadrature error accumulates
    # from the top down: at the outer edge, where the vortex vanishes and the
    # fields must reduce to the reference exactly, it left p 4.6 hPa low and T
    # up to 12 K cold. Because q_v holds its ambient profile and rho_vs is
    # exponentially T-sensitive, that cold bias drove RH from 0.32 to 1.09 near
    # the tropopause -- the source of the initial supersaturation and of the
    # spurious condensate layers. In perturbation form the quadrature error acts
    # only on the (small) vortex perturbation, and rho_t' -> 0 in the far field
    # makes it reduce to the reference identically.
    p = zeros(nz, nr)
    for j in 1:nr
        pprime = 0.0
        p[nz, j] = pbar[nz]
        for k in (nz-1):-1:1
            pprime += 0.5 * ((rho_t[k, j] - rho_tbar[k]) +
                             (rho_t[k+1, j] - rho_tbar[k+1])) *
                      gravity * (z[k+1] - z[k])
            p[k, j] = pbar[k] + pprime
        end
    end

    rho_d = rho_t ./ (1.0 .+ q_vbar)     # broadcast q̄_v(z) down the columns
    rho_v = rho_t .- rho_d
    Tk = p ./ ((rho_d .* Rd) .+ (rho_v .* Rv))

    if RH_core !== nothing
        # Environmental RH from the reference column itself
        Tbar = pbar ./ ((rho_dbar .* Rd) .+ (rho_vbar .* Rv))
        RHbar = rho_vbar ./ rho_v_sat.(Tbar, pbar ./ 100.0)
        RHbl = RH_bl === nothing ? RH_core : RH_bl
        for j in 1:nr, k in 1:nz
            taper = z[k] >= z_moist ? 0.0 :
                    0.5 * (1.0 + cos(pi * z[k] / z_moist))
            # Radial weight. :gaussian peaks ON THE AXIS, which loads the most
            # CAPE exactly where the cylindrical 1/r geometry makes convection
            # easiest to trigger -- a small radial convergence u gives a large
            # divergence u/r as r -> 0. In RE87/CM1-class runs the axis stays
            # stable because the storm-scale secondary circulation builds from
            # the BL AWAY from the axis, and once the eyewall ascends, mass
            # continuity forces SUBSIDENCE at the axis. Pre-loading axis CAPE
            # short-circuits that: the axis convects on its own before the eye
            # can establish. (Measured in the crashed 1 km run: CAPE 5861 J/kg at
            # r = 0 vs 3327 at 100 km -- maximum exactly where it is least wanted;
            # the axis lit up at 6 h and ran away to w = 20.9 m/s by 8 h even
            # though the eyewall had properly organized at r = 30-50 km by 7 h.)
            #
            # :vortex weights by the surface tangential wind instead -- zero at
            # the axis, peak at the RMW, decaying outward -- so the initial
            # moisture sits where the SURFACE FLUXES are strongest (fluxes scale
            # with wind speed) and the eye starts dry and stable. The Gaussian
            # still multiplies it to confine the moist annulus radially, since
            # the modified-Rankine tail decays only as r^-alpha.
            W = if moist_profile === :vortex
                vsfc = vortex_profile === :re87 ?
                       re87_v(r_axis[j], 0.0; v_m, r_m, r_0, fcor, z_sponge = v_top) :
                       modified_rankine_v(r_axis[j], 0.0; Vmax, RMW, alpha, v_top, z_bt)
                vscale = vortex_profile === :re87 ? v_m : Vmax
                (vscale > 0.0 ? vsfc / vscale : 0.0) *
                    exp(-((r_axis[j] / r_moist)^2)) * taper
            else
                exp(-((r_axis[j] / r_moist)^2)) * taper
            end
            # Boundary layer and free troposphere get separate targets. CAPE is
            # set by the SURFACE parcel's theta_e, so moistening the BL inflates
            # it; the free-troposphere target is what matters for keeping deep
            # convection steady rather than intermittent (the Dunion MT sounding
            # dries to RH ~0.48 at 5-8 km, and that dry layer is what makes
            # explicit convection downdraft-driven and explosive).
            RHz = z[k] <= z_bl ? RHbl :
                  RHbl + ((RH_core - RHbl) * min(1.0, (z[k] - z_bl) / z_bl))
            target = min(RHbar[k] + (W * (RHz - RHbar[k])), RH_max)
            target <= RHbar[k] && continue          # moisten only, never dry out
            # Fixed point on rho_v at FIXED rho_t and p (balance preserving)
            rv = rho_v[k, j]
            for _ in 1:100
                T = p[k, j] / (((rho_t[k, j] - rv) * Rd) + (rv * Rv))
                rvnew = target * rho_v_sat(T, p[k, j] / 100.0)
                rvnew = clamp(rvnew, 0.0, 0.999 * rho_t[k, j])
                abs(rvnew - rv) < 1.0e-14 && (rv = rvnew; break)
                rv = rvnew
            end
            rv <= rho_v[k, j] && continue           # never remove vapor
            rho_v[k, j] = rv
            rho_d[k, j] = rho_t[k, j] - rv
            Tk[k, j] = p[k, j] / ((rho_d[k, j] * Rd) + (rv * Rv))
        end
    end

    # Never initialize at or above saturation. The reference sounding carries a
    # stratospheric q_v FLOOR (0.01 g/kg in the Dunion MT profile) which becomes
    # supersaturated wherever the vertical mish is too coarse to resolve the
    # tropopause cold point: at 1 km cells the spline undershoots to 187 K (vs
    # 199 K at 300 m), and rho_vs at 187 K is vanishingly small, so 33206
    # work-grid points came out supersaturated at 15.5-20 km. Capping RH is
    # resolution-general and physically correct -- supersaturated initial air
    # just condenses immediately into spurious cloud. Like the core moistening,
    # this holds rho_t and p FIXED and moves mass between the dry and vapor
    # components, so it is exactly balance preserving. It is a no-op (bitwise)
    # at resolutions that resolve the tropopause.
    for j in 1:nr, k in 1:nz
        cap = RH_max * rho_v_sat(Tk[k, j], p[k, j] / 100.0)
        rho_v[k, j] <= cap && continue
        rho_v[k, j] = cap
        rho_d[k, j] = rho_t[k, j] - cap
        Tk[k, j] = p[k, j] / ((rho_d[k, j] * Rd) + (cap * Rv))
    end

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

# ── NATIVE-GRID balanced vortex ────────────────────────────────────────────────
#
# WHY A SECOND CONSTRUCTION. `balanced_vortex_fields` above solves the balance
# analytically on a foreign 500 m radial work grid (FD ∂/∂z, Heun march in r,
# trapezoid hydrostatic integral) and `balanced_vortex_mc!` transfers it onto the
# model by piecewise-LINEAR radial interpolation. It reports a gradient-wind
# residual of 3.6e-4, but that number is measured with ITS OWN operators on ITS OWN
# grid. Under the model's operators the hydrostatic residual is 5.1e-3 -- 14x
# larger (model_tests/tc_discrete_balance.jl) -- and it is the largest imbalance
# left in the TC system now that the reference state is an exact discrete fixed
# point.
#
# The reason a "more accurate analytic solution" cannot close that gap is that the
# model's fit is NOT INTERPOLATORY: l_q defaults to 2.0, so
# spectralTransform!/gridTransform! smooth. Handing the model analytically balanced
# VALUES leaves it balanced only to the fit error of a DERIVATIVE. The fix is to put
# the fit inside the solve -- choose mish values so that the FITTED field is
# balanced. This is the same move that made the reference state exact (Springsteel
# `_hydrostatic_pressure_profile`: integrate, do not re-fit).
#
# THE EQUATIONS, in the model's own variables. The two residuals the equation set
# actually computes are (mc_geometry.jl `mc_u_forcing!`, moist_compressible.jl
# slots 4-5)
#
#     expdot[u] = -pp_x/ρ_t + (f + v/r)·v = -(pp_x - ρ_t·C)/ρ_t,  C = v²/r + f·v
#     dw/dt     = -(pp_z + g·ρ_t')/ρ_t
#
# with pp_x, pp_z, v and ρ_t' all FITTED values. Eliminating ρ_t' between the two
# gives ONE equation, and it is exactly LINEAR in p' -- no thermal-wind log-density
# march, no nonlinearity:
#
#     ∂p'/∂r + (C/g)·∂p'/∂z = ρ̄_t·C ,     ρ_t' = -(1/g)·∂p'/∂z
#
# The characteristic slope C/g is ~2.6e-4 and the (C/g)·∂p'/∂z term is ~1 % of
# ρ̄_t·C, so a Picard sweep -- correct p' by the radial antiderivative of the
# current gradient-wind residual -- contracts at ~0.01 per iteration.
#
# THE NEST JUNCTION IS A RANK-3 CONSTRAINT, NOT AN ANCHOR. `apply_interface_payload!`
# writes THREE border spline coefficients into the child's `ahat`, and
# `set_ahat_r3x!` shows what they encode: u(x₀), u'(x₀) AND u''(x₀) -- for every
# variable, not just p. A child solved in isolation with only its junction VALUE
# matched would have its dp'/dr and d²p'/dr² overwritten on the first exchange,
# which is precisely the discrepancy this function exists to remove. So every fit
# inside a child's iteration runs with the parent's payload applied, in the same
# spectralTransform! -> apply_interface_payload! -> gridTransform! order the model
# itself uses, and patches are solved OUTERMOST FIRST so a child only ever inherits
# a converged trio. Consistency at the junction is then INHERITED rather than
# imposed: v and ρ_t are R3X'd too, so the child's fitted v and ρ_t at the junction
# ARE the parent's, and a parent satisfying ∂p'/∂r = ρ_t·C there hands down the
# right u'(x₀).

"""
    _nest_parent_order(topo, npatch) -> Vector{Int}

Patch indices ordered so every patch follows all of its parents. Mirrors the
interface-graph walk in `load_all_patches` (model_tests/tc_discrete_balance.jl),
but over patches rather than interfaces.
"""
function _nest_parent_order(topo, npatch::Int)
    order = Int[]
    done = falses(npatch)
    while length(order) < npatch
        progressed = false
        for i in 1:npatch
            done[i] && continue
            all(done[topo.interfaces[j].parent] for j in topo.parent_ifaces[i]) || continue
            push!(order, i)
            done[i] = true
            progressed = true
        end
        progressed || error("nest interface graph has a cycle")
    end
    return order
end

"""
    _native_radial_column(gp) -> Spline1D

A natural-BC (R0) radial basis sharing the patch's own i-mish, for use as an
integrator. Mirrors `Springsteel.natural_column`, which does the same for the
VERTICAL basis: reference/integrand profiles can have nonzero boundary gradients,
so the model variables' wall conditions must not be imposed on them.

Every `grid.ibasis.data[z, v]` on a RiRk patch is built from exactly these
parameters (Springsteel factory.jl), so the mish points coincide.

`l_q = 0.0` IS LOAD-BEARING, and it is not the same knob as the model's `l_q`.
Ooyama's Q-penalty is a smoothing constraint on the fitted curvature, and while it
costs almost nothing on a VALUE fit (1e-4 relative) or a DERIVATIVE (7e-8), it
wrecks the ANTIDERIVATIVE: measured on `exp(-(r/60 km)²)` over this patch's own
15-cell radial basis, recovering `f` from `df/dr` via `IInttransform` is in error by
**13 %** at `l_q = 2.0` and by 4e-5 at `l_q = 0.0`. With the smoothed integrator the
Picard sweep below stalls at a gradient-wind residual of 2e-2 m/s² -- 1000x WORSE
than the analytic init it replaces -- because the correction it applies is not the
antiderivative of the residual it measured. This column is a private numerical
tool, not a model field, so there is nothing for the penalty to regularize here.
"""
function _native_radial_column(gp)
    return Spline1D(SplineParameters(
        xmin = gp.iMin, xmax = gp.iMax, num_cells = gp.num_cells,
        mubar = gp.mubar, quadrature = gp.quadrature, l_q = 0.0,
        BCL = Springsteel.CubicBSpline.R0, BCR = Springsteel.CubicBSpline.R0))
end

"""
    _native_fit!(patch, mish, payloads)

The model's own fit operator: install `mish` as the patch's physical values, run
`spectralTransform!`, apply every incoming R3X payload, then `gridTransform!`.

This is exactly the chain `load_initial_conditions!` + the t = 0 interface exchange
perform, so a state converged under this operator is reproduced bit-for-bit by
writing `mish` to the IC file and letting the model load it. Note `mish` is kept
SEPARATE from `patch.physical`: `gridTransform!` overwrites slice 1 with the
FITTED values, so reusing the patch array as the control variable would silently
iterate F∘F∘F… instead of F.
"""
function _native_fit!(patch, mish::AbstractMatrix{Float64}, payloads)
    patch.physical[:, :, 1] .= mish
    spectralTransform!(patch)
    for (meta, payload) in payloads
        Springsteel.apply_interface_payload!(meta, patch, payload)
    end
    gridTransform!(patch)
    return patch.physical
end

"""
    _radial_antiderivative!(out, G, rcol, kDim, ncol)

Level-by-level radial antiderivative of `G`, shifted to vanish at the OUTERMOST
mish point: `out(r) = ∫ G - ∫ G |_(r = r_end)`. Rows are the RiRk layout,
`(c-1)*kDim + k`.

Integrate, do not re-fit: `IInttransform` returns the antiderivative SPLINE of the
fitted integrand, so `d(out)/dr` is the fit of `G` by construction. Re-fitting the
integrated values instead is the defect that cost 17 % in the reference-state
hydrostatic profile.
"""
function _radial_antiderivative!(out::AbstractVector{Float64}, G::AbstractVector{Float64},
                                 rcol, kDim::Int, ncol::Int)
    for k in 1:kDim
        @inbounds for c in 1:ncol
            rcol.uMish[c] = G[((c - 1) * kDim) + k]
        end
        Btransform!(rcol)
        Atransform!(rcol)
        A = IInttransform(rcol, 0.0)
        @inbounds for c in 1:ncol
            out[((c - 1) * kDim) + k] = A[c] - A[ncol]
        end
    end
    return out
end

"""
    _native_vertical_column(gp) -> Spline1D

The vertical twin of [`_native_radial_column`](@ref): a natural-BC, unpenalized
column on the patch's own k-mish, used to integrate the hydrostatic residual.
`l_q = 0.0` for the same reason -- see that function.
"""
function _native_vertical_column(gp)
    return Spline1D(SplineParameters(
        xmin = gp.kMin, xmax = gp.kMax, num_cells = gp.num_cells_k,
        mubar = gp.mubar, quadrature = gp.quadrature, l_q = 0.0,
        BCL = Springsteel.CubicBSpline.R0, BCR = Springsteel.CubicBSpline.R0))
end

"""
    _vertical_antiderivative!(out, H, kcol, kDim, ncol)

Column-by-column vertical antiderivative of `H`, shifted to vanish at the TOP mish
point. The top is the right anchor: the vortex vanishes above `v_top`, so `p'` is
already zero there and the correction must not move it.
"""
function _vertical_antiderivative!(out::AbstractVector{Float64}, H::AbstractVector{Float64},
                                   kcol, kDim::Int, ncol::Int)
    for c in 1:ncol
        base = (c - 1) * kDim
        @inbounds for k in 1:kDim
            kcol.uMish[k] = H[base + k]
        end
        Btransform!(kcol)
        Atransform!(kcol)
        A = IInttransform(kcol, 0.0)
        @inbounds for k in 1:kDim
            out[base + k] = A[k] - A[kDim]
        end
    end
    return out
end

"""
    _native_deconvolve!(mish, patch, vi, target, payloads; iters, verbose)

Choose mish values for variable slot `vi` so that the FITTED field matches
`target`, by Picard iteration `mish ← mish + (target − F·mish)`. Returns the best
achieved `max|target − F·mish|`, and leaves `mish` at the best iterate.

Why this is needed at all: with `l_q = 2.0` the fit smooths, so `F·target ≠ target`
and simply writing the target leaves a value error that lands directly in the
hydrostatic residual as `g·(ρ̂_t' − ρ_t')`.

Why the BEST iterate rather than the last: at an R3X junction the border
coefficients come from the parent and cannot be moved, and under a `d²/dz² = 0`
wall the required curvature is not in the basis. In those rows `(I − F)` has a
unit eigendirection, so the iteration stalls there (and the mish value drifts)
while it converges everywhere else. Keeping the best iterate bounds the drift; the
returned error is the honest floor.
"""
function _native_deconvolve!(mish::AbstractMatrix{Float64}, patch, vi::Int,
                             target::AbstractVector{Float64}, payloads;
                             iters::Int = 8)
    npts = length(target)
    best = fill(0.0, npts)
    best .= view(mish, :, vi)
    best_err = Inf
    best_idx = 1
    work = Vector{Float64}(undef, npts)
    for _ in 1:iters
        _native_fit!(patch, mish, payloads)
        @inbounds for i in 1:npts
            work[i] = target[i] - patch.physical[i, vi, 1]
        end
        idx = argmax(abs.(work))
        err = abs(work[idx])
        if err < best_err
            best_err = err
            best_idx = idx
            best .= view(mish, :, vi)
        end
        @inbounds for i in 1:npts
            mish[i, vi] += work[i]
        end
    end
    mish[:, vi] .= best
    _native_fit!(patch, mish, payloads)
    return best_err, best_idx
end

"""
    balanced_vortex_native!(patches, topo, ref; kwargs...) -> Vector{Matrix{Float64}}

Solve the gradient-wind / hydrostatic balance for a tropical-cyclone vortex
**on the model's own mish, under the model's own fit**, and return one
`(npts, nvars)` matrix of mish values per patch, in the perturbation form
`load_initial_conditions!` expects (i.e. ready for [`write_ics_csv`](@ref) once
copied into `patch.physical[:, :, 1]`, which this function also does).

`patches` are the real patch grids (real BCs), `topo` the `NestTopology` from
`build_nest`, and `ref` the shared `PressureReferenceState` on the common vertical
mish. The vortex profile keywords are the same as [`balanced_vortex_fields`](@ref).

See the long comment above for the derivation. In outline, per patch, outermost
first:

1. `v` from `re87_v`/`modified_rankine_v`, fitted; `C = v²/r + f·v` from the
   FITTED `v`, since that is the `v` the `u` tendency sees.
2. Picard sweeps on `p'`: correct by the radial antiderivative of the current
   gradient-wind residual `G = pp_x − ρ_t·C`, with `ρ_t'` re-derived from the
   fitted `pp_z` each sweep and deconvolved so its FITTED value matches `−pp_z/g`.
3. Moisture, `E_t` and `Q_ss` diagnosed pointwise from the converged `(p, ρ_t)` —
   exactly balance preserving, since they hold `ρ_t` and `p` fixed — then
   deconvolved so their fitted values match too.

Prints the discrete residuals per patch (`verbose`), bucketed into interior, wall
cells and junction cells, because the last two carry irreducible floors: a
`d²/dz² = 0` wall cannot represent the balanced curvature, and an R3X junction
inherits its trio from a coarser parent.
"""
function balanced_vortex_native!(patches::AbstractVector, topo,
                                 ref::Springsteel.PressureReferenceState;
                                 zcol::Int = 2, fcor = 3.775e-5,
                                 vortex_profile = :re87, v_m = 15.0, r_m = 82.5e3,
                                 r_0 = 412.5e3, Vmax = 15.0, RMW = 50.0e3,
                                 alpha = 0.3, v_top = 15.0e3, z_bt = 0.0,
                                 RH_core = nothing, r_moist = 150.0e3,
                                 z_moist = 8.0e3, RH_max = 0.98, RH_bl = nothing,
                                 z_bl = 1.5e3, moist_profile = :gaussian,
                                 z_round = 1.0,
                                 outer_iters::Int = 20, inner_iters::Int = 8,
                                 tol = 1.0e-11, verbose::Bool = true)

    npatch = length(patches)
    pbar     = ref_pressure(ref)[:, 1]
    rho_dbar = ref_rho_d(ref)[:, 1]
    rho_vbar = ref_rho_v(ref)[:, 1]
    rho_tbar = ref_rho_t(ref)[:, 1]
    E_tbar   = ref_total_energy(ref)[:, 1]
    Q_ssbar  = ref_qss(ref)[:, 1]
    q_vbar   = rho_vbar ./ rho_dbar
    Tbar     = pbar ./ ((rho_dbar .* Rd) .+ (rho_vbar .* Rv))
    RHbar    = rho_vbar ./ rho_v_sat.(Tbar, pbar ./ 100.0)
    RHbl     = RH_bl === nothing ? RH_core : RH_bl

    mishes   = Vector{Matrix{Float64}}(undef, npatch)
    payloads = [Vector{Any}() for _ in 1:npatch]   # incoming R3X data per patch
    diag     = Vector{Any}(undef, npatch)

    for ip in _nest_parent_order(topo, npatch)
        patch = patches[ip]
        gp    = patch.params
        vars  = gp.vars
        p_i = vars["p"]; rd_i = vars["rho_d"]; rt_i = vars["rho_t"]
        u_i = vars["u"]; w_i = vars["w"];      et_i = vars["E_t"]
        qs_i = vars["Q_ss"]; rr_i = vars["rho_r"]; v_i = vars["v"]

        gpts = getGridpoints(patch)
        kDim = gp.kDim
        ncol = num_columns(patch)
        npts = ncol * kDim
        nvar = length(vars)
        rmish = [gpts[((c - 1) * kDim) + 1, 1] for c in 1:ncol]
        zmish = [gpts[k, zcol] for k in 1:kDim]
        rcol  = _native_radial_column(gp)
        kcol  = _native_vertical_column(gp)

        # Which end of the patch anchors the radial integration: the side facing
        # the parent for a child (its R3X trio fixes the level there), the far
        # field for the root (where the compact vortex has p' == 0 exactly).
        pifs = topo.parent_ifaces[ip]
        anchor_right = isempty(pifs) ? true : (topo.interfaces[pifs[1]].child_side === :right)
        anchor_c = anchor_right ? ncol : 1
        parent_edge = if isempty(pifs)
            nothing
        else
            itf = topo.interfaces[pifs[1]]
            ev = Springsteel.evaluate_grid_ipoints(patches[itf.parent], [rmish[anchor_c]])
            ev[1:kDim, p_i, 1]        # parent p' on the shared vertical mish
        end

        mish = zeros(Float64, npts, nvar)
        # ── (1) tangential wind, then C from the FITTED v ──────────────────────
        i = 1
        for c in 1:ncol, k in 1:kDim
            r = rmish[c]; z = zmish[k]
            mish[i, v_i] = vortex_profile === :re87 ?
                re87_v(r, z; v_m, r_m, r_0, fcor, z_sponge = v_top, z_round) :
                modified_rankine_v(r, z; Vmax, RMW, alpha, v_top, z_bt)
            i += 1
        end
        _native_fit!(patch, mish, payloads[ip])
        vfit = copy(view(patch.physical, 1:npts, v_i, 1))
        Cfit = similar(vfit)
        i = 1
        for c in 1:ncol, _ in 1:kDim
            r = rmish[c]
            Cfit[i] = r > 0.0 ? ((vfit[i] * vfit[i]) / r) + (fcor * vfit[i]) :
                                fcor * vfit[i]
            i += 1
        end

        # ── (2) Picard on p', with ρ_t' deconvolved from the fitted pp_z ───────
        G     = Vector{Float64}(undef, npts)
        H     = Vector{Float64}(undef, npts)
        Phi   = Vector{Float64}(undef, npts)
        Psi   = Vector{Float64}(undef, npts)
        rhot_target = Vector{Float64}(undef, npts)
        gw_res = Inf; hy_res = Inf; rt_fiterr = Inf; rt_fitidx = 1
        for it in 1:outer_iters
            _native_fit!(patch, mish, payloads[ip])
            i = 1
            for _ in 1:ncol, k in 1:kDim
                rhot_target[i] = -patch.physical[i, p_i, 4] / gravity   # slot 4 = ∂z
                i += 1
            end
            mish[:, rt_i] .= rhot_target
            rt_fiterr, rt_fitidx = _native_deconvolve!(mish, patch, rt_i, rhot_target,
                                                      payloads[ip]; iters = inner_iters)
            # G = pp_x - ρ_t·C  ==  -ρ_t·expdot[u]   (the gradient-wind residual)
            # H = pp_z + g·ρ_t'                      (the hydrostatic residual)
            i = 1
            gw_res = 0.0; hy_res = 0.0
            for _ in 1:ncol, k in 1:kDim
                rtp   = patch.physical[i, rt_i, 1]
                rho_t = rtp + rho_tbar[k]
                G[i]  = patch.physical[i, p_i, 2] - (rho_t * Cfit[i])
                H[i]  = patch.physical[i, p_i, 4] + (gravity * rtp)
                gw_res = max(gw_res, abs(G[i]) / rho_t)
                hy_res = max(hy_res, abs(H[i]) / rho_t)
                i += 1
            end
            verbose && println("    patch $ip sweep $it  |gw| = " *
                               "$(round(gw_res; sigdigits=4))  |hyd| = " *
                               "$(round(hy_res; sigdigits=4))  " *
                               "(rho_t fit $(round(rt_fiterr; sigdigits=4)))")
            max(gw_res, hy_res) < tol && break
            # BOTH legs correct the SAME field p'. That is legitimate rather than
            # over-determined because the two targets are compatible to the extent
            # the discrete thermal-wind relation holds, and it is NECESSARY because
            # the leftover ρ_t' fit error is not removable from ρ_t: the sounding's
            # kinks (the Dunion MT profile has a level at 810 m, where d²ρ̄_t/dz²
            # changes sign) reach ρ_t' through ρ̄_t·C, and a cubic spline cannot hold
            # that curvature jump -- the ρ_t deconvolution stalls at 1.3e-4 kg/m³ no
            # matter how many sweeps it is given (measured: identical at 8 and 40).
            # Letting p' bend instead moves the error into ∂p'/∂z, where it costs
            # only a small ∂r perturbation, because the fit error is localized in z
            # and smooth in r.
            _vertical_antiderivative!(Psi, H, kcol, kDim, ncol)
            @inbounds for j in 1:npts
                mish[j, p_i] -= Psi[j]
            end
            _native_fit!(patch, mish, payloads[ip])
            i = 1
            for _ in 1:ncol, k in 1:kDim
                rho_t = patch.physical[i, rt_i, 1] + rho_tbar[k]
                G[i]  = patch.physical[i, p_i, 2] - (rho_t * Cfit[i])
                i += 1
            end
            _radial_antiderivative!(Phi, G, rcol, kDim, ncol)
            # Re-anchor a child on its parent: R3X pins the junction trio, but the
            # fit is a compromise between that and the mish values, so the DC mode
            # converges slowly without this. A no-op for the root.
            if parent_edge !== nothing
                for k in 1:kDim
                    dc = patch.physical[((anchor_c - 1) * kDim) + k, p_i, 1] - parent_edge[k]
                    for c in 1:ncol
                        Phi[((c - 1) * kDim) + k] += dc
                    end
                end
            end
            @inbounds for j in 1:npts
                mish[j, p_i] -= Phi[j]
            end
        end
        _native_fit!(patch, mish, payloads[ip])

        # ── (3) moisture, E_t and Q_ss from the converged (p, ρ_t) ─────────────
        rd_target = Vector{Float64}(undef, npts)
        et_target = Vector{Float64}(undef, npts)
        qs_target = Vector{Float64}(undef, npts)
        n_supersat = 0
        i = 1
        for c in 1:ncol, k in 1:kDim
            r = rmish[c]; z = zmish[k]
            p     = patch.physical[i, p_i, 1] + pbar[k]
            rho_t = patch.physical[i, rt_i, 1] + rho_tbar[k]
            v     = vfit[i]
            rho_d = rho_t / (1.0 + q_vbar[k])
            rho_v = rho_t - rho_d
            Tk    = p / ((rho_d * Rd) + (rho_v * Rv))
            if RH_core !== nothing
                rho_v, rho_d, Tk = _native_moisten(rho_v, rho_d, Tk, p, rho_t, r, z,
                                                   RHbar[k], RHbl, RH_core, RH_max,
                                                   r_moist, z_moist, z_bl, moist_profile,
                                                   vortex_profile, v_m, r_m, r_0, fcor,
                                                   Vmax, RMW, alpha, v_top, z_bt, z_round)
            end
            cap = RH_max * rho_v_sat(Tk, p / 100.0)
            if rho_v > cap
                rho_v = cap
                rho_d = rho_t - cap
                Tk = p / ((rho_d * Rd) + (cap * Rv))
            end
            rho_v > rho_v_sat(Tk, p / 100.0) && (n_supersat += 1)
            q_v = rho_v / rho_d
            E_t = (rho_d * internal_energy_bf02(Tk, q_v, 0.0)) +
                  (rho_t * ((gravity * z) + (0.5 * v * v)))
            rd_target[i] = rho_d - rho_dbar[k]
            et_target[i] = E_t - E_tbar[k]
            qs_target[i] = (rho_v - rho_v_sat(Tk, p / 100.0)) - Q_ssbar[k]
            i += 1
        end
        mish[:, u_i] .= 0.0
        mish[:, w_i] .= 0.0
        mish[:, rr_i] .= 0.0

        # CHAIN THE TARGETS OFF SETTLED FITTED VALUES, in dependency order. The
        # model diagnoses vapor as ρ_v = ρ_t − ρ_d from the fields it actually
        # holds, so a Q_ss built from the ρ_d TARGET is wrong by ρ_d's fit error
        # (6.3e-4 kg/m³ here) -- which is the whole of the Q_ss floor, and shows up
        # as a spurious condensation source in the p tendency. Settle ρ_d first,
        # then re-derive ρ_v, T, E_t and Q_ss from the FITTED ρ_d and ρ_t.
        mish[:, rd_i] .= rd_target
        rd_err, _ = _native_deconvolve!(mish, patch, rd_i, rd_target, payloads[ip];
                                        iters = inner_iters)
        i = 1
        for c in 1:ncol, k in 1:kDim
            z = zmish[k]
            p     = patch.physical[i, p_i, 1] + pbar[k]
            rho_t = patch.physical[i, rt_i, 1] + rho_tbar[k]
            rho_d = patch.physical[i, rd_i, 1] + rho_dbar[k]
            rho_v = rho_t - rho_d
            Tk    = p / ((rho_d * Rd) + (rho_v * Rv))
            et_target[i] = ((rho_d * internal_energy_bf02(Tk, rho_v / rho_d, 0.0)) +
                            (rho_t * ((gravity * z) + (0.5 * vfit[i] * vfit[i])))) - E_tbar[k]
            qs_target[i] = (rho_v - rho_v_sat(Tk, p / 100.0)) - Q_ssbar[k]
            i += 1
        end
        mish[:, et_i] .= et_target
        mish[:, qs_i] .= qs_target
        et_err, _ = _native_deconvolve!(mish, patch, et_i, et_target, payloads[ip];
                                        iters = inner_iters)
        qs_err, _ = _native_deconvolve!(mish, patch, qs_i, qs_target, payloads[ip];
                                        iters = inner_iters)

        # Final fit, then hand the converged trio down to this patch's children.
        _native_fit!(patch, mish, payloads[ip])
        for j in topo.child_ifaces[ip]
            itf = topo.interfaces[j]
            push!(payloads[itf.child], (itf.meta, compute_interface_payload(itf.meta, patch)))
        end

        diag[ip] = _native_residual_report(patch, ip, gp, kDim, ncol, rho_tbar, Cfit,
                                           anchor_right, isempty(pifs), verbose)
        verbose && println("    patch $ip rho_t fit floor at r = " *
                           "$(round(gpts[rt_fitidx, 1] / 1e3; digits=1)) km, z = " *
                           "$(round(gpts[rt_fitidx, zcol] / 1e3; digits=2)) km")
        verbose && println("    patch $ip fit floors: rho_t $(round(rt_fiterr; sigdigits=3))  " *
                           "rho_d $(round(rd_err; sigdigits=3))  " *
                           "E_t $(round(et_err; sigdigits=3))  " *
                           "Q_ss $(round(qs_err; sigdigits=3))  supersat $n_supersat")

        # `write_ics_csv` reads slice 1, and the IC file must carry the CONTROL
        # values, not the fitted ones -- the model re-fits on load.
        patch.physical[:, :, 1] .= mish
        mishes[ip] = mish
    end
    return mishes
end

# Inner-core moistening for the native path: blend RH from the environment toward
# `RH_core` (`RHbl` below `z_bl`), holding ρ_t and p FIXED so the repartition is
# exactly balance preserving -- gradient-wind and hydrostatic balance constrain only
# ρ_t and p, so moisture is a free knob. The fixed point on ρ_v converges because
# more vapor => larger R_m => lower T => lower ρ_vs, a contracting feedback.
# Mirrors the block in `balanced_vortex_fields`; kept separate so that function
# stays bitwise untouched.
function _native_moisten(rho_v, rho_d, Tk, p, rho_t, r, z, RHbar, RHbl, RH_core,
                         RH_max, r_moist, z_moist, z_bl, moist_profile,
                         vortex_profile, v_m, r_m, r_0, fcor, Vmax, RMW, alpha,
                         v_top, z_bt, z_round = 1.0)

    taper = z >= z_moist ? 0.0 : 0.5 * (1.0 + cos(pi * z / z_moist))
    W = if moist_profile === :vortex
        vsfc = vortex_profile === :re87 ?
               re87_v(r, 0.0; v_m, r_m, r_0, fcor, z_sponge = v_top, z_round) :
               modified_rankine_v(r, 0.0; Vmax, RMW, alpha, v_top, z_bt)
        vscale = vortex_profile === :re87 ? v_m : Vmax
        (vscale > 0.0 ? vsfc / vscale : 0.0) * exp(-((r / r_moist)^2)) * taper
    else
        exp(-((r / r_moist)^2)) * taper
    end
    RHz = z <= z_bl ? RHbl : RHbl + ((RH_core - RHbl) * min(1.0, (z - z_bl) / z_bl))
    target = min(RHbar + (W * (RHz - RHbar)), RH_max)
    target <= RHbar && return (rho_v, rho_d, Tk)

    rv = rho_v
    for _ in 1:100
        T = p / (((rho_t - rv) * Rd) + (rv * Rv))
        rvnew = clamp(target * rho_v_sat(T, p / 100.0), 0.0, 0.999 * rho_t)
        abs(rvnew - rv) < 1.0e-14 && (rv = rvnew; break)
        rv = rvnew
    end
    rv <= rho_v && return (rho_v, rho_d, Tk)          # never remove vapor
    rd = rho_t - rv
    return (rv, rd, p / ((rd * Rd) + (rv * Rv)))
end

"""
    _native_residual_report(patch, ip, gp, kDim, ncol, rho_tbar, Cfit,
                            anchor_right, is_root, verbose) -> NamedTuple

Discrete gradient-wind and hydrostatic residuals of the patch's FITTED state
[m/s²], bucketed. The two boundary buckets exist because each carries a floor that
the solve cannot remove and that must therefore be reported rather than averaged
away:

- `wall`   -- the end cells in z. `SecondDerivativeBC` forces `d²/dz² = 0` there,
              and the balanced ρ_t' has nonzero curvature, so its deconvolution
              stalls in those rows.
- `junction` -- the end cells in r on the parent side of a child patch, whose
              three border coefficients are the parent's and are not ours to move.
"""
function _native_residual_report(patch, ip, gp, kDim, ncol, rho_tbar, Cfit,
                                 anchor_right, is_root, verbose)
    vars = gp.vars
    p_i = vars["p"]; rt_i = vars["rho_t"]
    mubar = gp.mubar
    npts = ncol * kDim
    ures = Vector{Float64}(undef, npts)
    wres = Vector{Float64}(undef, npts)
    i = 1
    for _ in 1:ncol, k in 1:kDim
        rtp = patch.physical[i, rt_i, 1]
        rho_t = rtp + rho_tbar[k]
        ures[i] = -(patch.physical[i, p_i, 2] - (rho_t * Cfit[i])) / rho_t
        wres[i] = -(patch.physical[i, p_i, 4] + (gravity * rtp)) / rho_t
        i += 1
    end
    wall_k = vcat(1:mubar, (kDim - mubar + 1):kDim)
    jrows = is_root ? Int[] :
            (anchor_right ? [((c - 1) * kDim) + k for c in (ncol - mubar + 1):ncol for k in 1:kDim] :
                            [((c - 1) * kDim) + k for c in 1:mubar for k in 1:kDim])
    wrows = [((c - 1) * kDim) + k for c in 1:ncol for k in wall_k]
    interior = setdiff(1:npts, union(wrows, jrows))
    m(x, idx) = isempty(idx) ? 0.0 : maximum(abs, view(x, idx))
    rep = (; gw_all = maximum(abs, ures), hy_all = maximum(abs, wres),
             gw_int = m(ures, interior), hy_int = m(wres, interior),
             gw_wall = m(ures, wrows),   hy_wall = m(wres, wrows),
             gw_junc = m(ures, jrows),   hy_junc = m(wres, jrows))
    if verbose
        f(x) = string(round(x; sigdigits = 4))
        println("    patch $ip residual [m/s^2]        gradient-wind    hydrostatic")
        println("      all              " * rpad(f(rep.gw_all), 17) * f(rep.hy_all))
        println("      interior         " * rpad(f(rep.gw_int), 17) * f(rep.hy_int))
        println("      wall cells (z)   " * rpad(f(rep.gw_wall), 17) * f(rep.hy_wall))
        is_root || println("      junction cells   " * rpad(f(rep.gw_junc), 17) * f(rep.hy_junc))
    end
    return rep
end
