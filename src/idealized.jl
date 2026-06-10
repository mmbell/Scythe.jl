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
    theta = potential_temperature.(ref.sbar[:, 1], ref.xibar[:, 1], q_v)
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
