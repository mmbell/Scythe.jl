# test/vapor_transform_fixtures.jl -- shared fixtures for test/test_vapor_transform.jl (the
# vapor control variable, `options[:vapor_transform]`, Stage 1 of the rho_t retirement plan).
#
# The tile is the EOS-temperature one (test_eos_temperature.jl): a periodic XZ RiRk slice with
# 500 m vertical cells (the RCE's dz), horizontally uniform seeds, over a HYDROSTATIC reference
# so `:hydrostatic_reference` and `:consistent_qss_reference` -- the production reference build
# -- can be on and a resting column is a discrete fixed point. Loaded with `include`, like
# mynn_test_fixtures.jl; it is not a test file of its own.
module VaporTransformFixtures

using Scythe
using Springsteel
using SparseArrays
using Springsteel: PeriodicBC, NeumannBC, DirichletBC
import Springsteel.Thermodynamics: rho_v_sat, L_v, Rd, Rv, Cpv, gravity

const SPAN = 8.0e3
const NCELL = 4
const ZTOP = 10.0e3
const NCELL_K = 20
const MU = 1.0e-7

"""Reference column: 6.5 K/km lapse rate, RH `rh(z)`, HYDROSTATIC (dp/dz = -g rho_t by RK4
on 10 m sub-steps), so `:hydrostatic_reference` accepts it."""
function vt_column(z, rh)
    Tf(zz) = 300.0 - 0.0065 * zz
    function rho_t_of(zz, p)
        T = Tf(zz); rv = rh(zz) * rho_v_sat(T, p / 100.0)
        rd = (p - Rv * T * rv) / (Rd * T)
        return rd + rv
    end
    f(zz, p) = -gravity * rho_t_of(zz, p)
    p_Pa = similar(z); zc = 0.0; pc = 100000.0
    for (k, zk) in enumerate(z)
        n = max(1, ceil(Int, (zk - zc) / 10.0)); h = (zk - zc) / n
        for _ in 1:n
            k1 = f(zc, pc); k2 = f(zc + h / 2, pc + h * k1 / 2)
            k3 = f(zc + h / 2, pc + h * k2 / 2); k4 = f(zc + h, pc + h * k3)
            pc += h * (k1 + 2k2 + 2k3 + k4) / 6; zc += h
        end
        p_Pa[k] = pc
    end
    Tk = Tf.(z)
    rho_v = rh.(z) .* rho_v_sat.(Tk, p_Pa ./ 100.0)
    rho_d = (p_Pa .- (Rv .* Tk .* rho_v)) ./ (Rd .* Tk)
    return (; Tk, p_Pa, rho_d, rho_v, rho_c = zeros(length(z)))
end

"""The periodic XZ tile. `vtr` is `options[:vapor_transform]`, or `nothing` to leave the key
absent (the library default). `seed!(patch, pts, vars, col)` writes the PHYSICAL perturbation
state through the slot helpers, which is then refit (spectral + grid transform). `ice` adds
two-moment rain and the ISHMAEL ice family, `louis` the Louis boundary layer with surface
fluxes."""
function vt_tile(tmpdir, tag; seed! = (p, x, V, col) -> nothing, vtr = nothing,
                 ice = false, louis = false, sfc = true, tr = :bhyp, ts = 0.5, rh = z -> 0.7,
                 cqss = true, hydro = true, extra = Dict{Symbol,Any}(),
                 pp_extra = Dict{Symbol,Float64}(), output_dir = "./output/",
                 initial_conditions = "ics.csv")
    opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                            :consistent_qss_reference => cqss,
                            :hydrostatic_reference => hydro,
                            :precipitation => ice, :vertical_mixing => false,
                            :condensate_transform => tr, :rain_transform => tr)
    vtr === nothing || (opts[:vapor_transform] = vtr)
    if ice
        opts[:rain_moments] = 2
        opts[:rain_number_transform] = tr
        opts[:ice_microphysics] = :ishmael
        opts[:ice_transform] = tr
    end
    if louis
        opts[:louis_bl] = true
        opts[:surface_fluxes] = sfc
    end
    merge!(opts, extra)
    varnames = Scythe.mc_var_names(opts)
    vars = Dict(v => i for (i, v) in enumerate(varnames))
    side_bc = Dict{String,Any}(v => PeriodicBC() for v in varnames)
    wall_bc = merge(Dict{String,Any}(v => NeumannBC() for v in varnames),
                    Dict{String,Any}("w" => DirichletBC()))
    gp = GridParameters(geometry = "RiRk",
        iMin = 0.0, iMax = SPAN, num_cells_i = NCELL,
        kMin = 0.0, kMax = ZTOP, num_cells_k = NCELL_K,
        BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
    ref_file = joinpath(tmpdir, "vt_$(tag).ref")
    pp = merge(Dict{Symbol,Float64}(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                                    :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                    :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3,
                                    :f => 0.0, :N_r => 1.0e-3, :SST => 302.0,
                                    :Cd => 1.2e-3, :Ck => 1.2e-3, :U_min => 1.0,
                                    :l_inf => 80.0), pp_extra)
    model = ModelParameters(ts = ts, integration_time = 10.0, output_interval = 10.0,
        equation_set = "moist_compressible_XZ", ref_state_file = ref_file,
        grid_params = gp, physical_params = pp, options = opts,
        output_dir = output_dir, initial_conditions = initial_conditions)
    gp = model.grid_params
    patch = createGrid(gp)
    pts = Scythe.getGridpoints(patch)
    z = pts[1:gp.kDim, end]
    col = vt_column(z, rh)
    Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
    patch.physical .= 0.0
    seed!(patch, pts, gp.vars, col)
    spectralTransform!(patch)
    gridTransform!(patch)
    hrm = sparse(Int64[], Int64[], Float64[], size(patch.spectral, 1),
                 size(patch.spectral, 2))
    mtile = createModelTile(patch, patch, model, hrm)
    return mtile, patch, model
end

vt_ncols(patch, model) = div(size(patch.physical, 1), model.grid_params.kDim)

"""Advance `nsteps` full steps (every column, then the refit), starting at step `t0`."""
function vt_run!(m, p, mo, nsteps; t0 = 1)
    for t in t0:(t0 + nsteps - 1)
        for c in 1:vt_ncols(p, mo)
            Scythe.advance_column(m, c, t)
        end
        Scythe.calcTendency(m)
        gridTransform!(p)
    end
end

"""A kinked cloud sheet 200-1150 m of amplitude `amp` (its top sharper than one 500 m cell,
so the refit rings the cloud slot below zero above it), added at FIXED T: rho_t and E_t carry
the cloud's mass and BF02 energy. A vapor bump `vamp` at 3 km (rho_t carries it, E_t its
C_vv T + g z) and a smooth w `wamp`."""
vt_kink(z; zb = 200.0, zt = 1150.0, amp = 1.0e-3) =
    max(0.0, amp * (z - zb) * (zt - z) / ((zt - zb) / 2)^2)

function vt_seed!(patch, pts, V, col; amp = 1.0e-3, wamp = 0.5, vamp = 1.0e-3)
    rc_i = Scythe.mc_slot(V, "rho_c")
    kDim = length(col.Tk)
    for i in 1:size(patch.physical, 1)
        z = pts[i, end]; k = mod1(i, kDim)
        rc = vt_kink(z; amp)
        T = col.Tk[k]
        dv = vamp * exp(-((z - 3000.0) / 1200.0)^2)
        patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rc, 0.0, :bhyp, MU)
        patch.physical[i, V["rho_t"], 1] = rc + dv
        patch.physical[i, V["E_t"], 1] = rc * ((Cpv * T) - L_v(T) + (gravity * z)) +
                                          dv * ((Scythe.Cvv * T) + (gravity * z))
        patch.physical[i, V["w"], 1] = wamp * sin(2pi * pts[i, 1] / SPAN) * sin(pi * z / ZTOP)
    end
    return nothing
end

"""[`vt_seed!`](@ref) plus a PROGNOSTIC vapor bump `vslot` at 5 km, raised in rho_t and E_t too
(so no gap there), written into the vapor slot through `Scythe.vapor_slot` under the tile's
own transform (`vtr`, `vmu`) -- the density perturbation under `:none`."""
function vt_seed_vapor!(patch, pts, V, col; vslot = 5.0e-4, vtr = :none, vmu = 1.0e-9,
                        rho_vbar = nothing, kw...)
    vt_seed!(patch, pts, V, col; kw...)
    rv_i = Scythe.mc_slot(V, "rho_v")
    kDim = length(col.Tk)
    for i in 1:size(patch.physical, 1)
        z = pts[i, end]; k = mod1(i, kDim)
        T = col.Tk[k]
        dv = vslot * exp(-((z - 5000.0) / 1000.0)^2)
        if vtr === :none
            patch.physical[i, rv_i, 1] = dv
        else
            rvb = rho_vbar[k]
            patch.physical[i, rv_i, 1] = Scythe.bhyp(rvb + dv, vmu) - Scythe.bhyp(rvb, vmu)
        end
        patch.physical[i, V["rho_t"], 1] += dv
        patch.physical[i, V["E_t"], 1] += dv * ((Scythe.Cvv * T) + (gravity * z))
    end
    return nothing
end

"Order-sensitive exact checksum of a Float64 array (bit patterns, position-weighted)."
vt_checksum(A) = sum(UInt128(reinterpret(UInt64, x)) * UInt128(j) for (j, x) in enumerate(vec(A)))

"""
    s4_perm(vars) -> Vector{Int}

The slot columns of a state array in the STAGE-4 LAYOUT of the rho_t retirement (slot 3 deleted,
the vapor -- `"rho_v"` or `"nu_v"` -- moved into its position, every later slot one further in):
`p, rho_d, <vapor>, u, w, E_t, Q_ss, ...` in the order the remaining names have in `vars`. On a
layout that already has no `"rho_t"` and the vapor at 3 it is the identity, so a checksum taken
through it means the same thing on both sides of Stage 4: the goldens re-seeded at bf577b2 (slot
3 the derived anchor's passive shadow, which no reader takes) compare against the Stage-4 state
bit for bit.
"""
function s4_perm(vars)
    names = first.(sort(collect(vars), by = last))
    vname = haskey(vars, "nu_v") ? "nu_v" : "rho_v"
    rest = filter(n -> n != "rho_t" && n != vname, names)
    return [vars[n] for n in [rest[1:2]; vname; rest[3:end]]]
end

"""[`vt_checksum`](@ref) of a slot matrix (`var_np1`) or a patch's `physical` array, read in the
Stage-4 layout ([`s4_perm`](@ref)): the dropped slot 3 is not part of the sum."""
s4_checksum(A::AbstractMatrix, vars) = vt_checksum(A[:, s4_perm(vars)])
s4_checksum(A::AbstractArray{<:Any,3}, vars) = vt_checksum(A[:, s4_perm(vars), :])

end # module
