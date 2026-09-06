# ── Reader for the MYNN-EDMF Fortran reference harness ────────────────────────
#
# Parses the artefacts of tools/mynn_fortran_driver/ so the parity tests
# (test/test_mynn_closure.jl, later test/test_mynn_edmf.jl) can run with NO Fortran
# at test time:
#
#   * `ref_driver_output_r8.txt` — 5.9 MB / ~184k lines, the double-promoted reference
#     run of the verbatim ccpp-physics scheme. Blocks look like
#
#         ## <case> closure=<c> mode=<A|B> step=<s> <name> n=<len>
#         <len> lines of  "    k  <ES25.17E3 value>"
#
#     plus a `## constants n=24` block at the very top, `## <case> BEGIN n=<n>`,
#     `## <case> GATE ...` and `## <case> END` marker lines. Scalars are `n=1`;
#     integers (kpbl, ktop_plume) are printed as reals; case 1 carries genuine `NaN`
#     values (ust = 0, see the harness README item 4) which parse as `NaN` here.
#
#   * `columns/<case>.txt` — the frozen input columns.
#   * `columns/constants.txt` — the 14 host constants, written from
#     `Springsteel.Thermodynamics` by tools/mynn_dump_columns.jl.
#
# Everything is cached in module-level `Dict`s keyed by absolute path, so a test file
# that reads the reference many times pays the parse once. The full parse is ~1 s.
#
# This file defines plain functions and no module, so it can be `include`d from
# test/runtests.jl or from a standalone test script.

const MYNN_DRIVER_DIR = normpath(joinpath(@__DIR__, "..", "..", "tools", "mynn_fortran_driver"))
const MYNN_REF_FILE   = joinpath(MYNN_DRIVER_DIR, "ref_driver_output_r8.txt")
const MYNN_COLUMN_DIR = joinpath(MYNN_DRIVER_DIR, "columns")

"""
    MYNN_CASES

The five reference columns, in the order the driver runs them.
"""
const MYNN_CASES = ("case1_rest", "case2_o01_sea", "case3_tc_rmw",
                    "case4_convective", "case5_highwind")

"""
    MYNNRefKey

Block key inside one case: `(closure, mode, step)`, e.g. `(2.5, "B", 1)` for the mode-B
per-routine dump at step 1 of closure 2.5, or `(2.5, "B", 0)` for the initialization
block. Mode "A" carries only the end-of-step driver state (`closure` 2.5 and 2.6).
"""
const MYNNRefKey = Tuple{Float64,String,Int}

const MYNNCaseRefs = Dict{MYNNRefKey,Dict{String,Vector{Float64}}}

"""
    MYNNReference

What `mynn_reference()` returns.

* `constants` — the 24 values of the `## constants` block, in the driver's print order
  (which is `MYNN_CONSTANT_ORDER` of src/mynn_constants.jl).
* `blocks` — `blocks[case][(closure, mode, step)][name] :: Vector{Float64}`.
* `nlev` — `nlev[case]`, the column length from the `## <case> BEGIN n=` line.
* `gate` — `gate[case]`, `true` when the driver printed `GATE PASS` for that case, i.e.
  when its mode-B replay reproduced mode A bitwise at step 30. A test that leans on the
  mode-B intermediates should assert this.
"""
struct MYNNReference
    constants::Vector{Float64}
    blocks::Dict{String,MYNNCaseRefs}
    nlev::Dict{String,Int}
    gate::Dict{String,Bool}
end

const _MYNN_REF_CACHE = Dict{String,MYNNReference}()

# ES25.17E3 fields are fixed-width: 5 columns of index, one blank, 25 columns of value.
@inline function _mynn_value(line::AbstractString)
    return parse(Float64, SubString(line, 7))
end

"""
    mynn_parse_reference(path = MYNN_REF_FILE) -> MYNNReference

Parse a driver output file. Prefer `mynn_reference()`, which caches.
"""
function mynn_parse_reference(path::AbstractString = MYNN_REF_FILE)
    isfile(path) || error("MYNN reference output not found at $path; regenerate it " *
                          "with tools/mynn_fortran_driver/run.sh")
    constants = Float64[]
    blocks = Dict{String,MYNNCaseRefs}()
    nlev   = Dict{String,Int}()
    gate   = Dict{String,Bool}()

    current::Union{Nothing,Vector{Float64}} = nothing   # block being filled
    idx = 0                                             # values seen in it
    want = 0                                            # values expected

    open(path, "r") do io
        for line in eachline(io)
            if !isempty(line) && line[1] == '#'
                # ---- header / marker line ----
                current = nothing
                tok = split(line)
                if length(tok) == 3 && tok[2] == "constants"
                    want = parse(Int, SubString(tok[3], 3))
                    constants = Vector{Float64}(undef, want)
                    current = constants
                    idx = 0
                elseif length(tok) == 4 && tok[3] == "BEGIN"
                    nlev[tok[2]] = parse(Int, SubString(tok[4], 3))
                    get!(blocks, tok[2], MYNNCaseRefs())
                elseif length(tok) >= 4 && tok[3] == "GATE"
                    if tok[4] == "PASS"
                        gate[tok[2]] = true
                    elseif tok[4] == "FAIL"
                        gate[tok[2]] = false
                    end
                elseif length(tok) == 7 && startswith(tok[3], "closure=")
                    cas     = tok[2]
                    closure = parse(Float64, SubString(tok[3], 9))
                    mode    = String(SubString(tok[4], 6))
                    step    = parse(Int, SubString(tok[5], 6))
                    name    = String(tok[6])
                    want    = parse(Int, SubString(tok[7], 3))
                    v = Vector{Float64}(undef, want)
                    caseref = get!(blocks, cas, MYNNCaseRefs())
                    key = get!(caseref, (closure, mode, step),
                               Dict{String,Vector{Float64}}())
                    key[name] = v
                    current = v
                    idx = 0
                end
                # "## <case> END" and anything else: no block follows
            elseif current !== nothing
                idx += 1
                idx <= want || error("MYNN reference: too many values in a block near '$line'")
                @inbounds current[idx] = _mynn_value(line)
            end
        end
    end
    return MYNNReference(constants, blocks, nlev, gate)
end

"""
    mynn_reference(path = MYNN_REF_FILE) -> MYNNReference

Cached `mynn_parse_reference`.
"""
function mynn_reference(path::AbstractString = MYNN_REF_FILE)
    return get!(() -> mynn_parse_reference(path), _MYNN_REF_CACHE, abspath(path))
end

"""
    mynn_block(ref, case, closure, mode, step, name) -> Vector{Float64}

One reference block, with a message that says what is missing rather than a `KeyError`.
"""
function mynn_block(ref::MYNNReference, case::AbstractString, closure::Real,
                    mode::AbstractString, step::Integer, name::AbstractString)
    caseref = get(ref.blocks, String(case)) do
        error("MYNN reference has no case '$case'")
    end
    key = (Float64(closure), String(mode), Int(step))
    blk = get(caseref, key) do
        error("MYNN reference has no block group $key for case '$case'")
    end
    return get(blk, String(name)) do
        error("MYNN reference has no '$name' in $key of case '$case'; " *
              "available: $(join(sort!(collect(keys(blk))), ", "))")
    end
end

"""
    mynn_scalar(ref, case, closure, mode, step, name) -> Float64

`mynn_block` for an `n=1` block.
"""
function mynn_scalar(ref::MYNNReference, case::AbstractString, closure::Real,
                     mode::AbstractString, step::Integer, name::AbstractString)
    v = mynn_block(ref, case, closure, mode, step, name)
    length(v) == 1 || error("MYNN reference block '$name' of case '$case' has " *
                            "$(length(v)) values, expected a scalar")
    return v[1]
end

# ── Column inputs ─────────────────────────────────────────────────────────────

const _MYNN_COLUMN_CACHE = Dict{String,NamedTuple}()

"""
    mynn_column(case) -> NamedTuple

Read `columns/<case>.txt`, the frozen input column, in the format documented in
tools/mynn_fortran_driver/README.md:

    line 1        n
    line 2        ps ts qsfc ust hfx qfx wspd znt xland dx rmol delt
    lines 3..n+2  z dz u v w T th exner p rho sqv sqc sqi

Returns `(; n, ps, ts, qsfc, ust, hfx, qfx, wspd, znt, xland, dx, rmol, delt,
           z, dz, u, v, w, T, th, exner, p, rho, sqv, sqc, sqi)`.

`sqv/sqc/sqi` are SPECIFIC contents (`rho_x/rho_t`) and `rho` is the moist density —
the `mynnedmf_wrapper` conventions — and `ts` is `T_sfc/exner(1)`. `rmol` here is the
column's initial value, which the driver uses as `rmol0` before recomputing it each
step.
"""
function mynn_column(case::AbstractString)
    path = joinpath(MYNN_COLUMN_DIR, String(case) * ".txt")
    return get!(_MYNN_COLUMN_CACHE, path) do
        isfile(path) || error("MYNN column file not found at $path")
        vals = Float64[]
        n = 0
        open(path, "r") do io
            n = parse(Int, strip(readline(io)))
            for tokn in eachsplit(readline(io))
                push!(vals, parse(Float64, tokn))
            end
            for _ in 1:n, tokn in eachsplit(readline(io))
                push!(vals, parse(Float64, tokn))
            end
        end
        length(vals) == 12 + 13*n ||
            error("MYNN column $path: expected $(12 + 13n) numbers, got $(length(vals))")
        hdr = @view vals[1:12]
        # the per-level records were pushed row-major, 13 fields per level
        col(j) = Float64[vals[12 + (k-1)*13 + j] for k in 1:n]
        return (; n,
                ps = hdr[1], ts = hdr[2], qsfc = hdr[3], ust = hdr[4], hfx = hdr[5],
                qfx = hdr[6], wspd = hdr[7], znt = hdr[8], xland = hdr[9],
                dx = hdr[10], rmol = hdr[11], delt = hdr[12],
                z = col(1), dz = col(2), u = col(3), v = col(4), w = col(5),
                T = col(6), th = col(7), exner = col(8), p = col(9), rho = col(10),
                sqv = col(11), sqc = col(12), sqi = col(13))
    end
end

"""
    mynn_host_constants() -> Vector{Float64}

The 14 numbers of `columns/constants.txt`, in the order
`cp cpv cliq cice p608 ep_2 grav karman t0c rcp r_d r_v xlf xlv`.
"""
function mynn_host_constants()
    path = joinpath(MYNN_COLUMN_DIR, "constants.txt")
    isfile(path) || error("MYNN constants file not found at $path")
    return parse.(Float64, split(read(path, String)))
end

"""
    mynn_face_heights(dz) -> Vector{Float64}

The `zw` array the driver builds (ref_driver.f90 mode_b, mirroring mynn_bl_driver
:1002-1008): `zw(1) = 0`, `zw(k) = zw(k-1) + dz(k-1)` up to `zw(kte+1)`. Length `n+1`.
"""
function mynn_face_heights(dz::Vector{Float64})
    n = length(dz)
    zw = zeros(Float64, n + 1)
    for k in 2:n
        zw[k] = zw[k-1] + dz[k-1]
    end
    zw[n+1] = zw[n] + dz[n]
    return zw
end

"""
    mynn_conserved(colm, c) -> (thl, sqw, thetav)

The conserved variables the driver forms from a frozen column before every call
(ref_driver.f90 mode_b, mirroring mynn_bl_driver :1009-1013):

    sqw    = sqv + sqc + sqi
    thl    = th - xlvcp/exner*sqc - xlscp/exner*sqi
    thetav = th*(1 + p608*sqv)

`c` is a `MYNNConstants`. Snow is excluded from `sqw` — the driver passes a zero column
as `qs` (README item 5).
"""
function mynn_conserved(colm, c)
    n = colm.n
    thl = Vector{Float64}(undef, n)
    sqw = Vector{Float64}(undef, n)
    thetav = Vector{Float64}(undef, n)
    for k in 1:n
        sqw[k] = colm.sqv[k] + colm.sqc[k] + colm.sqi[k]
        thl[k] = colm.th[k] - c.xlvcp/colm.exner[k]*colm.sqc[k] -
                              c.xlscp/colm.exner[k]*colm.sqi[k]
        thetav[k] = colm.th[k]*(1.0 + c.p608*colm.sqv[k])
    end
    return (thl, sqw, thetav)
end

"""
    mynn_taper_qke(ust, zw, n) -> Vector{Float64}

The first-guess TKE the driver hands `GET_PBLH` in its init block (ref_driver.f90
mode_b, mirroring mynn_bl_driver's `INITIALIZE_QKE` pre-pass):

    qke(k) = 5*ust*max((ust*700 - zw(k))/(max(ust, 0.01)*700), 0.01)

Note the 5*ust prefactor, which is NOT the `1.5*ust^2*(b1*pmz)^(2/3)` that
`mym_initialize` then overwrites it with.
"""
function mynn_taper_qke(ust::Float64, zw::Vector{Float64}, n::Integer)
    qke = Vector{Float64}(undef, n)
    for k in 1:n
        qke[k] = 5.0*ust*max((ust*700.0 - zw[k])/(max(ust, 0.01)*700.0), 0.01)
    end
    return qke
end
