#!/usr/bin/env julia
# tools/convert_ishmael_tables.jl
#
# One-time converter: reads the ISHMAEL ice-microphysics lookup tables from
# the CM1 Fortran binaries and writes a single compressed JLD2 file for the
# Julia port (see src/ishmael_tables.jl).
#
# Fortran source of truth:
#   /Users/mmbell/Development/cm1r21.1/src/module_mp_jensen_ishmael.F
#   subroutine jensen_ishmael_init, lines 86-232 (read loops: lines 132-187)
#
# ishmael-gamma-tab.bin is deliberately DROPPED: the port uses
# SpecialFunctions.gamma in place of the tabulated Cody GAMMA the Fortran
# used for speed, so there is nothing to convert for it.
#
# ── Record structure (Fortran SEQUENTIAL UNFORMATTED, big-endian, 4-byte REAL) ──
#
# Both files are opened with FORM='unformatted', convert='big_endian'
# (lines 137, 157). Each iteration of the innermost read loop issues ONE
# Fortran READ statement that pulls ALL of the fifth-dimension values for
# that (i1,i2,i3,i4) at once -- i.e. one SEQUENTIAL UNFORMATTED record per
# (i1,i2,i3,i4), holding the fifth-dimension slice contiguously:
#
#   itab  (ishmael-qi-qc.bin, lines 138-147): do i1=1:51, i2=1:51, i3=1:51,
#     i4=1:11 -- read(20) itab_o(i1,i2,i3,i4,1), itab_o(i1,i2,i3,i4,2)
#     => 2 Float32 values per record.
#   itabr (ishmael-qi-qr.bin, lines 158-171): same loop nest, 6 values
#     (dims 1..6) per record.
#
# i4 is the fastest-varying loop index (innermost), then i3, then i2, then
# i1 (outermost) -- so records appear in that nesting order in the file.
# Each gfortran sequential-unformatted record is framed by a leading and a
# trailing 4-byte record-length marker (payload byte count), both of which
# we validate on every record read (the markers double as a built-in check
# that our assumed record shape is right).
#
#   record count            = 51*51*51*11 = 1,459,161
#   itab  payload/record    = 2 x Float32 = 8 bytes;  framed = 4+8+4  = 16 bytes
#   itabr payload/record    = 6 x Float32 = 24 bytes; framed = 4+24+4 = 32 bytes
#
#   1,459,161 * 16 = 23,346,576 bytes == filesize(ishmael-qi-qc.bin) exactly
#   1,459,161 * 32 = 46,693,152 bytes == filesize(ishmael-qi-qr.bin) exactly
#
# (Both byte-count identities were confirmed against the actual files before
# writing this reader, which is the real validation that the assumed shape
# -- "one record per (i1,i2,i3,i4) tuple, not one record per scalar" -- is
# correct; the per-record marker checks below are the ongoing guard.)
#
# Usage:
#   julia --project=. tools/convert_ishmael_tables.jl [conversion_date_string]

using JLD2
using SHA
using Printf

const QI_QC_PATH = "/Users/mmbell/Development/cm1r21.1/run/ishmael-qi-qc.bin"
const QI_QR_PATH = "/Users/mmbell/Development/cm1r21.1/run/ishmael-qi-qr.bin"
const OUT_PATH   = normpath(joinpath(@__DIR__, "..", "data", "ishmael_tables.jld2"))

"""
    read_be_record!(io, buf::Vector{Float32}) -> Vector{Float32}

Read one Fortran SEQUENTIAL UNFORMATTED big-endian record into `buf` (length
= number of Float32 payload values). Validates that the leading and
trailing 4-byte record-length markers both equal the expected payload byte
count, erroring loudly on any mismatch -- an assumed record shape that is
wrong (wrong element count, wrong endianness, wrong marker width) will
almost certainly desync the markers on the very first record.
"""
function read_be_record!(io::IO, buf::Vector{Float32})
    nbytes = 4 * length(buf)
    pos0 = position(io)
    marker1 = ntoh(read(io, Int32))
    marker1 == nbytes || error(
        "leading record marker $marker1 bytes != expected payload $nbytes bytes " *
        "(record starting at byte $pos0) -- assumed record shape is wrong")
    @inbounds for i in eachindex(buf)
        buf[i] = ntoh(read(io, Float32))
    end
    marker2 = ntoh(read(io, Int32))
    marker2 == nbytes || error(
        "trailing record marker $marker2 bytes != expected payload $nbytes bytes " *
        "(record starting at byte $pos0, leading marker was $marker1) -- " *
        "assumed record shape is wrong")
    return buf
end

"""
    read_itab_family(path, nfifth) -> Array{Float32,5}

Read a `(51,51,51,11,nfifth)` ISHMAEL collection table from `path`, one
record per `(i1,i2,i3,i4)` holding `nfifth` big-endian Float32 values
(fifth-dimension slice). The array is kept in `Float32` — the storage
precision of the source data — so the JLD2 stores no fabricated bits;
the loader widens to `Float64` at load time. Validates that the number
of bytes consumed equals the file size exactly.
"""
function read_itab_family(path::String, nfifth::Int)
    sz = filesize(path)
    framed_record_bytes = 8 + 4 * nfifth   # 4 (leading marker) + payload + 4 (trailing marker)
    nrecords_expected = 51 * 51 * 51 * 11
    expected = nrecords_expected * framed_record_bytes
    expected == sz || error(
        "$path: file size $sz bytes != expected $expected bytes for " *
        "nfifth=$nfifth (record shape assumption is wrong)")

    arr = Array{Float32}(undef, 51, 51, 51, 11, nfifth)
    buf = Vector{Float32}(undef, nfifth)
    nread = 0
    open(path, "r") do io
        for i1 in 1:51, i2 in 1:51, i3 in 1:51, i4 in 1:11
            read_be_record!(io, buf)
            @inbounds for k in 1:nfifth
                arr[i1, i2, i3, i4, k] = buf[k]
            end
            nread += 1
        end
        eof(io) || error("$path: unconsumed bytes remain after reading all $nread records")
    end
    consumed = nread * framed_record_bytes
    consumed == sz || error("$path: consumed $consumed bytes but file is $sz bytes")
    nread == nrecords_expected || error("$path: read $nread records, expected $nrecords_expected")
    return arr
end

function main()
    conversion_date = length(ARGS) >= 1 ? ARGS[1] : "2026-08-12"

    println("Reading itab (ice-cloud collection) from $QI_QC_PATH ...")
    itab = read_itab_family(QI_QC_PATH, 2)
    println("Reading itabr (ice-rain collection) from $QI_QR_PATH ...")
    itabr = read_itab_family(QI_QR_PATH, 6)

    all(isfinite, itab)  || error("itab contains non-finite values after conversion")
    all(isfinite, itabr) || error("itabr contains non-finite values after conversion")

    println("Hashing source binaries ...")
    sha_qc = bytes2hex(open(sha256, QI_QC_PATH))
    sha_qr = bytes2hex(open(sha256, QI_QR_PATH))

    mkpath(dirname(OUT_PATH))

    provenance = Dict{String,Any}(
        "qi_qc_source"     => QI_QC_PATH,
        "qi_qc_bytes"      => filesize(QI_QC_PATH),
        "qi_qc_sha256"     => sha_qc,
        "qi_qr_source"     => QI_QR_PATH,
        "qi_qr_bytes"      => filesize(QI_QR_PATH),
        "qi_qr_sha256"     => sha_qr,
        "gamma_table_note" => "ishmael-gamma-tab.bin deliberately dropped; " *
                               "port uses SpecialFunctions.gamma instead",
        "conversion_date"  => conversion_date,
        "fortran_source"   => "module_mp_jensen_ishmael.F, jensen_ishmael_init lines 86-232",
        "index_order"      => "itab/itabr[i1,i2,i3,i4,k]: i1,i2,i3 in 1:51, i4 in 1:11, " *
                               "k in 1:2 (itab) or 1:6 (itabr); matches the Fortran loop " *
                               "nest do i1=1:51, i2=1:51, i3=1:51, i4=1:11 with dim5 (k) " *
                               "read together in one record",
    )

    println("Writing $OUT_PATH ...")
    jldsave(OUT_PATH; compress = true,
        itab = itab, itabr = itabr, provenance = provenance)

    outsize = filesize(OUT_PATH)
    @printf("Wrote %s (%.3f MB)\n", OUT_PATH, outsize / 1024^2)
    if outsize > 20 * 1024^2
        println("WARNING: output JLD2 exceeds 20 MB -- repo-size concern if ever committed.")
    end

    println()
    println("Spot values -- cross-check these against the Fortran itab/itabr arrays:")
    for (i1, i2, i3, i4, k) in ((1, 1, 1, 1, 1), (26, 26, 26, 6, 1), (51, 51, 51, 11, 2))
        @printf("  itab[%d,%d,%d,%d,%d]  = %.10g\n", i1, i2, i3, i4, k, itab[i1, i2, i3, i4, k])
    end
    # Same index pattern as requested, applied to itabr (note itabr's 5th dim
    # only reaches 6, not 2 as its natural upper bound -- shown separately below).
    for (i1, i2, i3, i4, k) in ((1, 1, 1, 1, 1), (26, 26, 26, 6, 1), (51, 51, 51, 11, 2))
        @printf("  itabr[%d,%d,%d,%d,%d] = %.10g\n", i1, i2, i3, i4, k, itabr[i1, i2, i3, i4, k])
    end
    @printf("  itabr[51,51,51,11,6] = %.10g   (itabr's true final corner, dim5 up to 6)\n",
        itabr[51, 51, 51, 11, 6])

    return nothing
end

main()
