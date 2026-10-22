# test/test_rho_t_conversion.jl -- the pre-Stage-4 checkpoint conversion (Stage 4 of the rho_t
# retirement deleted slot 3, the prognostic total density "rho_t", and moved the vapor into it;
# src/checkpoint_conversion.jl, tools/convert_rho_t_checkpoint.jl).
#
#   * the round trip: a Stage-4 state -> `to_pre_stage4_grid` (slot 3 re-inserted, the vapor
#     appended) -> `convert_rho_t_grid` is the state again, bit for bit, every column and every
#     name-keyed grid parameter;
#   * the derive path (no vapor slot, pre-Stage A): the vapor rebuilt from slot 3 is the one
#     that was removed, to the rounding of the water sum;
#   * the .jld2 file path and its refusals; the loader refuses a pre-Stage-4 checkpoint naming
#     the tool and its invocation, and loads the converted one;
#   * the CSV path (every value's text unchanged) and the CSV reader's handling of a rho_t
#     column.
#
# Standalone:
#   julia --project=. -e 'using Test, Scythe; include("test/test_rho_t_conversion.jl")'

using Test
using Scythe
using Springsteel

isdefined(Main, :VaporTransformFixtures) || include(joinpath(@__DIR__, "vapor_transform_fixtures.jl"))

@testset "pre-Stage-4 checkpoint conversion (tools/convert_rho_t_checkpoint.jl)" begin
    F = Main.VaporTransformFixtures
    errmsg(f) = try
        f(); ""
    catch e
        sprint(showerror, e)
    end
    keyed = (:BCL, :BCR, :BCU, :BCD, :BCB, :BCT, :l_q, :max_wavenumber, :fourier_filter,
             :chebyshev_filter, :spline_filter, :positivity)
    with_ic(mo, path) = Scythe.ModelParameters(; (n => getfield(mo, n)
                                                  for n in fieldnames(Scythe.ModelParameters))...,
                                               initial_conditions = path)

    @testset "the layout map by name" begin
        # XZ: the vapor was appended at 10; cylinders: at 11, after "v"
        xz_old = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "rho_c", "rho_v", "nu_nr"]
        cy_old = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "rho_c", "v", "nu_v", "nu_nr"]
        for (old, cyl, vn) in ((xz_old, false, "rho_v"), (cy_old, true, "nu_v"))
            names, v = Scythe.stage4_var_names(Dict(n => i for (i, n) in enumerate(old)))
            @test v == vn
            @test names[3] == vn && !("rho_t" in names)
            @test names == filter(!=("rho_t"), [old[1:2]; vn; filter(!=(vn), old[3:end])])
            # the Stage-4 library layout itself
            opts = Dict{Symbol,Any}(:vapor_transform => vn == "nu_v" ? :bhyp : :none,
                                    :rain_moments => 2, :rain_number_transform => :bhyp)
            @test Scythe.mc_var_names(opts; cyl)[1:length(names)] == names
        end
        @test Scythe.pre_stage4_layout(Dict("rho_t" => 3))
        @test !Scythe.pre_stage4_layout(Dict("rho_v" => 3))
        @test Scythe.RHO_T_CONVERTER == "tools/convert_rho_t_checkpoint.jl"
        @test isfile(joinpath(@__DIR__, "..", Scythe.RHO_T_CONVERTER))
    end

    mktempdir() do tmp
        for (tag, vtr, ice) in (("bhyp_ice", :bhyp, true), ("none_warm", :none, false))
            m, p, mo = F.vt_tile(tmp, "cv_$(tag)"; vtr, ice,
                                 seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
            F.vt_run!(m, p, mo, 2)
            V = p.params.vars
            vn = vtr === :bhyp ? "nu_v" : "rho_v"

            @testset "round trip ($(tag)): new -> pre-Stage-4 -> new is bitwise" begin
                old = Scythe.to_pre_stage4_grid(p, mo)
                @test Scythe.pre_stage4_layout(old.params.vars)
                @test old.params.vars["rho_t"] == 3 && old.params.vars[vn] == 10
                # slots 1-2 and 4-9 never moved; the ones after the old vapor moved one in
                @test all(old.params.vars[n] == (n == vn ? 10 : (i <= 9 ? i : i + 1))
                          for (n, i) in V)
                # the re-inserted slot 3 is the derived total rho_d + rho_w - rho_tbar
                @test any(!=(0.0), old.physical[:, 3, 1])
                new = Scythe.convert_rho_t_grid(old; report = false)
                @test new.params.vars == V
                @test new.spectral == p.spectral
                @test new.physical == p.physical
                for f in keyed
                    hasproperty(p.params, f) || continue
                    @test getproperty(new.params, f) == getproperty(p.params, f)
                end
                for f in propertynames(p.params)
                    f in keyed || f === :vars || @test isequal(getproperty(new.params, f),
                                                               getproperty(p.params, f))
                end
                # a Stage-4 grid is not converted twice
                @test occursin("already in the Stage-4 layout",
                               errmsg(() -> Scythe.convert_rho_t_grid(new; report = false)))
                @test occursin("already pre-Stage-4", errmsg(() -> Scythe.to_pre_stage4_grid(old, mo)))
            end

            @testset "derive path ($(tag)): no vapor slot, the vapor rebuilt from slot 3" begin
                old = Scythe.to_pre_stage4_grid(p, mo)
                ov = old.params.vars
                ivo = ov[vn]
                names = [n for n in first.(sort(collect(ov), by = last)) if n != vn]
                vars = Dict{String,Int64}(n => i for (i, n) in enumerate(names))
                kw = Dict{Symbol,Any}(n => getproperty(old.params, n) for n in propertynames(old.params))
                kw[:vars] = vars
                for f in keyed
                    haskey(kw, f) || continue
                    d = copy(kw[f])
                    for sfx in ("", "_k", "_j")
                        haskey(d, vn * sfx) && delete!(d, vn * sfx)
                    end
                    kw[f] = d
                end
                nov = createGrid(Springsteel.SpringsteelGridParameters(; kw...))
                for (n, i) in vars
                    nov.spectral[:, i] .= view(old.spectral, :, ov[n])
                    nov.physical[:, i, :] .= view(old.physical, :, ov[n], :)
                end
                @test occursin("pass `model`",
                               errmsg(() -> Scythe.convert_rho_t_grid(nov; report = false)))
                new = Scythe.convert_rho_t_grid(nov; model = mo, report = false)
                iv = V[vn]
                @test new.params.vars == V
                # every other variable bit for bit
                for (n, i) in V
                    i == iv && continue
                    @test new.spectral[:, i] == p.spectral[:, i]
                    @test new.physical[:, i, :] == p.physical[:, i, :]
                end
                # the vapor: the same fit of the same gridpoint values, to the rounding of
                # rho_t - rho_d - sum_{k != v} alpha_k nu_k
                q = deepcopy(p); spectralTransform!(q)
                sc = maximum(abs, q.spectral[:, iv])
                @test sc > 0.0
                @test maximum(abs, new.spectral[:, iv] .- q.spectral[:, iv]) <= 1.0e-9 * sc
                gridTransform!(q)
                @test maximum(abs, new.physical[:, iv, 1] .- q.physical[:, iv, 1]) <=
                      1.0e-9 * maximum(abs, p.physical[:, iv, 1])
            end

            @testset ".jld2 files and the loader ($(tag))" begin
                old = Scythe.to_pre_stage4_grid(p, mo)
                src = joinpath(tmp, "old_$(tag).jld2")
                Springsteel.save_grid(src, old)
                # the loader refuses it, naming the tool and how to run it
                msg = errmsg(() -> Scythe.load_initial_conditions!(createGrid(mo.grid_params),
                                                                   with_ic(mo, src)))
                @test occursin("PRE-STAGE-4", msg)
                @test occursin(Scythe.RHO_T_CONVERTER, msg)
                @test occursin(Scythe.rho_t_converter_invocation(src), msg)
                dst = joinpath(tmp, "conv_$(tag)", "new.jld2")
                @test Scythe.convert_rho_t_checkpoint(src, dst) == dst
                g = load_grid(dst)
                @test g.params.vars == V && g.spectral == p.spectral && g.physical == p.physical
                # never overwritten; a missing source is named
                @test occursin("exists", errmsg(() -> Scythe.convert_rho_t_checkpoint(src, dst)))
                @test occursin("does not exist",
                               errmsg(() -> Scythe.convert_rho_t_checkpoint(src * ".x", dst * ".y")))
                # ...and the converted file loads as the state it was converted from
                lp = Scythe.load_initial_conditions!(createGrid(mo.grid_params), with_ic(mo, dst))
                @test lp.spectral == p.spectral
                @test lp.physical == p.physical
            end
        end

        @testset "CSV: the rho_t column dropped, the vapor moved, every value's text kept" begin
            hdr = "r,z,p,rho_d,rho_t,u,w,E_t,Q_ss,rho_r,rho_c,rho_v,nu_nr"
            rows = ["0.0,125.0,1.5,-2.0e-5,3.25e-4,0.0,1.0e-17,12.5,0.0,0.0,1.0e-6,-7.5e-9,2.0",
                    "1000.0,375.0,-0.0,1.0e-300,NaN,5.0,0.0,1.0,2.0,3.0,4.0,5.5e-5,0.0"]
            src = joinpath(tmp, "old_ics.csv")
            write(src, join([hdr; rows], "\n") * "\n")
            dst = joinpath(tmp, "new_ics.csv")
            Scythe.convert_rho_t_csv(src, dst)
            want = ["r,z,p,rho_d,rho_v,u,w,E_t,Q_ss,rho_r,rho_c,nu_nr",
                    "0.0,125.0,1.5,-2.0e-5,-7.5e-9,0.0,1.0e-17,12.5,0.0,0.0,1.0e-6,2.0",
                    "1000.0,375.0,-0.0,1.0e-300,5.5e-5,5.0,0.0,1.0,2.0,3.0,4.0,0.0"]
            @test readlines(dst) == want
            @test occursin("exists", errmsg(() -> Scythe.convert_rho_t_csv(src, dst)))
            @test occursin("no rho_t column", errmsg(() -> Scythe.convert_rho_t_csv(dst, dst * ".2")))
            nov = joinpath(tmp, "nov_ics.csv")
            write(nov, "r,z,p,rho_d,rho_t,u\n0.0,0.0,1.0,2.0,3.0,4.0\n")
            @test occursin("no vapor column", errmsg(() -> Scythe.convert_rho_t_csv(nov, nov * ".2")))
            # the reader: a rho_t column with the vapor is read by name (rho_t not read, a
            # note printed); without a vapor column it is refused, naming the tool
            V4 = Dict("p" => 1, "rho_d" => 2, "rho_v" => 3)
            out = mktemp() do path, io
                redirect_stdout(io) do
                    Scythe._check_csv_layout(src, V4)
                end
                flush(io); read(path, String)
            end
            @test occursin("not read", out)
            @test Scythe._check_csv_layout(dst, V4) === nothing
            m2 = errmsg(() -> Scythe._check_csv_layout(nov, V4))
            @test occursin("PRE-STAGE-4", m2) && occursin(Scythe.RHO_T_CONVERTER, m2)
        end
    end
end
