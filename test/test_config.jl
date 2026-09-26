# test/test_config.jl -- the typed run configuration (src/config.jl): defaults, the TOML
# round trip, overlays, --set typing, the refusals, path resolution, and the registry grep
# that keeps KNOWN_OPTION_KEYS / KNOWN_PHYSICAL_KEYS in step with src/.

using Test
using TOML
using Scythe
using Scythe: RunSection, OutputSection, GridSection, BoundaryConditionsSection, TimestepSection, ReferenceStateSection,
    InitialConditions, VortexIC, NoiseIC, BubbleIC, FileIC, Environment, Microphysics,
    BoundaryLayer, Surface, Radiation, HorizontalMixing, Sponge, Physics, Numerics,
    Diagnostics, TimestepAdvisory, Extra, KNOWN_OPTION_KEYS, KNOWN_PHYSICAL_KEYS,
    OWNED_OPTION_KEYS, OWNED_PHYSICAL_KEYS, RETIRED_OPTION_KEYS, EQUATION_SETS,
    config_hash, to_dict, parse_set, MOVED_KEYS

# The message a call dies with ("" if it does not), so a refusal is checked for SAYING
# where the problem is, not just for throwing.
errmsg(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

writetoml(path, s) = (write(path, s); path)

@testset "config" begin

@testset "defaults" begin
    c = RunConfig()
    @test c == RunConfig()
    @test hash(c) == hash(RunConfig())
    @test c.run.equation_set == "moist_compressible_XZ"
    @test c.grid.geometry == "RiRk" && c.boundary_conditions.lateral === :periodic
    @test c.boundary_conditions.vertical_scalars === :d2
    @test c.boundary_conditions.sedimentation_lid === :natural
    @test isempty(c.boundary_conditions.override)
    @test c.grid.boundaries == [0.0, 300.0e3] && c.grid.num_cells == [50]
    @test c.physics.microphysics.ice === :ishmael
    @test c.physics.boundary_layer.scheme === :mynn && !c.physics.boundary_layer.mynn_edmf
    @test c.physics.boundary_layer.K_max == Inf
    @test c.physics.surface.stability && c.physics.surface.fluxes
    @test c.physics.radiation.solar === :diurnal
    @test c.numerics.state_dependent_si && !c.numerics.exact_si
    @test c.run.segment_length == 0.0
    # Paths are absolute in the struct; @scythe/ is the package root.
    @test isabspath(c.output.dir)
    @test c.reference_state.sounding ==
          joinpath(pkgdir(Scythe), "benchmarks", "reference_data", "o01_rainfall", "dunion_MT_hum90.ref")
    @test isfile(c.reference_state.sounding)
    # Default struct equality would compare Vector fields by identity.
    @test GridSection() == GridSection() && TimestepSection() == TimestepSection()
    # load_config with no files is the defaults.
    @test load_config() == c
end

@testset "round trip" begin
    base_phys = Physics(
        microphysics = Microphysics(ice = :none, rain_moments = 2, rain_transform = :bhyp_smooth,
                                    N_0 = 1.0e7),
        boundary_layer = BoundaryLayer(scheme = :louis, mynn_fidelity = "gtr_local,pdk1",
                                       mynn_init = :zero, K_max = 500.0),
        surface = Surface(roughness = :charnock, stability = false, Cd = 1.2e-3),
        radiation = Radiation(method = :clearsky, solar = :fixed, forcing = :anomaly),
        horizontal_mixing = HorizontalMixing(scheme = :smagorinsky, water = true),
        sponge = Sponge(alpha = 0.01))
    extra = Extra(
        options = Dict{Symbol,Any}(:radiation_level_interp => :bestfit, :ice_agg_caps => false,
                                   :sfc_stability_iterations => 5, :hsi_scheme => "am2",
                                   :radiation_gases => Dict{String,Float64}("co2" => 8.0e-4)),
        physical_params = Dict{Symbol,Float64}(:Kvdiff_heat => 10.0, :co2_ppm => 800.0))
    ics = Dict(
        :balanced_vortex    => InitialConditions(method = :balanced_vortex,
                                   balanced_vortex = VortexIC(profile = :rankine, v_m = 20.0,
                                                              moist_profile = :gaussian)),
        :temperature_noise  => InitialConditions(method = :temperature_noise,
                                   temperature_noise = NoiseIC(seed = 7, dT_max = 0.2)),
        :temperature_bubble => InitialConditions(method = :temperature_bubble,
                                   temperature_bubble = BubbleIC(xc = 1.0e3, dT_max = 2.0)),
        :file               => InitialConditions(method = :file,
                                   file = FileIC(paths = ["/abs/a.csv", "/abs/nest2_a.jld2"])))
    mktempdir() do dir
        for (m, ic) in ics
            cfg = RunConfig(
                run = RunSection(name = "rt_$m", equation_set = "moist_compressible_axisym",
                                 segment_length = 86400.0),
                output = OutputSection(dir = joinpath(dir, "out"), formats = [:netcdf, :csv],
                                       sidecars = true),
                grid = GridSection(boundaries = [0.0, 150.0e3, 450.0e3], num_cells = [15, 15]),
                boundary_conditions = BoundaryConditionsSection(lateral = :walls,
                    vertical_scalars = :r1t1x, sedimentation_lid = :wall,
                    override = Dict("nu_r" => Dict("top" => "dirichlet", "bottom" => "neumann"),
                                    "p" => Dict("outer" => "second_derivative"))),
                timestep = TimestepSection(ts = 0.25, ts_per_patch = [0.25, 0.5]),
                initial_conditions = ic,
                physics = base_phys,
                numerics = Numerics(wall_bc_tau = Inf, positivity = :off,
                                    state_dependent_si = false, exact_si = true),
                diagnostics = Diagnostics(stiffness_trace = 10,
                                          timestep_advisory = TimestepAdvisory(state_deviation = 0.15)),
                extra = extra)
            path = write_config(cfg, joinpath(dir, "cfg_$m.toml"); header = "provenance line 1\nline 2")
            text = read(path, String)
            @test startswith(text, "# provenance line 1\n# line 2\n")
            @test occursin("wall_bc_tau = +inf", text) || occursin("wall_bc_tau = inf", text)
            @test occursin("[initial_conditions.$m]", text)
            # Only the active IC sub-table is written.
            @test count(occursin(text), ["[initial_conditions.$k]" for k in keys(ics)]) == 1
            back = load_config(path)
            @test back == cfg
            @test hash(back) == hash(cfg)
            @test config_hash(back) == config_hash(cfg)
            @test back.numerics.wall_bc_tau == Inf
            @test back.extra.options[:radiation_level_interp] === :bestfit
            @test back.extra.options[:radiation_gases] == Dict("co2" => 8.0e-4)
            @test back.output.formats == [:netcdf, :csv]
            @test back.boundary_conditions.override ==
                  Dict("nu_r" => Dict("top" => "dirichlet", "bottom" => "neumann"),
                       "p" => Dict("outer" => "second_derivative"))
            @test back.boundary_conditions.vertical_scalars === :r1t1x
            @test occursin("[boundary_conditions.override.nu_r]", text)
        end
        # An inactive sub-table that differs from its default is NOT written back.
        cfg = RunConfig(initial_conditions = InitialConditions(method = :file,
                                                                temperature_noise = NoiseIC(seed = 1)))
        back = load_config(write_config(cfg, joinpath(dir, "inactive.toml")))
        @test back != cfg
        @test back == RunConfig(initial_conditions = InitialConditions(method = :file))
        # The canonical hash tracks content.
        @test config_hash(RunConfig()) != config_hash(RunConfig(timestep = TimestepSection(ts = 0.3)))
    end
end

@testset "overlay precedence" begin
    mktempdir() do dir
        a = writetoml(joinpath(dir, "a.toml"), """
            [run]
            name = "base"
            integration_time = 7200
            [grid]
            boundaries = [0.0, 150.0e3, 450.0e3]
            num_cells = [10, 10]
            [physics.radiation]
            method = "gray"
            interval = 600.0
            [initial_conditions.temperature_bubble]
            xc = 5.0e3
            zc = 700.0
            """)
        b = writetoml(joinpath(dir, "b.toml"), """
            [run]
            name = "overlay"
            [grid]
            boundaries = [0.0, 100.0e3]
            num_cells = [20]
            [physics.radiation]
            interval = 120.0
            [initial_conditions.temperature_bubble]
            zc = 900.0
            """)
        c = load_config(a, b)
        @test c.run.name == "overlay"                      # scalar replaced
        @test c.run.integration_time == 7200.0             # untouched scalar kept, Int -> Float64
        @test c.grid.boundaries == [0.0, 100.0e3]          # array replaced, not concatenated
        @test c.grid.num_cells == [20]
        @test c.physics.radiation.method === :gray         # nested table merged key by key
        @test c.physics.radiation.interval == 120.0
        @test c.initial_conditions.temperature_bubble.xc == 5.0e3
        @test c.initial_conditions.temperature_bubble.zc == 900.0
        # Order matters: the later file wins.
        @test load_config(b, a).run.name == "base"
        # --set beats every file.
        @test load_config(a, b; set = ["run.name=cli"]).run.name == "cli"
    end
end

@testset "--set typing" begin
    c = load_config(set = ["timestep.ts=0.25",                          # Float
                           "physics.boundary_layer.scheme=louis",       # bare Symbol enum
                           "physics.surface.stability=false",           # Bool
                           "grid.num_cells=[15,15,15]",                 # Vector{Int}
                           "grid.boundaries=[0, 1.5e5, 4.5e5, 1.05e6]", # Vector{Float64} from mixed
                           "physics.boundary_layer.K_max=inf",          # Inf
                           "numerics.wall_bc_tau=\"inf\"",              # quoted "inf"
                           "output.formats=[\"netcdf\",\"csv\"]",       # Vector{Symbol}
                           "physics.boundary_layer.mynn_fidelity=gtr_local,pdk1",
                           "extra.options.radiation_level_interp=bestfit",
                           "extra.physical_params.Kvdiff_heat=5"])
    @test c.timestep.ts === 0.25
    @test c.physics.boundary_layer.scheme === :louis
    @test c.physics.surface.stability === false
    @test c.grid.num_cells == [15, 15, 15] && eltype(c.grid.num_cells) === Int
    @test c.grid.boundaries == [0.0, 1.5e5, 4.5e5, 1.05e6] && eltype(c.grid.boundaries) === Float64
    @test c.physics.boundary_layer.K_max == Inf
    @test c.numerics.wall_bc_tau == Inf
    @test c.output.formats == [:netcdf, :csv]
    @test c.physics.boundary_layer.mynn_fidelity == "gtr_local,pdk1"
    @test c.extra.options[:radiation_level_interp] === :bestfit
    @test c.extra.physical_params[:Kvdiff_heat] === 5.0
    @test occursin("not of the form", errmsg(() -> parse_set("run.name")))
    @test parse_set("a.b=3") == Dict("a" => Dict("b" => 3))
end

@testset "refusals" begin
    mktempdir() do dir
        f(s) = writetoml(joinpath(dir, "bad.toml"), s)
        # Unknown keys, with the section named.
        m = errmsg(() -> load_config(f("[physics.radiation]\nmethd = \"gray\"\n")))
        @test occursin("methd", m) && occursin("physics.radiation", m)
        m = errmsg(() -> load_config(f("[physics.radiaton]\nmethod = \"gray\"\n")))
        @test occursin("radiaton", m) && occursin("[physics]", m)
        m = errmsg(() -> load_config(f("[initial_conditions.balanced_vortex]\nvmax = 3.0\n")))
        @test occursin("vmax", m) && occursin("initial_conditions.balanced_vortex", m)
        m = errmsg(() -> load_config(f("[runn]\nname = \"x\"\n")))
        @test occursin("runn", m) && occursin("top level", m)
        @test occursin("extra", errmsg(() -> load_config(set = ["extra.foo.x=1"])))
        # Bad enum values.
        m = errmsg(() -> load_config(set = ["physics.radiation.method=allskyy"]))
        @test occursin("physics.radiation.method", m) && occursin(":allsky", m)
        @test occursin("physics.surface.roughness", errmsg(() -> load_config(set = ["physics.surface.roughness=cod"])))
        @test occursin("output.formats", errmsg(() -> load_config(set = ["output.formats=[\"jld2\"]"])))
        @test occursin("rain_moments", errmsg(() -> load_config(set = ["physics.microphysics.rain_moments=3"])))
        @test occursin("mynn_fidelity", errmsg(() -> load_config(set = ["physics.boundary_layer.mynn_fidelity=bogus"])))
        # The same checks guard Julia construction.
        @test occursin("physics.radiation.method",
                       errmsg(() -> RunConfig(physics = Physics(radiation = Radiation(method = :nope)))))
        # Wrong value types.
        @test occursin("physics.sponge.alpha", errmsg(() -> load_config(set = ["physics.sponge.alpha=true"])))
        @test occursin("integer", errmsg(() -> load_config(set = ["grid.num_cells_k=50.5"])))
        @test occursin("true or false", errmsg(() -> load_config(set = ["numerics.exact_si=1"])))
        @test occursin("run.name", errmsg(() -> load_config(set = ["run.name=3"])))
        # extra.options: not in the registry.
        m = errmsg(() -> load_config(set = ["extra.options.mynn_intervall=10"]))
        @test occursin("extra.options.mynn_intervall", m) && occursin("KNOWN_OPTION_KEYS", m)
        # extra.options: owned by a typed section.
        m = errmsg(() -> load_config(set = ["extra.options.radiation_method=gray"]))
        @test occursin("physics.radiation.method", m)
        m = errmsg(() -> load_config(f("[extra.options]\nsfc_z0 = \"charnock\"\n")))
        @test occursin("physics.surface.roughness", m)
        # extra.options: retired key.
        @test occursin("retired", errmsg(() -> load_config(set = ["extra.options.vapor_retrieval=composite"])))
        # extra.options: wrong registered type.
        @test occursin("integer", errmsg(() -> load_config(set = ["extra.options.sfc_stability_iterations=2.5"])))
        # extra.physical_params: non-number, unknown, owned.
        m = errmsg(() -> load_config(f("[extra.physical_params]\nco2_ppm = \"lots\"\n")))
        @test occursin("co2_ppm", m) && occursin("number", m)
        @test occursin("number", errmsg(() -> load_config(set = ["extra.physical_params.co2_ppm=true"])))
        @test occursin("KNOWN_PHYSICAL_KEYS", errmsg(() -> load_config(set = ["extra.physical_params.Kvdif=1"])))
        @test occursin("physics.sponge.z_damp", errmsg(() -> load_config(set = ["extra.physical_params.z_damp=1"])))
        @test occursin("KNOWN_OPTION_KEYS",
                       errmsg(() -> RunConfig(extra = Extra(options = Dict{Symbol,Any}(:nope => 1)))))
        # Equation set / geometry.
        m = errmsg(() -> load_config(set = ["grid.geometry=RLR"]))
        @test occursin("grid.geometry", m) && occursin("moist_compressible_XZ", m)
        @test occursin("run.equation_set", errmsg(() -> load_config(set = ["run.equation_set=moist_compressible_3D"])))
        @test load_config(set = ["run.equation_set=moist_compressible_RLR", "grid.geometry=RLR"]).grid.geometry == "RLR"
        @test load_config(set = ["run.equation_set=moist_compressible_axisym", "grid.geometry=RZ"]).grid.geometry == "RZ"
        # [boundary_conditions]: the three moved keys are refused with the new location.
        @test MOVED_KEYS["grid.lateral"] == "boundary_conditions.lateral"
        for (old, new) in MOVED_KEYS
            sec, key = split(old, '.')
            m = errmsg(() -> load_config(f("[$sec]\n$key = \"natural\"\n")))
            @test occursin(old, m) && occursin(new, m) && occursin("moved", m)
            m = errmsg(() -> load_config(set = ["$old=walls"]))
            @test occursin(new, m)
        end
        # Family enums.
        m = errmsg(() -> load_config(set = ["boundary_conditions.vertical_scalars=d3"]))
        @test occursin("boundary_conditions.vertical_scalars", m) && occursin(":r1t1x", m)
        @test occursin("boundary_conditions.sedimentation_lid",
                       errmsg(() -> load_config(set = ["boundary_conditions.sedimentation_lid=open"])))
        @test occursin("boundary_conditions.lateral",
                       errmsg(() -> load_config(set = ["boundary_conditions.lateral=open"])))
        # Overrides: a bad side or value dies at load, naming the entry and the vocabulary.
        m = errmsg(() -> load_config(set = ["boundary_conditions.override.nu_r.lid=natural"]))
        @test occursin("boundary_conditions.override.nu_r.lid", m) && occursin("\"top\"", m)
        m = errmsg(() -> load_config(f("[boundary_conditions.override.nu_r]\ntop = \"zero_flux\"\n")))
        @test occursin("override.nu_r.top", m) && occursin("\"second_derivative\"", m)
        @test occursin("override.nu_r", errmsg(() -> load_config(set = ["boundary_conditions.override.nu_r=3"])))
        # A good override loads (from TOML and from --set) ...
        good = load_config(f("[boundary_conditions.override.nu_r]\ntop = \"dirichlet\"\n");
                           set = ["boundary_conditions.override.p.bottom=r1t1x"])
        @test good.boundary_conditions.override == Dict("nu_r" => Dict("top" => "dirichlet"),
                                                        "p" => Dict("bottom" => "r1t1x"))
        # ... but a variable the run does not have dies at build, listing the run's variables.
        m = errmsg(() -> build_model(load_config(set = ["output.dir=$(dir)",
                         "boundary_conditions.override.rho_r.top=dirichlet"])))
        @test occursin("override.rho_r", m) && occursin("nu_r", m) && occursin("rho_e", m)
        # inner/outer overrides are refused under periodic lateral BCs.
        m = errmsg(() -> build_model(load_config(set = ["output.dir=$(dir)",
                         "boundary_conditions.override.p.inner=neumann"])))
        @test occursin("periodic", m) && occursin("override.p.inner", m)
        # Missing file.
        @test occursin("not found", errmsg(() -> load_config(joinpath(dir, "nope.toml"))))
    end
end

@testset "paths" begin
    mktempdir() do root
        root = realpath(root)
        sub = mkpath(joinpath(root, "case"))
        ovl = mkpath(joinpath(root, "overlays"))
        a = writetoml(joinpath(sub, "scythe_config.toml"), """
            [output]
            dir = "out"
            [reference_state]
            sounding = "../soundings/s.ref"
            [initial_conditions]
            method = "file"
            [initial_conditions.file]
            paths = ["ic1.csv", "/abs/ic2.jld2"]
            """)
        b = writetoml(joinpath(ovl, "o.toml"), """
            [output]
            dir = "out2"
            """)
        c = load_config(a)
        @test c.output.dir == joinpath(sub, "out")
        @test c.reference_state.sounding == joinpath(root, "soundings", "s.ref")
        @test c.initial_conditions.file.paths == [joinpath(sub, "ic1.csv"), "/abs/ic2.jld2"]
        # A path set in an overlay resolves against the OVERLAY's directory.
        c2 = load_config(a, b)
        @test c2.output.dir == joinpath(ovl, "out2")
        @test c2.reference_state.sounding == joinpath(root, "soundings", "s.ref")
        # @scythe/ expands to the package root.
        c3 = load_config(a; set = ["reference_state.sounding=@scythe/benchmarks/x.ref"])
        @test c3.reference_state.sounding == joinpath(pkgdir(Scythe), "benchmarks", "x.ref")
        # --set paths resolve against the working directory.
        c4 = cd(ovl) do
            load_config(a; set = ["output.dir=here", "initial_conditions.file.paths=[\"p.csv\"]"])
        end
        @test c4.output.dir == joinpath(ovl, "here")
        @test c4.initial_conditions.file.paths == [joinpath(ovl, "p.csv")]
        # Julia construction resolves against the working directory too; "" stays "".
        c5 = cd(() -> RunConfig(output = OutputSection(dir = "o"),
                                reference_state = ReferenceStateSection(sounding = "")), sub)
        @test c5.output.dir == joinpath(sub, "o") && c5.reference_state.sounding == ""
        # The written file holds absolute paths, so it re-reads identically from anywhere.
        w = write_config(c, joinpath(root, "elsewhere", "resolved.toml"))
        @test cd(() -> load_config(w), ovl) == c
        @test TOML.parsefile(w)["output"]["dir"] == joinpath(sub, "out")
    end
end

@testset "registries" begin
    # Every registered type is a Type, every description names the reading file.
    for reg in (KNOWN_OPTION_KEYS, KNOWN_PHYSICAL_KEYS), (k, (T, d)) in reg
        @test T isa Type
        @test occursin(".jl", d)
    end
    # Owned keys are real keys; retired keys are not registered.
    @test issubset(keys(OWNED_OPTION_KEYS), keys(KNOWN_OPTION_KEYS))
    @test issubset(keys(OWNED_PHYSICAL_KEYS), keys(KNOWN_PHYSICAL_KEYS))
    @test isempty(intersect(keys(RETIRED_OPTION_KEYS), keys(KNOWN_OPTION_KEYS)))
    # Kvdiff* stay overridable through [extra.physical_params] (plan §2.1/§3).
    @test !any(k -> haskey(OWNED_PHYSICAL_KEYS, k), (:Kvdiff, :Kvdiff_heat, :Kvdiff_water))
    # Each owning field named in OWNED_* exists in the struct tree.
    function has_field(path)
        x = RunConfig()
        for p in split(path, '.')
            hasfield(typeof(x), Symbol(p)) || return false
            x = getfield(x, Symbol(p))
        end
        return true
    end
    for (k, owner) in merge(OWNED_OPTION_KEYS, OWNED_PHYSICAL_KEYS)
        startswith(owner, "(") && continue
        @test has_field(owner)
    end
    # EQUATION_SETS geometries are the canonical Springsteel names.
    @test all(g in ("RiRk", "RZ", "RLR", "RRR") for e in values(EQUATION_SETS) for g in e.geometries)

    # THE GREP: every key src/ reads through get(...)/haskey(...) on options or
    # physical_params (and the `opts`/`pp` aliases the readers use) must be registered.
    srcdir = joinpath(pkgdir(Scythe), "src")
    opt_re = r"(?:get|haskey|get!)\(\s*(?:[A-Za-z_][A-Za-z_0-9]*\.)*(?:options|opts)\s*,\s*:([A-Za-z_][A-Za-z_0-9]*)"
    pp_re  = r"(?:get|haskey|get!)\(\s*(?:[A-Za-z_][A-Za-z_0-9]*\.)*(?:physical_params|pp)\s*,\s*:([A-Za-z_][A-Za-z_0-9]*)"
    missing_opt = String[]
    missing_pp = String[]
    n_opt = Set{Symbol}()
    n_pp = Set{Symbol}()
    for f in sort(readdir(srcdir))
        (endswith(f, ".jl") && f != "config.jl") || continue
        for (i, line) in enumerate(eachline(joinpath(srcdir, f)))
            startswith(lstrip(line), "#") && continue
            for m in eachmatch(opt_re, line)
                k = Symbol(m.captures[1])
                push!(n_opt, k)
                (haskey(KNOWN_OPTION_KEYS, k) || haskey(RETIRED_OPTION_KEYS, k)) ||
                    push!(missing_opt, "$f:$i :$k")
            end
            for m in eachmatch(pp_re, line)
                k = Symbol(m.captures[1])
                push!(n_pp, k)
                haskey(KNOWN_PHYSICAL_KEYS, k) || push!(missing_pp, "$f:$i :$k")
            end
        end
    end
    isempty(missing_opt) || @info "options keys read in src/ but missing from KNOWN_OPTION_KEYS" missing_opt
    isempty(missing_pp) || @info "physical_params keys read in src/ but missing from KNOWN_PHYSICAL_KEYS" missing_pp
    @test isempty(missing_opt)
    @test isempty(missing_pp)
    # The grep must actually be seeing the reads (a broken regex would pass vacuously).
    @test length(n_opt) > 60
    @test length(n_pp) > 40
end

end # config
