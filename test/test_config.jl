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
    @test c.boundary_conditions.vertical_scalars === :natural
    @test c.boundary_conditions.vertical_scalars_lid === :natural
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
                    vertical_scalars = :r1t1x, vertical_scalars_lid = :neumann,
                    sedimentation_lid = :wall,
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
            @test back.boundary_conditions.vertical_scalars_lid === :neumann
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
        # numerics.water_anchor: retired with slot 3 at Stage 4 (the derived anchor is the
        # only behaviour), refused by name with the reason, whatever the value.
        for wa in ("slot", "derived")
            m = errmsg(() -> load_config(set = ["numerics.water_anchor=$wa"]))
            @test occursin("numerics.water_anchor", m) && occursin("retired", m)
        end
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
        # numerics.wall_bc set both ends; its refusal says the lid is now a separate key.
        @test occursin("boundary_conditions.vertical_scalars_lid",
                       errmsg(() -> load_config(set = ["numerics.wall_bc=d2"])))
        # Family enums.
        m = errmsg(() -> load_config(set = ["boundary_conditions.vertical_scalars=d3"]))
        @test occursin("boundary_conditions.vertical_scalars", m) && occursin(":r1t1x", m)
        m = errmsg(() -> load_config(set = ["boundary_conditions.vertical_scalars_lid=d3"]))
        @test occursin("boundary_conditions.vertical_scalars_lid", m) && occursin(":r1t1x", m)
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

@testset "vertical scalar family: natural ground and lid" begin
    # build_boundary_conditions on the default XZ config and on the axisym set: the scalar
    # family is natural at the ground (`vertical_scalars`) and at the lid (`vertical_scalars_lid`), `w`
    # Dirichlet at both, the sedimenting slots natural at both.
    for es in ("moist_compressible_XZ", "moist_compressible_axisym")
        geo = es == "moist_compressible_XZ" ? "RiRk" : "RZ"
        cf(xs...) = load_config(set = ["run.equation_set=$es", "grid.geometry=$geo", xs...])
        cfg = cf()
        options, _ = Scythe.build_physics(cfg)
        vars = Scythe.mc_var_names(options; cyl = EQUATION_SETS[es].cyl)
        sed = Scythe.sedimenting_var_names(options)
        _, _, bcb, bct = Scythe.build_boundary_conditions(cfg, options, vars)
        @test bcb["w"] == DirichletBC() && bct["w"] == DirichletBC()
        for v in vars
            v == "w" && continue
            @test bcb[v] == NaturalBC()
            @test bct[v] == NaturalBC()
        end
        # (slot 3 is the vapor since Stage 4: there is no "rho_t" in `vars`)
        @test !("rho_t" in vars)
        rv = Scythe.vapor_var_name(options)   # "nu_v" under the default bhyp vapor
        for v in ("p", "rho_d", rv, "E_t")
            @test bcb[v] == NaturalBC() && bct[v] == NaturalBC()
        end
        # d2 on both keys restores the old both-sides table exactly: the scalar family d2 at
        # ground and lid, the sedimenting slots natural at both.
        cfg2 = cf("boundary_conditions.vertical_scalars=d2",
                  "boundary_conditions.vertical_scalars_lid=d2")
        _, _, bcb2, bct2 = Scythe.build_boundary_conditions(cfg2, options, vars)
        @test all(bcb2[v] == (v == "w" ? DirichletBC() : v in sed ? NaturalBC() : SecondDerivativeBC()) for v in vars)
        @test bct2 == bcb2
        # A lid mode applies to the lid only; the ground keeps vertical_scalars.
        cfg3 = cf("boundary_conditions.vertical_scalars=natural",
                  "boundary_conditions.vertical_scalars_lid=r1t1x")
        _, _, bcb3, bct3 = Scythe.build_boundary_conditions(cfg3, options, vars)
        @test all(bcb3[v] == (v == "w" ? DirichletBC() : NaturalBC()) for v in vars)
        # the acoustic set (rho_d, E_t, Q_ss; rho_t left it with its slot at Stage 4) is
        # Neumann at an r1t1x lid, the vapor d2 like the rest of the family
        @test bct3["p"] === Scythe.Springsteel.CubicBSpline.R1T1X && bct3["rho_d"] == NeumannBC()
        @test bct3["E_t"] == NeumannBC() && bct3["Q_ss"] == NeumannBC()
        @test bct3[rv] == SecondDerivativeBC()
        @test bct3["w"] == DirichletBC()
    end
end

@testset "spline filter: horizontal and vertical l_q" begin
    SS = Scythe.Springsteel
    @test Numerics().l_q == 2.0
    @test Numerics().l_q_horizontal == 0.0 && Numerics().l_q_vertical == 0.0
    mktempdir() do dir
        cf(xs...) = load_config(set = ["output.dir=$(dir)", xs...])
        vars(cfg) = Scythe.mc_var_names(first(Scythe.build_physics(cfg));
                                        cyl = EQUATION_SETS[cfg.run.equation_set].cyl)

        # Round trip of the two new keys.
        cfg = cf("numerics.l_q_horizontal=1.0", "numerics.l_q_vertical=2.5")
        back = load_config(write_config(cfg, joinpath(dir, "lq.toml")))
        @test back == cfg
        @test back.numerics.l_q_horizontal == 1.0 && back.numerics.l_q_vertical == 2.5

        # INHERIT: both 0 is the one-entry dict every run had before the split -- and so is
        # any setting that resolves both directions to one value.
        @test Scythe.build_l_q(cf(), vars(cf())) == Dict("default" => 2.0)
        @test build_model(cf()).grid_params.l_q == Dict("default" => 2.0)
        @test Scythe.build_l_q(cf("numerics.l_q=1.5"), vars(cf())) == Dict("default" => 1.5)
        @test Scythe.build_l_q(cf("numerics.l_q_horizontal=3.0", "numerics.l_q_vertical=3.0"),
                               vars(cf())) == Dict("default" => 3.0)
        @test Scythe.build_l_q(cf("numerics.l_q_vertical=2.0"), vars(cf())) ==
              Dict("default" => 2.0)

        # SPLIT: horizontal is the default, vertical one `_k` key per slot of THIS
        # configuration -- the default config carries the transformed `nu_*` names, ice and
        # `rho_e`, so `check_mc_var_names` (run by build_model) is genuinely exercised.
        c1 = cf("numerics.l_q_horizontal=1.0")
        m1 = build_model(c1)
        lq = m1.grid_params.l_q
        vs = vars(c1)
        @test "nu_c" in vs && "rho_e" in vs
        @test lq["default"] == 1.0
        @test Set(keys(lq)) == Set(["default"; [v * "_k" for v in vs]])
        @test all(lq[v * "_k"] == 2.0 for v in vs)
        # ... and the BUILT grid carries it: every horizontal spline at 1.0, every vertical
        # one at 2.0. This is what Springsteel actually made of the dict, not the dict.
        grid = SS.createGrid(m1.grid_params)
        gvars = m1.grid_params.vars
        @test all(grid.ibasis.data[z, gvars[v]].params.l_q == 1.0
                  for v in vs for z in axes(grid.ibasis.data, 1))
        @test all(grid.kbasis.data[gvars[v]].params.l_q == 2.0 for v in vs)
        # The vertical override alone, the other way round.
        m2 = build_model(cf("numerics.l_q_vertical=3.0"))
        g2 = SS.createGrid(m2.grid_params)
        @test all(g2.ibasis.data[1, i].params.l_q == 2.0 for i in values(m2.grid_params.vars))
        @test all(g2.kbasis.data[i].params.l_q == 3.0 for i in values(m2.grid_params.vars))
        # The unsplit grid is the old grid.
        g0 = SS.createGrid(build_model(cf()).grid_params)
        @test all(g0.kbasis.data[i].params.l_q == 2.0 for i in 1:length(g0.kbasis.data))

        # check_mc_var_names: a `_k` key is checked by the slot it qualifies, on l_q only.
        bad = deepcopy(m1)
        bad.grid_params.l_q["rho_c_k"] = 1.0
        m = errmsg(() -> Scythe.check_mc_var_names(bad))
        @test occursin("\"rho_c_k\"", m) && occursin("\"nu_c_k\"", m)
        bad = deepcopy(m1)
        bad.grid_params.l_q["bogus_k"] = 1.0
        @test occursin("bogus_k", errmsg(() -> Scythe.check_mc_var_names(bad)))
        bad = deepcopy(m1)
        bad.grid_params.BCB["u_k"] = bad.grid_params.BCB["u"]
        @test occursin("grid_params.BCB declares \"u_k\"",
                       errmsg(() -> Scythe.check_mc_var_names(bad)))

        # Refusals: negative lengths, and a vertical length on the Chebyshev vertical.
        @test occursin("numerics.l_q_vertical",
                       errmsg(() -> build_model(cf("numerics.l_q_vertical=-1.0"))))
        @test occursin("numerics.l_q_horizontal",
                       errmsg(() -> build_model(cf("numerics.l_q_horizontal=-0.5"))))
        @test occursin("numerics.l_q ", errmsg(() -> build_model(cf("numerics.l_q=-2.0"))))
        m = errmsg(() -> build_model(cf("run.equation_set=moist_compressible_axisym",
                                        "grid.geometry=RZ", "boundary_conditions.lateral=walls",
                                        "numerics.l_q_vertical=1.0")))
        @test occursin("numerics.l_q_vertical", m) && occursin("Chebyshev", m)
    end
end

@testset "horizontal mixing: the Smagorinsky length rule" begin
    # Default: Ls = 0, i.e. the grid-scaled C_s * dx on every patch of every geometry. The
    # absolute 500 m the axisymmetric TC was tuned with is pinned in that case's own config.
    @test HorizontalMixing().Ls == 0.0 && HorizontalMixing().C_s == 0.2
    mktempdir() do dir
        cf(xs...) = load_config(set = ["output.dir=$(dir)", xs...])
        SM = "physics.horizontal_mixing.scheme=smagorinsky"
        # Round trip of the new key.
        c = cf(SM, "physics.horizontal_mixing.Ls=0", "physics.horizontal_mixing.C_s=0.3")
        @test load_config(write_config(c, joinpath(dir, "hm.toml"))) == c

        # scheme none emits neither key; the driver then has no length.
        _, p = Scythe.build_physics(cf())
        @test !haskey(p, :Ls) && !haskey(p, :C_s)
        @test Scythe.smagorinsky_length(build_model(cf())) == 0.0

        # XZ, Ls > 0 set explicitly: absolute, :C_s not emitted.
        m = build_model(cf(SM, "physics.horizontal_mixing.Ls=500"))
        @test m.physical_params[:Ls] == 500.0 && !haskey(m.physical_params, :C_s)
        @test Scythe.smagorinsky_length(m) == 500.0
        # XZ, Ls left at its default 0: C_s times this grid's cell width (300 km / 50 cells),
        # the same as writing Ls = 0 explicitly.
        m = build_model(cf(SM))
        @test m.physical_params[:Ls] == 0.0 && m.physical_params[:C_s] == 0.2
        @test Scythe.smagorinsky_length(m) == 0.2 * (300.0e3 / 50)
        m = build_model(cf(SM, "physics.horizontal_mixing.Ls=0"))
        @test m.physical_params[:C_s] == 0.2
        @test Scythe.smagorinsky_length(m) == 0.2 * (300.0e3 / 50)
        m = build_model(cf(SM, "physics.horizontal_mixing.Ls=0", "physics.horizontal_mixing.C_s=0.4"))
        @test Scythe.smagorinsky_length(m) == 0.4 * (300.0e3 / 50)
        @test m.physical_params[:Khdiff_heat] == -1.0     # heat on Smagorinsky by default

        # Refusals: no length at all; negatives.
        msg = errmsg(() -> build_model(cf(SM, "physics.horizontal_mixing.Ls=0",
                                          "physics.horizontal_mixing.C_s=0")))
        @test occursin("no length scale", msg) && occursin("C_s", msg)
        @test occursin("must be >= 0", errmsg(() -> build_model(cf(SM,
                                              "physics.horizontal_mixing.C_s=-0.2"))))

        # NESTS: the C_s length is per patch, from each patch's own cell width; an absolute
        # Ls is the same on every patch. The TC as shipped pins Ls = 500 in its own config,
        # so it is bitwise what it was before the default moved to 0.
        tc = joinpath(pkgdir(Scythe), "simulations", "tc_axisym", "scythe_config.toml")
        tcf(xs...) = load_config(tc; set = ["output.dir=$(dir)", xs...])
        @test tcf().physics.horizontal_mixing.Ls == 500.0
        @test tcf().physics.horizontal_mixing.Ls != HorizontalMixing().Ls   # a pin, not the default
        models, _ = Scythe.build_nest(build_model(tcf()))
        @test length(models) > 1
        @test all(Scythe.smagorinsky_length(mm) == 500.0 for mm in models)
        @test all(!haskey(mm.physical_params, :C_s) for mm in models)
        models, _ = Scythe.build_nest(build_model(tcf("physics.horizontal_mixing.Ls=0")))
        dxs = [(mm.grid_params.iMax - mm.grid_params.iMin) / mm.grid_params.num_cells_i
               for mm in models]
        @test length(unique(dxs)) == length(models)   # the nests do differ
        @test [Scythe.smagorinsky_length(mm) for mm in models] == 0.2 .* dxs
    end
end

@testset "RRR: the 3-D Cartesian box (grid, four lateral faces, bubble)" begin
    # Stream C2/R2a. The second horizontal direction of the box is configured by
    # grid.boundaries_j / grid.num_cells_j (required on RRR, refused elsewhere), the bubble
    # gains a y centre (temperature_bubble.yc, RRR only), `lateral` governs all four faces and
    # `override` gains the j faces `down` (y = jMin, Springsteel BCD) and `up` (y = jMax, BCU).
    # None of it may touch the canonical text of a 2-D config: the new keys are written only
    # when set, so every existing config hashes as it did.
    @test GridSection().boundaries_j == Float64[] && GridSection().num_cells_j == Int[]
    @test BubbleIC().yc === nothing
    for c in (RunConfig(), RunConfig(initial_conditions = InitialConditions(method = :temperature_bubble)))
        t = Scythe.canonical_toml(c)
        @test !occursin("boundaries_j", t) && !occursin("num_cells_j", t) && !occursin("yc", t)
    end
    mktempdir() do dir
        rrr = ["run.equation_set=moist_compressible_RRR", "grid.geometry=RRR",
               "grid.boundaries=[0.0, 24.0e3]", "grid.num_cells=[6]",
               "grid.boundaries_j=[0.0, 24.0e3]", "grid.num_cells_j=[6]",
               "grid.z_top=4000.0", "grid.num_cells_k=8",
               "boundary_conditions.lateral=walls",
               "initial_conditions.method=temperature_bubble",
               "initial_conditions.temperature_bubble.xc=12.0e3",
               "initial_conditions.temperature_bubble.yc=12.0e3",
               "initial_conditions.temperature_bubble.xr=8.0e3",
               "initial_conditions.temperature_bubble.zc=500.0",
               "initial_conditions.temperature_bubble.zr=1000.0",
               "physics.radiation.scheme=none", "environment.f_latitude=0.0"]
        cf(xs...) = load_config(set = ["output.dir=$(joinpath(dir, "out"))", rrr..., xs...])
        cfg = cf()
        @test cfg.grid.boundaries_j == [0.0, 24.0e3] && cfg.grid.num_cells_j == [6]
        @test cfg.initial_conditions.temperature_bubble.yc == 12.0e3
        # Round trip: the RRR keys are written and read back.
        path = write_config(cfg, joinpath(dir, "rrr.toml"))
        text = read(path, String)
        @test occursin("boundaries_j", text) && occursin("num_cells_j", text) && occursin("yc", text)
        back = load_config(path)
        @test back == cfg && config_hash(back) == config_hash(cfg)

        # ── refusals (all at build: the constructor only checks well-formedness)
        bm(xs...) = errmsg(() -> build_model(cf(xs...)))
        noj = filter(s -> !startswith(s, "grid.boundaries_j") && !startswith(s, "grid.num_cells_j"), rrr)
        m = errmsg(() -> build_model(load_config(set = ["output.dir=$(dir)", noj...])))
        @test occursin("grid.boundaries_j", m) && occursin("RRR", m)
        m = errmsg(() -> build_model(load_config(set = ["output.dir=$(dir)",
                                                        "grid.boundaries_j=[0.0, 1.0e3]",
                                                        "grid.num_cells_j=[4]"])))
        @test occursin("grid.boundaries_j", m) && occursin("RRR", m)
        @test occursin("grid.num_cells_j", bm("grid.num_cells_j=[6, 6]"))
        @test occursin("grid.boundaries_j", bm("grid.boundaries_j=[24.0e3, 0.0]"))
        @test occursin("grid.num_cells_j", bm("grid.num_cells_j=[0]"))
        # yc is the second horizontal coordinate: refused off RRR, required on it.
        m = errmsg(() -> build_model(load_config(set = ["output.dir=$(dir)",
                         "initial_conditions.method=temperature_bubble",
                         "initial_conditions.temperature_bubble.yc=1.0"])))
        @test occursin("temperature_bubble.yc", m) && occursin("RRR", m)
        noyc = filter(s -> !startswith(s, "initial_conditions.temperature_bubble.yc"), rrr)
        m = errmsg(() -> build_model(load_config(set = ["output.dir=$(dir)", noyc...])))
        @test occursin("temperature_bubble.yc", m)
        # A noise-started box needs no yc.
        @test build_model(load_config(set = ["output.dir=$(dir)", noyc...,
                                             "initial_conditions.method=temperature_noise"])) isa ModelParameters
        # One patch; no horizontal SI / exact solve (neither has an RRR operator).
        @test occursin("nest", bm("grid.boundaries=[0.0, 12.0e3, 24.0e3]", "grid.num_cells=[3, 6]"))
        @test occursin("numerics.exact_si", bm("numerics.exact_si=true", "numerics.state_dependent_si=false"))
        @test occursin("numerics.horizontal_semiimplicit", bm("numerics.horizontal_semiimplicit=true"))
        # The j faces exist on RRR only, and a periodic pair is not overridden on one side.
        m = errmsg(() -> build_model(load_config(set = ["output.dir=$(dir)",
                         "boundary_conditions.lateral=walls",
                         "boundary_conditions.override.p.up=neumann"])))
        @test occursin("override.p.up", m) && occursin("RRR", m)
        m = bm("boundary_conditions.lateral=periodic", "boundary_conditions.override.p.down=natural")
        @test occursin("periodic", m) && occursin("override.p.down", m)

        # An untransformed rain slot would get a positivity bound the RRR grid drops: refused.
        m = bm("physics.microphysics.rain_transform=none")
        @test occursin("positivity", m) && occursin("RRR", m)
        @test build_model(cf("physics.microphysics.rain_transform=none",
                             "numerics.positivity=off")) isa ModelParameters

        # ── the build: one patch, all four lateral faces, v (slot 10) carried
        model = build_model(cfg)
        gp = model.grid_params
        @test model isa ModelParameters && model.equation_set == "moist_compressible_RRR"
        @test gp.geometry == "RRR"
        @test (gp.jMin, gp.jMax, gp.num_cells_j) == (0.0, 24.0e3, 6)
        vars = first.(sort(collect(gp.vars), by = last))
        @test vars[10] == "v" && vars == Scythe.mc_var_names(model.options; cyl = true)
        @test haskey(model.physical_params, :f)               # an f-plane box rotates
        # walls: (u, w) Dirichlet on the x faces, (v, w) Dirichlet on the y faces, every
        # other variable Neumann; the vertical families as on every geometry.
        for v in vars
            @test gp.BCL[v] == (v in ("u", "w") ? DirichletBC() : NeumannBC())
            @test gp.BCR[v] == gp.BCL[v]
            @test gp.BCD[v] == (v in ("v", "w") ? DirichletBC() : NeumannBC())
            @test gp.BCU[v] == gp.BCD[v]
        end
        @test gp.BCB["w"] == DirichletBC() && gp.BCT["w"] == DirichletBC()
        # periodic: every variable on all four faces.
        gpp = build_model(cf("boundary_conditions.lateral=periodic")).grid_params
        @test all(gpp.BCL[v] == PeriodicBC() && gpp.BCR[v] == PeriodicBC() &&
                  gpp.BCD[v] == PeriodicBC() && gpp.BCU[v] == PeriodicBC() for v in vars)
        # override: down/up act on the j faces only (the up and down dicts are separate).
        gpo = build_model(cf("boundary_conditions.override.p.up=natural",
                             "boundary_conditions.override.u.inner=natural")).grid_params
        @test gpo.BCU["p"] == NaturalBC() && gpo.BCD["p"] == NeumannBC()
        @test gpo.BCL["u"] == NaturalBC() && gpo.BCR["u"] == DirichletBC()
        # 2-D sets still get the 4 dicts (positions 5, 6 are nothing) and no j BCs passed.
        x2 = load_config(set = ["output.dir=$(dir)"])
        o2, _ = Scythe.build_physics(x2)
        bcs = Scythe.build_boundary_conditions(x2, o2, Scythe.mc_var_names(o2))
        @test length(bcs) == 6 && bcs[5] === nothing && bcs[6] === nothing
        # The resolved table reports the j faces on RRR (and only there).
        t = Scythe.boundary_condition_table(model, cfg)
        rv = only(filter(r -> r.variable == "v", t))
        @test rv.down == "dirichlet" && rv.up == "dirichlet" && rv.inner == "neumann"
        @test all(r.down == "" && r.up == "" for r in Scythe.boundary_condition_table(build_model(x2), x2))
        f1 = Scythe.write_boundary_conditions(joinpath(dir, "bc.toml"), t)
        @test only(filter(x -> x["name"] == "v", TOML.parsefile(f1)["patch"][1]["variable"]))["up"] == "dirichlet"
        @test occursin("down|up", sprint(Scythe.print_boundary_conditions, t))

        # The run's preflight accepts the box (stage R2c added the RRR halo/tile methods
        # whose absence it used to refuse).
        @test redirect_stdout(devnull) do
            Scythe.preflight_checks(cfg)
        end === nothing

        # ── the initial condition: a sphere-like bubble centred at (xc, yc, zc)
        _, topo = Scythe.initialize_case!(cfg, model)
        @test topo === nothing && isfile(model.initial_conditions)
        patch = Scythe.createGrid(gp)
        Scythe.read_physical_grid(model.initial_conditions, patch)
        gpts = Scythe.getGridpoints(patch)
        et = patch.physical[:, gp.vars["E_t"], 1]
        imax = argmax(et)
        kD = gp.kDim
        # The warmest point is at the bubble centre's horizontal position, near zc.
        x0, y0 = gpts[imax, 1], gpts[imax, 2]
        dxs = unique(gpts[:, 1]); dys = unique(gpts[:, 2])
        @test abs(x0 - 12.0e3) <= maximum(diff(sort(dxs)))
        @test abs(y0 - 12.0e3) <= maximum(diff(sort(dys)))
        # Square box, centred bubble, identical i/j mish: the field is x<->y symmetric.
        key(i) = (gpts[i, 1], gpts[i, 2], gpts[i, 3])
        lookup = Dict(key(i) => i for i in axes(gpts, 1))
        worst = 0.0
        for i in axes(gpts, 1)
            j = lookup[(gpts[i, 2], gpts[i, 1], gpts[i, 3])]
            worst = max(worst, abs(et[i] - et[j]))
        end
        @test worst <= 1.0e-12 * maximum(abs, et)
        # It is a bubble, not a slab: outside it (here two opposite corners at the same
        # level) the state is the horizontally uniform unperturbed one. (E_t' is not 0 there:
        # the unperturbed point differs from the stored reference by the consistent-Q_ss
        # reconstruction, as on the 2-D grids; it is the same at every such point.)
        lvl(i) = gpts[i, 3] == gpts[imax, 3]
        c1 = findfirst(i -> gpts[i, 1] == minimum(dxs) && gpts[i, 2] == minimum(dys) && lvl(i), axes(gpts, 1))
        c2 = findfirst(i -> gpts[i, 1] == maximum(dxs) && gpts[i, 2] == maximum(dys) && lvl(i), axes(gpts, 1))
        @test et[c1] == et[c2]
        @test et[imax] - et[c1] > 100.0          # a 3 K bubble: ~2.4e2 J m^-3 at its centre
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
