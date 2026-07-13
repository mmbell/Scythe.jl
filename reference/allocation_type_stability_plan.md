# Scythe allocation / type-stability refactor — plan

> **UPDATE 2026-07-12 — Phases 1, 2 and 4 are IMPLEMENTED. Read §0 first: it corrects four
> claims in this document that measurement disproved. Phase 1 alone did NOT stop the crash;
> Phases 1+2+4 together appear to.**

---

## 0. What was actually implemented (measured)

### 0z. Final state — Phases 1 + 2 + 4

| straka93 quick/mc/rirk | before | Phase 1 | Phase 2+4 | **final** |
|---|---|---|---|---|
| allocations | 4.73 G | 1.94 G | 757 M | **215 M** (22×) |
| allocated | 836.6 GiB | 680.9 GiB | 224.9 GiB | **38.3 GiB** (22×) |
| wall clock | 484.5 s | 484.0 s | 392 s | **379 s** (1.28×) |
| GC time | 14.2 % | 20.2 % | ~6 % | **1.0 %** |
| lock conflicts | 1 506 732 | — | 1 535 657 | **840** (1794×) |
| crash | ~1 in 3 runs | 1 in 3 runs | 0 in 8 runs | **0 in 5 runs** |

| bf02_dry quick/mc/rz | before | **final** |
|---|---|---|
| allocations | 991.2 M | **161.2 M** (6.1×) |
| allocated | 213.2 GiB | **16.6 GiB** (12.8×) |
| wall clock | 269.8 s | **99.8 s** (2.7×) |
| GC time | 12.0 % | **1.5 %** |

Per-column call (`moist_compressible_XZ`, RiRk): **1447 → 7 allocations** (206×);
`diffusion_timestep_mc` **249 → 0**.
Output is **bit-identical** throughout (all 8 bf02_dry CSVs diff byte-for-byte clean against a
pre-refactor run, at every stage). Tests: **4389 passing, 0 failures.**

**What Phase 2 changed.** The four `deepcopy(tile.kbasis.data[v])` sites in
`moist_compressible.jl` now borrow a persistent **per-thread** work column from
`mtile.scratch_columns[threadid(), var]`, and the loop-invariant `ref_*(refstate)[:,N]` copies
became views. Per-thread, *not* per-column: per-column scratch would need **2.9 GB** on a
full-mode RZ run (1536 columns × 1.9 MiB/column), versus ~90 MiB per-thread. That makes
`threadid()` an ownership tag, which is only sound under **`Threads.@threads :static`** — the
column loop in `advanceTimestep` was switched accordingly (the columns are equal-cost, so
there is nothing for dynamic scheduling to balance).

The scratch columns are keyed per **variable** as well as per thread, for two reasons:
`Btransform!`/`Atransform!` consult the column's own BCs, and `semiimplicit_adjustment_p` holds
the p- and w-columns live simultaneously (`p_nstar` aliases the p-column's `uMish` and is read
*after* the w-column is transformed) — handing it one object twice would corrupt the pressure
update silently.

### 0a. Phase 1 results (for the record)

`ModelTile` and `ModelParameters` are now fully concrete (every field passes
`isconcretetype`). Output is **bit-identical** — the 8 `bf02_dry quick/mc` output CSVs diff
byte-for-byte clean against a pre-change run. Test suite: **4374 passing, 0 failures** (4328
baseline + 46 new in `test/test_allocations.jl`).

| straka93 quick/mc/rirk | before | after Phase 1 |
|---|---|---|
| allocations | 4.73 G | **1.94 G** (2.4×) |
| allocated | 836.6 GiB | 680.9 GiB |
| wall clock | 484.5 s | 484.0 s (**unchanged**) |
| GC time | 14.2 % | 20.2 % |
| lock conflicts | 1.51 M | 1.74 M |

| bf02_dry quick/mc/rz | before | after Phase 1 |
|---|---|---|
| allocations | 991.2 M | **359.2 M** (2.8×) |
| wall clock | 269.8 s | **175.8 s** (1.53×) |

Per-column call (`moist_compressible_XZ`, RiRk): **1447 → 619 allocations**, 128,336 → 76,768 B.

### 0b. Phase 1 alone did NOT fix the crash

`straka93 --mode quick --stage mc --grid rirk` still died intermittently after Phase 1: **1 of 3
runs segfaulted at t = 793.6 s of 900 s** (worker 3, signal 11), the same late-run intermittent
signature as the original signal-4 death at t = 862.5 s. That is what motivated Phase 2.

(Phase 1 did not *cause* it: output was bit-identical, the suite was green, and every column of
every timestep runs the same code — a type-induced miscompile would fail on step 1, not after
12,700 successful steps.)

**After Phases 2+4: 8 consecutive clean runs to t = 900.** Not a proof of absence — but against a
~1-in-3 baseline crash rate, 8 clean runs is a ~4 % chance of luck, and the mechanism moved in
the right direction (6.2× fewer allocations, GC time 14 % → 6 %).

### 0c. Four claims in this document that measurement disproved

1. **§2e's "~170× reduction" premise is wrong.** It assumed the allocations were *boxed
   scalars* that concrete typing would collapse into one array per broadcast. They are not.
   They are **genuine array temporaries**, and concrete typing does not remove them. The real
   reduction is **2.4×**, not 170×.

2. **§4 Phase 2's priority order is backwards.** It calls the loop-invariant reference-column
   copies the "highest payoff, lowest risk" item. An allocation profile
   (`Profile.Allocs`, sample_rate=1.0) says otherwise — the ref-column `[:,1]` copies are ~2
   allocations each, while **the four `deepcopy(kbasis.data[...])` sites cost 93 allocations
   *each*** and are **60 % of everything left**:

   | site | allocs / column call |
   |---|---|
   | `moist_compressible.jl:527` `deepcopy` (moist_compressible_XZ) | 93 |
   | `moist_compressible.jl:645` `deepcopy` (semiimplicit_adjustment_p) | 93 |
   | `moist_compressible.jl:656` `deepcopy` (semiimplicit_adjustment_p) | 93 |
   | `moist_compressible.jl:790` `deepcopy` (diffusion_timestep_mc) | 93 |
   | `_vertical_solve!` internals (`semiimplicit.jl:1598,1602`) | 40 |
   | `SItransform` (`CubicBSpline.jl:1541`) | 6 |
   | each `ref_*(refstate)[:,N]` copy | 2 |
   | **total** | **619** |

   **Phase 2 must start with the `deepcopy` sites (item 3), not the ref columns (item 1).**

3. **§4's `tilepoints::Matrix{Float64}` would break every 1-D grid.** `getGridpoints` returns a
   bare `Vector` for `_1DCartesianGrid` / `_1DCartesianZ` / `_1DCartesianL` (it hands back
   `ibasis.data[1,1].mishPoints`), which is exactly why `createModelTile` reads
   `tilepoints[:, ndims(tilepoints)]`. It needs a **rank type parameter**.

4. **§4's `h_matrix::LU{Float64,Matrix{Float64},Vector{Int64}}` would break every explicit
   run.** The §7c table only sampled semi-implicit-*on* configs. The dummy placeholder
   `factorize([1 2; 2 1])` is detected as **tridiagonal**, giving
   `LU{Float64,Tridiagonal{...},Vector{Int64}}` — and that is what lands in `h_matrix` whenever
   `:semiimplicit` is false. Likewise `mc_diffusion_matrices` **cannot** be a
   `Dict{Symbol,F}`: its six entries are *not* one type (differing BCs flip `factorize`'s
   symmetry detection, so on RiRk `u` is a `BunchKaufman` while `w` is an `LU`). It is now a
   **`NamedTuple`** — concrete *and* heterogeneous, which a `Dict` cannot be.

### 0d. The rest of Phase 2 (done) — `_vertical_solve!` and the temporaries

Both are now done, for the mc set only.

**`_vertical_solve!`** was the biggest single item after the deepcopy fix (~40 allocs/call). It
allocated three temporaries per call (`d.W .* rhs_mish`, the `M0'*` product, `h_a \ b`) — and,
worse, reached `_rirk_solve_data`, which **took a global lock and hit a `Dict` on every call**.
That is ~22 M lock acquisitions across a straka93 run, from every thread at once. The solve
data is per-grid and invariant, so it now lives on the tile (`mtile.solve_data`), the work
vectors are per-thread, and the solve is `mul!`/`ldiv!` into them. Zero allocations.

**The broadcast temporaries** (the `p = pp .+ pbar` family) now write into `mtile.mc_scratch` —
one `NamedTuple` of `kDim` work vectors per thread, covering all 76 live temporaries across
`moist_compressible_XZ`, `semiimplicit_adjustment_p` and `diffusion_timestep_mc`. Keyed by NAME,
not index: a `NamedTuple` cannot hold a duplicate field, so two live temporaries sharing one
buffer — a silent, hard-to-see corruption — is not writeable. The three functions' slots are
namespaced (`si_`, `df_`) so they cannot collide either.

Watch out for one trap: **`@.` dots every call in the expression, `view` included.** Writing
`@. p_star = view(vnp1, r, i) + pbar` broadcasts the *view constructor* and throws a
`DimensionMismatch`. Bind views outside the `@.` block.

**Final per-column call (RiRk, mc):**

| | before | after |
|---|---|---|
| `moist_compressible_XZ` | 1447 allocs | **7** (206×) |
| `diffusion_timestep_mc` | 249 allocs | **0** |

### 0e. What is left (not done)

1. **`primitive_equations.jl` is untouched** and still has **13 per-column `deepcopy` sites**,
   plus the same ref-column copies and broadcast temporaries. Only `moist_compressible.jl` was
   converted (it is the crashing set). The same treatment applies, and the machinery
   (`scratch_column`, `_vertical_solve!`'s workspace) is already there — PE only needs its own
   scratch slots.
2. **The 7 remaining mc allocations** are Springsteel's allocating 1-arg
   `Ixtransform`/`Ixxtransform` (called 3× per column) plus the dynamic dispatch. Springsteel has
   a `mul!`-based out-arg variant for Chebyshev (`Chebyshev.jl:683-687`) but not, as far as I can
   see, for the B-spline basis, and neither is reachable through the top-level generic
   (`Springsteel.jl:228-232`) — so this needs a Springsteel change, not a Scythe one.
3. **Phase 3 (static equation-set dispatch) is not worth doing** — measured. The dynamic
   `getfield(Scythe, Symbol(equation_set))` costs exactly **1 allocation** (`physical_model` 8 vs
   `moist_compressible_XZ` 7) and one dynamic call per column per timestep, ≈0.5 s out of 392 s.
   It would have to cover 25 equation-set entry points to save that. **Skip it.**

---

**Status:** ~~not started~~ **Phase 1 implemented 2026-07-12 (see §0).** Written 2026-07-12,
immediately after commit `aedce4d`.
**Prerequisite (recommended, not required):** the Springsteel `vars::Dict{String,Int64}` typing —
see `Springsteel.jl/agent_files/plan_gridparams_dict_typing.md`. Doing that first avoids adding
~8 `::Int64` type assertions per equation set across 26 equation sets, all of which would then be
deleted again. It is *not* a blocker: Phase 1 below delivers the crash fix on its own.

This document is self-contained. Everything needed to execute is here.

---

## 1. The problem

Long benchmark runs die intermittently, at different points near the *end* of the integration —
`straka93 --mode quick --stage mc --grid rirk` died at t = 862.5 s of 900 s; `bf02` has died
similarly. The failure is **not** CFL (no CFL error is raised; the run does not blow up
numerically) and **not** system memory pressure (Activity Monitor stays green throughout).

A worker dies with:

```
[63704] signal 4: Illegal instruction: 4
  _xzm_xzone_malloc_freelist_outlined  (libsystem_malloc)
  gc_sweep_pool     at gc-stock.c:1390 [inlined]
  _jl_gc_collect    at gc-stock.c:3183
  ijl_gc_collect    at gc-stock.c:3471
  maybe_collect     at gc-stock.c:350 [inlined]
  jl_gc_small_alloc_inner ...
  ijl_new_bits      at datatype.c:1143        <-- BOXING
  ijl_get_nth_field at datatype.c:1842
  jl_f__apply_iterate at builtins.c:780       <-- splatted/dynamic getindex
  getindex          at ./abstractarray.jl:1342 [inlined]
  diffusion_timestep_mc at src/moist_compressible.jl:744
  moist_compressible_XZ at src/moist_compressible.jl:546
  physical_model    at src/semiimplicit.jl:655
  advance_column    at src/semiimplicit.jl:630
  #advanceTimestep##2 at ./threadingconstructs.jl:276   <-- inside Threads.@threads
...
Allocations: 12603504344 (Pool: 12603487934; Big: 16410); GC: 17330
```

**The crash is inside the garbage collector**, sweeping the small-object pool. The run made
**12.6 billion allocations** over ~13,800 timesteps — roughly **900,000 allocations per timestep**,
essentially all small pool objects.

This is **not a memory leak.** GC reclaims the memory, which is why RSS stays flat and the machine
looks healthy. It is an allocation *rate* problem: the GC is being driven so hard, from so many
threads, that it falls over.

---

## 2. Root cause (measured, not inferred)

### 2a. Scythe erases Springsteel's concrete types

Springsteel's grid type is fully concrete and parameterized
(`Springsteel.jl/src/types.jl:249`):

```julia
struct SpringsteelGrid{G <: AbstractGeometry, I, J, K} <: AbstractGrid
    params   :: SpringsteelGridParameters
    ibasis   :: I
    jbasis   :: J
    kbasis   :: K
    spectral :: Matrix{Float64}      # always 2D
    physical :: Array{Float64, 3}    # always 3D
end
```

Its transforms are allocation-free, as designed.

**Scythe throws that information away.** `ModelTile` (`src/semiimplicit.jl:24-58`) has
**16 of its 24 fields non-concrete**:

| field | declared | concrete? |
|---|---|---|
| `tile` | `AbstractGrid` | **no** — the critical one |
| `var_np1`, `expdot_incr`, `expdot_n/nm1/nm2`, `impdot_np1/n/nm1/nm2`, `diffdot_n/nm1`, `tilepoints` | `Array{Float64}` | **no** — `Array{Float64}` *is* `Array{Float64,N} where N`, an abstract type |
| `haloReceiveBuffer`, `splineBuffer` | `Array{Float64}` | **no** |
| `h_matrix`, `diffusion_matrix` | `Factorization` | **no** |
| `mc_diffusion_matrices` | `Dict{Symbol,Factorization}` | **no** (abstract valtype) |
| `model` | `ModelParameters` | concrete struct, but see 2b |
| `ref_state`, `patchMap`, `haloSendMap`, `haloReceiveMap`, `patch_b_iDim` | — | concrete |

Because `tile::AbstractGrid`, the expression `grid = mtile.tile; grid.physical` **infers as `Any`**.

### 2b. `ModelParameters` compounds it

`src/Scythe.jl:28-41`: `equation_set`, `initial_conditions`, `output_dir`, `ref_state_file` carry
**no type annotation at all** (⇒ `Any`), and `physical_params::Dict` / `options::Dict` are
`Dict{Any,Any}` at the type level. So every `model.physical_params[:Khdiff]` inside a per-column
function returns `Any` and boxes.

### 2c. `view(...)` does not save you

The equation sets **already use views everywhere** — 61 `view(` calls in `moist_compressible.jl`,
276 in `primitive_equations.jl`, 98 in `semiimplicit.jl`. This does **not** help: a view constructed
over an `Any`-typed container is itself type-unstable, and every element access boxes. (See the
isolation table in §3 — under abstract fields, views are *worse* than copies.)

### 2d. Measurements

Per-column-call allocations, straka93 `mc` config (scripts in §7):

| | bytes / column call |
|---|---|
| `moist_compressible_XZ` (`src/moist_compressible.jl:265`) | **130,064** |
| `diffusion_timestep_mc` (`src/moist_compressible.jl:724`) | **25,952** |
| `physical_model` dynamic-dispatch overhead alone | 736 (minor) |
| `::Any` count in the **optimized** IR of `diffusion_timestep_mc` | **113** |

Isolation test — **both halves of the fix are required**:

| inner loop | abstract fields (today) | concrete fields |
|---|---|---|
| slice copies (`A[r, j]`) | 1408 B | 1280 B |
| `@view` / `@views` | **4368 B — worse!** | **0 B** |

Views alone make it worse. Concrete fields alone still allocate (the broadcast temporaries remain).
Together: exactly zero.

### 2e. The number that sets the scope

What kills the GC is allocation **count**, not bytes. With concrete types, each broadcast becomes
*one* well-typed array allocation instead of thousands of boxed scalars:

- today: **12.6 × 10⁹** allocations
- after concrete typing: ~40 temporaries × 128 columns × 14,400 steps ≈ **74 × 10⁶**

a **~170× reduction in allocation count**. **Phase 1 alone is expected to stop the crash.**
Phase 2 is then a performance optimization, not a correctness requirement.

### 2f. Aggravating factor: thread oversubscription

`benchmarks/*.jl:24` does `addprocs(opts.workers, exeflags="--threads=auto")`. On a 12-core machine
`--threads=auto` gives **12 threads per worker**, and the default is 2 workers ⇒ **24 compute
threads on 12 cores**, all allocating into a shared GC, inside the `Threads.@threads` column loop at
`src/semiimplicit.jl:583`.

---

## 3. Baselines to beat

Recorded on a 12-core machine at commit `aedce4d`:

| run | wall clock | rate |
|---|---|---|
| `straka93 --mode quick --stage mc --grid rirk` | 668.2 s | 21.5 steps/s |
| `straka93 --mode quick --stage mc` (rz) | 567.9 s | 25.4 steps/s |
| `bf02_dry --mode quick --stage mc` (rz) | 191.9 s | 52.1 steps/s |

Full Scythe test suite at `aedce4d`: **4328 passing, 0 failures.**

---

## 4. Plan

### Phase 1 — Concrete types (the fix; global, semantics-preserving)

**`src/semiimplicit.jl:24-58` — reshape `ModelTile`.** Measured facts that keep the parameter count
down (verified with the script in §7c):

- `h_matrix` is `LU{Float64,Matrix{Float64},Vector{Int64}}` on **both** RZ and RiRk ⇒ concrete type,
  **no type parameter needed**.
- `diffusion_matrix` is `BunchKaufman` on RiRk and `LU` on RZ — but `mc_diffusion_matrices` always
  holds the **same** type as `diffusion_matrix` ⇒ **one shared parameter `F`** covers both.
- All 13 state arrays are `Matrix{Float64}` at construction (`src/semiimplicit.jl:76-86` allocates
  them all as `zeros(Float64, size(tile.physical,1), size(tile.physical,2))`, and
  `getGridpoints` returns a `Matrix{Float64}`). **Nothing indexes them N-dimensionally** — every use
  is 2-index. So they can be `Matrix{Float64}` with no parameter.
- `haloReceiveBuffer` is always `Vector{Float64}` (`zeros(Float64, nnz(haloReceiveMap))`, `:130`).
- **`splineBuffer` is DEAD CODE.** It is declared (`:42`), allocated (`:131`), and passed to the
  constructor (`:194`) — and **never read anywhere in `src/`**. It is also the source of the
  Vector-vs-3D-array inconsistency (`allocateSplineBuffer` returns a `Vector` for Cartesian
  spline-only grids and a 3-D array otherwise). **Delete the field.** This removes the
  inconsistency *and* a type parameter.

Result — **3 type parameters, not 6**:

```julia
struct ModelTile{G<:AbstractGrid, R<:AbstractReferenceState, F<:Factorization}
    model::ModelParameters
    tile::G                                    # was AbstractGrid   <-- the critical change
    var_np1::Matrix{Float64}                   # was Array{Float64}
    expdot_incr::Matrix{Float64}
    expdot_n::Matrix{Float64}
    expdot_nm1::Matrix{Float64}
    expdot_nm2::Matrix{Float64}
    impdot_np1::Matrix{Float64}
    impdot_n::Matrix{Float64}
    impdot_nm1::Matrix{Float64}
    impdot_nm2::Matrix{Float64}
    tilepoints::Matrix{Float64}
    ref_state::R
    patchMap::SparseMatrixCSC{Float64, Int64}
    haloSendMap::SparseMatrixCSC{Float64, Int64}
    haloReceiveMap::SparseMatrixCSC{Float64, Int64}
    haloReceiveBuffer::Vector{Float64}
    # splineBuffer DELETED — allocated but never read
    patch_b_iDim::Int64
    h_matrix::LU{Float64, Matrix{Float64}, Vector{Int64}}
    diffusion_matrix::F
    diffdot_n::Matrix{Float64}
    diffdot_nm1::Matrix{Float64}
    mc_diffusion_matrices::Dict{Symbol, F}
end
```

`createModelTile` (`src/semiimplicit.jl:71-199`) needs only to drop `splineBuffer`; it already
allocates everything at the right concrete type. **No call site changes** — `mtile.var_np1` etc.
stay exactly as they are.

**`src/Scythe.jl:28-41` — type `ModelParameters`:**

- `equation_set::String` — **keep it a `String`**: it is string-consumed by
  `uses_physical_reference` / `uses_pressure_reference` in `src/reference_state.jl`
  (`endswith`/`startswith` on the name).
- `initial_conditions::String`, `output_dir::String`, `ref_state_file::String`.
- `physical_params::Dict{Symbol,Float64}` — **normalize inside the existing inner constructor**
  (added in commit `aedce4d` for `compute_derived_params`) rather than tightening the keyword type,
  so a caller passing `:K => 75` (an `Int`) still works. Grep benchmarks/tests for integer-valued
  physical params first.
- Keep `options::Dict{Symbol,Any}` (it genuinely holds mixed `Bool`/`Float64`), and **hoist any
  option read out of per-column functions** into a local before the loop (e.g.
  `moist_compressible.jl:538`, `primitive_equations.jl:324`).

**Slot indices.** If the Springsteel `vars` typing has NOT been done, add `::Int64` assertions at
the `vars[...]` lookups at the top of each per-column function (e.g.
`p_index = vars["p"]::Int64`, `src/moist_compressible.jl:726-733`, `:594-598`). This is what makes
`view(grid.physical, colstart:colend, p_index, 1)` type-stable. **If Springsteel has typed
`vars::Dict{String,Int64}`, skip this entirely — nothing is needed.**

**Re-measure** with the script in §7a. Expect: `::Any` in the `diffusion_timestep_mc` IR → 0, and
per-column bytes down 1–2 orders of magnitude.

### Phase 2 — Remaining temporaries (optimization; GATED on Phase 1's numbers)

Do this **only if Phase 1's measured allocations still warrant it.** Scoped to
`src/moist_compressible.jl` and `src/primitive_equations.jl` — **all other equation sets untouched.**

1. **Hoist the loop-invariant reference-column copies.** `ref_pressure(refstate)[:,1]`,
   `ref_rho_d(...)[:,1]`, etc. — 11 of them at `moist_compressible.jl:342-352`, 3 at `:736-738`,
   8 at `:611-619`, and ~12 in *each* PE variant (`primitive_equations.jl:89-103`, `:432-446`,
   `:758-763`, `:1106-1118`, `:1460-1474`, `:1846-1856`). Each `[:,j]` on a `Matrix` field is a
   **fresh `kDim` copy, per column, per timestep** — and they never change over the whole run.
   Cache them as views on the `ModelTile` at construction. **Highest payoff, lowest risk.**
2. **Preallocated scratch for the broadcast temporaries** (~40 per column in
   `moist_compressible_XZ`; 60–80 in `primitive_equation_XZ`). Add a full-tile
   `scratch::Matrix{Float64}` to `ModelTile` (same shape as `expdot_n`) and take
   `view(scratch, colstart:colend, j)` in each equation set. **Thread-safe by construction:** the
   `@threads` loop at `semiimplicit.jl:583` hands each column a *disjoint* `colstart:colend` slice,
   so this needs no `threadid()` (which is unreliable under dynamic scheduling in Julia ≥1.7) — and
   it matches the convention the `expdot_*` arrays already use.
3. **The 22 `deepcopy(kbasis.data[...])` sites** across `src/` each create a scratch spline/Chebyshev
   column *per column, per timestep* (`moist_compressible.jl:527, 645, 656, 790`;
   `primitive_equations.jl:212, 226, 522, 536, 578, 871, 885, 928, 1233, 1247, 1291, 1587, 1596`;
   `semiimplicit.jl:781, 790, 862, 871, 947, 957, 1024, 1033, 1140, 1253`). Replace with a reusable
   scratch column — the code already documents reusing one column across multiple solves
   (`moist_compressible.jl:788-790`), since the BCs live in the factorization, not the column.
4. Also worth an `@allocated` check: the `@turbo expdot[colstart:colend,1] .= @. …` sites
   (`moist_compressible.jl:445, 452, 456, 462, 468, 471, 502, 507`). A plain `A[r,j] .= x` lowers
   through `Base.dotview` and does not copy, but `@turbo` on a sliced broadcast-assign target may
   materialize the LHS. Verify before assuming they are free.

### Phase 3 — Static equation-set dispatch (small)

`physical_model` (`src/semiimplicit.jl:651-657`) does
`getfield(Scythe, Symbol(mtile.model.equation_set))` and calls the result — a fully dynamic call,
**per column, per timestep**. Measured at only **736 B/column**, so this is low priority. Resolve the
function **once** in `createModelTile` and store it, or use `Val`-dispatch. It must cover all **26**
equation-set entry points (list in §5) — or keep the current `getfield` path as a fallback so no
equation set can regress.

### Phase 4 — Stop thread oversubscription (independent; do it any time)

In `benchmarks/common/harness.jl` / each `benchmarks/*.jl:24`:

```julia
nthreads = max(1, Sys.CPU_THREADS ÷ opts.workers)
addprocs(opts.workers, exeflags="--threads=$(nthreads)")
```

Useful on its own, regardless of Phases 1–3.

---

## 5. Blast radius — other equation sets

**Phase 1 is purely type-level and changes no semantics.** There are **26 equation-set entry points**
with the `(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)` signature, all reached
through the same `createModelTile → advance_column → physical_model` path, and all of them read
`mtile.<field>` identically. **None of them needs a single line changed**, and all of them get the
speedup for free — including **`Twoway_PV_mixing`** (`src/shallowWaterModels.jl:268`), which has
active users.

The 26: `LinearAdvection1D`, `LinearAdvectionRZ`, `LinearAdvectionRL`, `LinearAdvectionRLZ`,
`Euler_test`, `BF02_test`, `BF02_test_alt`, `rainfall_test` (`src/testModels.jl`);
`primitive_equation_XZ`, `_XZ_rhod`, `_XZ_rhod_pd`, `_XZ_sigma`, `_RZ`, `_cylindrical`
(`src/primitive_equations.jl`); `moist_compressible_XZ` (`src/moist_compressible.jl`);
`Oneway_ShallowWater_Slab`, `Twoway_ShallowWater_Slab`, **`Twoway_PV_mixing`**,
`LinearShallowWater1D`, `LinearShallowWaterRL`, `ShallowWaterRL`,
`Oneway_ShallowWater_HeightResolvedBL`, `Oneway_ShallowWater_Slab_Uniform_Flow`
(`src/shallowWaterModels.jl`); `Kepert2017_HeightResolvedTCBL`, `RLZ_HeightResolvedBL`
(`src/tcblModels.jl`). (Three stale 3-arg functions in `tcblModels.jl` —
`Williams2013_slabTCBL`, `RL_SlabTCBL`, `Kepert2017_TCBL` — would `MethodError` if ever named.)

**Phase 2 is the only per-equation-set work**, and it is scoped to `mc` + PE. The shallow-water,
TCBL and test sets are left alone.

**Known coverage gap:** `primitive_equation_XZ` has **no unit test** — it is only exercised by
`benchmarks/bf02_moist.jl` and `benchmarks/mass_error_diagnostic.jl`.

---

## 6. Verification

This is a typing/allocation refactor. **Any change in numerical output is a bug.**

1. **New `test/test_allocations.jl`** (add to `test/runtests.jl`) — the guard that stops the abstract
   types creeping back:
   - `@test isconcretetype(fieldtype(ModelTile, :tile))` — and the same for `var_np1`, `h_matrix`,
     `tilepoints`, `haloReceiveBuffer`.
   - Cap `@allocated moist_compressible_XZ(mtile, 1, kDim, 2)` and
     `@allocated diffusion_timestep_mc(...)` below a threshold.
2. **Full suite**: baseline **4328 passing, 0 failures** at `aedce4d`. Must stay ≥4328 with zero
   failures. `test/test_moist_compressible.jl` contains *bit-preservation* testsets ("resting cloudy
   base preserved" `:237`, "resting dry base is untouched by diffusion" `:461`) that will catch a
   copy→view semantics slip immediately.
3. **Other equation sets explicitly guarded**: run `test_pv_mixing_floor.jl`,
   `test_oneway_sw_slab.jl`, `test_bf02_restoration.jl`, `test_partial_density.jl`,
   `test_sigma_entropy.jl`. **Additionally**, diff a short `Twoway_PV_mixing` run's output fields
   before vs after — must be bit-identical.
4. **Benchmark control**: `bf02_dry --mode quick --stage mc` on `--grid rz` *and* `--grid rirk`;
   diagnostics must be **byte-identical** to a pre-change run. (This is the same control used for the
   reference-state dedup in `aedce4d`, where it correctly showed zero change.)
5. **The actual bug**: run `straka93 --mode quick --stage mc --grid rirk` to completion (t = 900 s)
   **2–3 times** — the crash is intermittent. Confirm no death, and that the `Allocations:` line has
   dropped by orders of magnitude from 12.6 × 10⁹.
6. **Report memory AND speed** as a before/after table: `Allocations:` count, `GC:` count, and
   wall-clock / steps-per-second, for each of the three baseline runs in §3.
7. **Do NOT run `--mode full`** — it is slow and the user runs it manually.

---

## 7. Reproduction scripts

### 7a. Per-column allocations + `::Any` count

```julia
using Scythe, Springsteel, SparseArrays
using Scythe: createModelTile, physical_model, diffusion_timestep_mc, moist_compressible_XZ

vars = ["p","rho_d","rho_t","u","w","E_t","Q_ss","rho_r"]
sbc = Dict(v=>NeumannBC() for v in vars)
bc_side = merge(sbc, Dict("u"=>DirichletBC())); bc_tb = merge(sbc, Dict("w"=>DirichletBC()))
gp = GridParameters(; geometry="RiRk", iMin=0.0, iMax=25.6e3, num_cells_i=16,
    kMin=0.0, kMax=6.4e3, num_cells_k=8, BCL=bc_side, BCR=bc_side, BCB=bc_tb, BCT=bc_tb,
    vars=Dict(v=>i for (i,v) in enumerate(vars)))
outdir = mktempdir()
model = ModelParameters(ts=0.0625, equation_set="moist_compressible_XZ",
    output_dir=outdir*"/", ref_state_file=joinpath(outdir,"r.ref"), grid_params=gp,
    physical_params=Dict(:Khdiff=>75.0,:Kvdiff=>75.0,:Kv_mudiff=>0.0,:Prandtl=>1.0,
                         :tau_qss=>10.0,:alpha=>0.0,:z_damp=>12.8e3),
    options=Dict(:semiimplicit=>true,:exact_reference_state=>true,
                 :precipitation=>false,:vertical_mixing=>false))
patch = createGrid(model.grid_params)
g = Scythe.getGridpoints(patch); kD = model.grid_params.kDim; z = g[1:kD,2]
ex = @. 1.0-(Scythe.gravity*z)/(Scythe.Cpd*300.0)
p = @. 100000.0*ex^(Scythe.Cpd/Scythe.Rd); T = 300.0 .* ex
Scythe.write_exact_ref_mc(model.ref_state_file, z, p, p./(Scythe.Rd .* T), zeros(kD), zeros(kD))
mtile = createModelTile(patch, patch, model, spzeros(1,1))

physical_model(mtile,1,kD,2); moist_compressible_XZ(mtile,1,kD,2); diffusion_timestep_mc(mtile,1,kD,2)
println("physical_model        : ", @allocated(physical_model(mtile,1,kD,2)), " B")
println("moist_compressible_XZ : ", @allocated(moist_compressible_XZ(mtile,1,kD,2)), " B")
println("diffusion_timestep_mc : ", @allocated(diffusion_timestep_mc(mtile,1,kD,2)), " B")

ct = code_typed(diffusion_timestep_mc, (typeof(mtile), Int64, Int64, Int64); optimize=true)[1]
println("::Any in optimized IR : ", length(collect(eachmatch(r"::Any", string(ct[1])))))
```

Reference values **before** the refactor: 130800 / 130064 / 25952 bytes, 113 `::Any`.

### 7b. The isolation table (pure Julia, no Scythe)

```julia
struct TileBad;  var_np1::Array{Float64};  params::Dict; end                    # today
struct TileGood; var_np1::Matrix{Float64}; params::Dict{Symbol,Float64}; end    # target

function hot(t, pbar, K)                       # slice copies
    p  = t.var_np1[1:24, 1] .+ pbar
    rd = t.var_np1[1:24, 2] .+ pbar
    T  = @. p / (287.04 * rd)
    return sum(T) * K
end
function hot_views(t, pbar, K)                 # views, no copies
    p = @view t.var_np1[1:24, 1]; rd = @view t.var_np1[1:24, 2]
    s = 0.0
    @inbounds for i in eachindex(p)
        s += (p[i] + pbar[i]) / (287.04 * (rd[i] + pbar[i]))
    end
    return s * K
end

A = rand(240, 8) .+ 1.0; pbar = rand(24) .+ 1.0
bad  = TileBad(A,  Dict{Any,Any}(:K=>75.0))
good = TileGood(A, Dict{Symbol,Float64}(:K=>75.0))
for f in (hot, hot_views), (name, t) in (("abstract", bad), ("concrete", good))
    K = t.params[:K]; f(t, pbar, K)
    println(rpad(string(nameof(f)),10), " | ", rpad(name,9), " -> ", @allocated(f(t, pbar, K)), " B")
end
```

Expected: `hot`/abstract 1408, `hot`/concrete 1280, `hot_views`/abstract **4368**,
`hot_views`/concrete **0**.

### 7c. Concrete types of every `ModelTile` field

Build `mtile` as in §7a for `geometry="RiRk"` and `geometry="RZ"`, then print
`typeof(mtile.tile)`, `typeof(mtile.h_matrix)`, `typeof(mtile.diffusion_matrix)`,
`unique(typeof.(values(mtile.mc_diffusion_matrices)))`, `typeof(mtile.var_np1)`,
`typeof(mtile.tilepoints)`, `typeof(mtile.haloReceiveBuffer)`,
`typeof(mtile.model.grid_params.vars)`.

Measured 2026-07-12:

| field | RiRk | RZ |
|---|---|---|
| `tile` | `RiRk_Grid` | `RZ_Grid` |
| `h_matrix` | `LU{Float64,Matrix{Float64},Vector{Int64}}` | same |
| `diffusion_matrix` | `BunchKaufman{...}` | `LU{...}` |
| `mc_diffusion_matrices` values | `BunchKaufman{...}` | `LU{...}` |
| `var_np1`, `tilepoints` | `Matrix{Float64}` | same |
| `haloReceiveBuffer` | `Vector{Float64}` | same |
| `grid_params.vars` | `Dict{String,Int64}` | same (**already concrete at runtime; only the declaration is abstract**) |

---

## 8. Risks

- **`Dict{Symbol,Float64}` for `physical_params`** would reject an integer value (`:K => 75`).
  Convert in the inner constructor rather than tightening the keyword type; grep benchmarks and
  tests for integer-valued physical params first.
- **Some `var_np1[colstart:colend, i]` slices are deliberate copies**, mutated before being written
  back — see the "Predictors (copies)" comment at `src/moist_compressible.jl:602` and the loop at
  `:625-643`. Converting *those* to views in Phase 2 would silently change results. Use scratch
  buffers there, not views. The bit-preservation tests (§6.2) are the backstop.
- **`createModelTile`'s return type becomes non-inferable at its call site** once `ModelTile` is
  parameterized. This is fine — it runs once per tile, not per timestep, and is already outside the
  hot loop.
- **The GC crash is arguably a Julia bug** — a correct program should not segfault inside
  `gc_sweep_pool`. Reducing allocations ~170× should make it unreachable. If a crash *survives*
  Phase 1, escalate: retest with `--threads=1` and with `JULIA_NUM_GC_THREADS=1` to confirm the
  threaded-GC interaction, and consider filing upstream.
