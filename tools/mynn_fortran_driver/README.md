# MYNN-EDMF / surface-layer Fortran reference drivers

Two standalone Fortran programs over **verbatim** ccpp-physics sources. `ref_driver.f90`
(the original, below) runs the MYNN-EDMF boundary layer; `sfc_ref_driver.f90` (stage S1b,
the last section of this file) prints the GFDL/HWRF v7 sea-surface roughness fits.

## MYNN-EDMF driver (`ref_driver.f90`)

Standalone Fortran program that runs the **verbatim** ccpp-physics MYNN-EDMF boundary-layer
scheme (`module_bl_mynn.F90`, `bl_mynn_common.f90`) on single columns and prints, at full
double precision, every quantity the pure-Julia port (`src/mynn_closure.jl`,
`src/mynn_edmf.jl`) is checked against. The checked-in `ref_driver_output_r8.txt` is what
`test/test_mynn_closure.jl` / `test_mynn_edmf.jl` parse (via
`test/reference/mynn_fortran_refs.jl`); there is **no Fortran at test time**. Plan:
`~/.claude/plans/in-the-last-2-enumerated-crane.md`, stage S0.

Source provenance: `~/Development/ccpp-physics` at `72570a3f` (ufs/dev, 2026-08-14;
`module_bl_mynn.F90` last touched 2026-06-17, header changelog ends at v4.5.2). Both files
are unmodified copies, Apache-2.0 header retained. `module_bl_mynn.F90` has no `private`
statement, so every routine is callable from the driver — no patch was needed (ISHMAEL's
driver had to widen a `public` list).

### Files

- `module_bl_mynn.F90`, `bl_mynn_common.f90` — verbatim copies.
- `stub_machine.f90` — stand-in for `physics/hooks/machine.F`: just `kind_phys = 8`.
- `ref_driver.f90` — the program (two modes, below).
- `run.sh` — builds both precision variants and runs them.
- `columns/` — inputs, written by `tools/mynn_dump_columns.jl` (`%.17g`, read back
  list-directed, so the round trip is exact):
  - `constants.txt` — the 14 host constants `bl_mynn_common` takes from the dycore
    (`cp cpv cliq cice p608 ep_2 grav karman t0c rcp r_d r_v xlf xlv`), from
    `Springsteel.Thermodynamics`, so the Fortran and the Julia `MYNNConstants` agree bitwise.
    Derived constants (`xls, rvovrd, ep_3, gtr, rk, tv0, tv1, xlscp, xlvcp, g_inv`) are formed
    exactly as `mynnedmf_wrapper_init` forms them and printed in the `## constants` block.
  - `case1_rest` — humidified Dunion sounding at rest, no fluxes (inertness reference).
  - `case2_o01_sea` — Dunion + `u = 10 tanh(z/150 m)` over SST 301.15 K with Scythe's own
    bulk surface layer (Komori `Cd`, `Ck = 1e-3`, `U_min = 2`); replaced by a column from
    the `ocean_warm_bubble` control run at S1.
  - `case3_tc_rmw` — the nest-1 column of maximum lowest-level tangential wind from
    `tc/output/tc_rad_24h_full` at 86400 s, rebuilt from the mish CSV + `tc_exact.ref` with
    the `tc_postprocess.jl` conventions (see the `.meta`). The 24-h vortex is weak at the
    lowest level (8 m/s), which is why case 5 exists.
  - `case4_convective` — synthetic dry-neutral mixed layer under a 3 K inversion, heated
    surface (`hfx = 200 W/m^2`): activates `DMP_mf`.
  - `case5_highwind` — Dunion + `u = 50 tanh(z/300 m)` over SST 302.65 K: the hurricane-force
    surface layer (Komori `Cd` capped at 2.55e-3), `dx = 3 km`.
  - `<case>.meta` — provenance (not read by Fortran).
  All columns are on a 50-cell (500 m) RiRk mish, `n = 150`, `z1 = 56.35 m`, with MYNN's
  layer thickness `dz` from faces at the midpoints between mish points plus `z = 0` and
  `z_top` (the radiation convention, `src/radiation.jl` D4). Column format: line 1 `n`;
  line 2 `ps ts qsfc ust hfx qfx wspd znt xland dx rmol delt`; then `n` lines
  `z dz u v w T th exner p rho sqv sqc sqi`. `sqv` is specific humidity `rho_v/rho_t` and
  `rho` the moist density, the wrapper's conventions; `ts` is `T_sfc/exner(1)`.
- `blend_ref_driver.f90` — a second, tiny program for the pure functions the main driver
  never prints directly: `esat_blend`, `qsat_blend`, `xl_blend` (only reachable from
  `mym_condensation`, whose *outputs* are what mode B dumps), the **stable** branch of
  `phim`/`phih` (every case here has `rmol <= 0`, so only the unstable branch is
  exercised), and `boulac_length0` on a synthetic 20-level column (CASE 2 of `mym_length`
  never calls it). Same output format; its numbers are transcribed by hand into
  `test/test_mynn_closure.jl`, so it is not run at test time and `run.sh` does not build
  it. To regenerate, after `run.sh` has built `build/r8`:
  ```sh
  cd build/r8 && /opt/homebrew/bin/gfortran -ffree-line-length-none -O0 -ffp-contract=off \
      -fdefault-real-8 -fdefault-double-8 stub_machine.o bl_mynn_common.o \
      module_bl_mynn.o ../../blend_ref_driver.f90 -o blend_ref_driver && ./blend_ref_driver
  ```
- `moisture_ref_driver.f90` — a third tiny program, for `moisture_check` (:5133-5220).
  The main driver prints none of its arguments, and every reference column is
  non-negative everywhere, so in `ref_driver_output_r8.txt` that routine only ever runs
  on its NO-OP path — its correction path (condense vapour into a negative condensate,
  borrow from the layer below, then redistribute the borrow over the column) has no
  coverage at all, and it is exactly the path a Scythe column with negative water takes.
  This program calls it directly on two synthetic 6-level columns, one that fires the
  condensation correction and a single-layer borrow and one that also fires the
  column-wide redistribution (:5198-5216). Same output format; its numbers are
  transcribed by hand into `test/test_mynn_closure.jl` (the `MC_*` constants), so it is
  not run at test time and `run.sh` does not build it. To regenerate, after `run.sh`
  has built `build/r8`:
  ```sh
  cd build/r8 && /opt/homebrew/bin/gfortran -ffree-line-length-none -O0 -ffp-contract=off \
      -fdefault-real-8 -fdefault-double-8 stub_machine.o bl_mynn_common.o \
      module_bl_mynn.o ../../moisture_ref_driver.f90 -o moisture_ref_driver && ./moisture_ref_driver
  ```
- `edmf_ref_driver.f90` — a fourth tiny program (stage S3), for the parts of `DMP_mf`
  that the five reference columns never reach. In `ref_driver_output_r8.txt` the
  plumes NEVER condense (`edmf_qc` is identically zero at every level of every case at
  both steps), so the Chaboureau-Bechtold shallow-cumulus block (:6631-6766, which
  overwrites `vt`, `vq`, `cldfra_bl1d` and `qc_bl1d`) and the `maxqc >= 1e-8`
  moist-plume branch of the `maxmf` sign (:6771-6775) have no coverage at all; nor
  does any `landsea < 1.5` (LAND) branch, all five columns being water. This program
  calls `DMP_mf` directly, twice, on a synthetic column built from
  `columns/case4_convective.txt` (vapour x1.5, `pblh = 1500 m`, `flt = 0.3`,
  `flq = 3e-4`, `ust = 0.4`) whose plumes reach 22 interfaces and saturate at 19 of
  them — once with `landsea = 2` and once with `landsea = 1`. Output blocks use the
  main driver's format, so `test/reference/mynn_fortran_refs.jl` parses them
  unchanged (cases `spot_moist_water` / `spot_moist_land`, closure 2.50, mode B,
  step 1); it is CHECKED IN as `edmf_ref_driver_output.txt` (8190 lines) rather than
  transcribed, and `run.sh` does not build it. To regenerate, after `run.sh` has
  built `build/r8`:
  ```sh
  cd build/r8 && /opt/homebrew/bin/gfortran -ffree-line-length-none -O0 -ffp-contract=off \
      -fdefault-real-8 -fdefault-double-8 stub_machine.o bl_mynn_common.o \
      module_bl_mynn.o ../../edmf_ref_driver.f90 -o edmf_ref_driver && \
      ./edmf_ref_driver > ../../edmf_ref_driver_output.txt
  ```
- `edmf_ref_driver_output.txt` — its reference (checked in, ~250 kB).
- `ref_driver_output_r8.txt` — the reference (checked in, 5.9 MB, ~184k lines).
- `ref_driver_output_native.txt` — UFS-precision run (gitignored; regenerate with `run.sh`).

### The two modes and the gate

For every case the driver runs, at `delt = 20 s`, 30 steps with the column state FROZEN
(`mynn_bl_driver` never updates `u/th/q` itself):

- **mode A** — `mynn_bl_driver` called as a host would (`initflag = 1` on the first call),
  closure 2.5 and 2.6. End-of-step state, tendencies and EDMF arrays printed at steps 1
  and 30.
- **mode B** — the per-column call sequence of `mynn_bl_driver` replayed in the driver
  (`GET_PBLH -> SCALE_AWARE -> mym_initialize`, then per step `GET_PBLH -> SCALE_AWARE ->
  surface fluxes/rmol/phim/phih -> mym_condensation -> DMP_mf -> [mym_level2 standalone] ->
  mym_turbulence -> mym_predict -> diss_heat -> mynn_tendencies -> retrieve_exchange_coeffs`),
  closure 2.5, printing the output of EVERY routine at steps 1 and 30 (and the init block).
- **GATE** — mode B's step-30 state must equal mode A's **bitwise**. `run.sh` prints one
  `GATE PASS/FAIL` line per case; all five pass. This is what makes the per-routine
  parity tests meaningful: the intermediates come from a replication proven faithful.

Output format: `## <case> closure=<c> mode=<A|B> step=<s> <name> n=<len>` followed by
`<len>` lines `k value` (`ES25.17E3`, so 3-digit exponents keep their `E`). Scalars have
`n=1`; integers are printed as reals.

### Build flags

`run.sh` builds twice (gfortran 15.2, `/opt/homebrew/bin/gfortran`):

- `build/r8` (**the reference**): `-fdefault-real-8 -fdefault-double-8 -O0 -ffp-contract=off`.
  The module mixes `real(kind_phys)` (double) with bare `real ::` locals and default-real
  literals (e.g. `mym_turbulence` :2678, `mym_condensation` :3652, every closure constant
  in :272-310); promoting them makes the whole computation double, which is what a Julia
  port computes. `-ffp-contract=off` because gfortran contracts `a*b+c` into FMA on arm64
  and Julia does not — without it bitwise agreement is unreachable. `DOUBLE PRECISION`
  locals stay 8-byte (`-fdefault-double-8`).
- `build/native` (informational): no promotion flags = UFS production semantics. Relative
  differences from the r8 run at step 30 are ~1e-7 (qke, el, exch_h) to ~1e-5 (rthblten),
  i.e. the 24-bit truncation of the bare-real locals and literals. Not tested against.

### Fortran behaviours the port must know about (found while building the gate)

1. **`mynn_tendencies` and `moisture_check` write their input columns.** `thl, sqw, sqv,
   sqc, sqi` are `intent(inout)` (:4092) and `moisture_check` corrects `thl` in place.
   The driver survives because it re-gathers every column from the frozen 3-D host arrays
   on every call; a replay that passes the host column directly lets the column EVOLVE.
   (This was the one gate failure: fixed by per-step working copies in mode B.)
2. **`ts` is divided by `exner(1)` twice**: the wrapper passes `ts = tsurf/exner(i,1)`
   (:659, "theta") and the driver forms `th_sfc = ts/ex1(kts)` again (:1071). `fltv` sees
   `theta_sfc/exner`, a ~1 % error at the surface. Reproduced verbatim; a `:mynn_fidelity`
   item for later.
3. **Drag is `ust**2/wspd`** (:4182): a truly resting column (`wspd = 0`) divides 0/0, so
   case 1 carries `wspd = 0.1` with `ust = 0`.
4. **`ust = 0` makes `mym_initialize` produce NaN** in `tsq(kts), qsq(kts), cov(kts)`
   (`phm*(flt/ust)**2`, :1585). They are overwritten by `mym_predict` at step 1 and never
   reach the state; the three `init_*` blocks of case 1 carry the NaN at `k = 1` and the
   parity test must compare them with `isequal`. The Julia port reproduces the NaN
   (0/0) unless it deviates on purpose.
5. **Snow is not mixed**: the driver passes a zero column (`kzero`) as `qs`/`sqs` to
   `mynn_tendencies` (:1242, :1246); `sqw` excludes snow (:1005).
6. **`vt, vq, sgm` are automatic arrays zeroed only in the init block**; `mym_condensation`
   (cloudpdf 2) writes every level so their entry values never matter, but mode B carries
   them anyway to mirror the driver exactly.
7. **The wrapper clips `hfx` to [-500, 1200] W/m^2 and `qfx` to [-2e-4, 5e-4]** (:630-633).
   The driver does not apply these; the port makes them counters (plan D5).
8. `closure = 2.6` exercises only the prognostic-`qsq` branch of `mym_predict` (:3392);
   `mym_turbulence` has `closure >= 3.0` branches only. Both closures are dumped in mode A.
9. **`mym_initialize` is passed `sqv`, not `sqw`, as its total-water argument `qw`**
   (:817, and the driver copies that). Cloud and ice are therefore excluded from `q_w`
   for the cold start only — every later `mym_level2`/`mym_turbulence` call gets the real
   `sqw`. Passing `sqw` at init still reproduces `init_el` and `init_qke` (the `qke` floor
   masks the difference in `pdk`) but moves `init_sh`/`init_sm` by ~3-8 % and
   `init_qsq`/`init_cov` by orders of magnitude, so it is easy to get wrong and hard to
   see. Found while writing `test/test_mynn_closure.jl`.

10. **`MAX` with a NaN quenches it in gfortran, and Julia's `max` propagates it.**
    Item 4's NaN in `tsq/qsq/cov(kts)` reaches exactly two `MAX` calls downstream —
    `mym_condensation` :3849 `r3sq = max(qsq(k), 0.)` and `mym_predict` :3427
    `qsq(k) = MAX(x(k), 1e-17)` — and the reference build returns the OTHER argument
    (0. and 1e-17), so the Fortran column stays finite. The Julia port keeps Julia's
    NaN-propagating `max`, so case 1 (the only `ust = 0` column) diverges at `k = 1`
    of `mym_condensation`'s outputs and across the whole closure-2.6 `qsq`. This is
    the port's ONE known divergence; it is documented at the top of part 2 of
    `src/mynn_closure.jl` and pinned by its own testset in
    `test/test_mynn_closure.jl`. Any column with `ust > 0` is unaffected.


### Behaviours found while porting `DMP_mf` (stage S3)

11. **`edmf_qc` is identically zero in the whole reference.** Case 4's plumes reach
    only 653 m and `condensation_edmf` forces `QC = 0` below 100 m (:6867), so no
    plume ever saturates. Consequences: `DMP_mf` never modifies `vt`, `vq`,
    `cldfra_bl` or `qc_bl` (the Chaboureau-Bechtold block :6631-6766 is gated on
    `0.5*(edmf_qc(k)+edmf_qc(k-1)) > 0`), so the printed post-`DMP_mf` blocks for
    EVERY case are still `mym_condensation`'s output; and `maxmf` is always
    sign-flipped negative by the dry-plume rule (:6771-6775). `edmf_ref_driver.f90`
    exists to cover what that leaves untested.

12. **The activation gate is not the whole story.** `fltv2 > 0.002 .AND. maxwidth >
    minwidth .AND. superadiabatic` (:6039) PASSES for case 2, case 4 and case 5 at
    step 1, yet only case 4 produces flux. For cases 2 and 5 all eight plumes fail to
    leave the surface interface, which trips `IF (k==kts+1 .AND. Wn == 0.) NUP2 = 0`
    (:6288); `nup2` stays 0 for the rest of the column loop even though the later
    plumes are still integrated and could still raise `ktop`, so `IF (nup2 > 0)`
    (:6389) skips the entire flux calculation. `maxwidth` still comes back nonzero
    (545 m and 590 m) because :6035 zeroes it only when the WIDTH criterion is what
    failed — which is how cases 1 and 3 (gate failed on `fltv2`/`superadiabatic` and
    on `maxwidth`, respectively) differ from cases 2 and 5 in the dump.

13. **`k50` (:5948) is read before it is necessarily assigned.** It is written only
    inside `if (ZW(k)<=50.)` in the taper loop, and read at :5977 as
    `do k=1,max(1,k50-1)`. `zw(kts) = 0` and the loop's `exit` cannot fire at
    `k = kts`, so it is always assigned at least once — but it is a genuine
    uninitialised-variable read as written.

14. **`dzp` (:6316-6318) can be read undefined.** It is assigned in the
    `Wn <= 0 .and. overshoot == 0` branch only when `THVk - THVkm1 > 0`, and read
    three lines later. It feeds only the `envm_*` arrays, which nothing reads unless
    `env_subs` is true, so it is dead on the shipped configuration.

15. **The subsidence block reads one past the end of `rhoz`.** With `env_subs`
    true, `sub_thl(kts)` (:6586) — and likewise `sub_sqv(kts)`, `sub_u(kts)`,
    `sub_v(kts)` — divides by `rhoz(k)` where `k` is the loop variable LEFT OVER
    from the `DO k=kts,kte` transform loop above, i.e. `k = kte+1`, whereas `rhoz` is
    `dimension(kts:kte)`. Unreachable in the shipped code (`env_subs = .false.`,
    :337), which is presumably why it has survived.

16. **Declared and never used, in `DMP_mf` alone:** `ENTf`/`ENTi` (:5793-5794, the
    stochastic-entrainment leftovers), `s_aw2` (:5779), `UPQV` (zeroed at :5877 and
    never written or read), `ERF` (:5824), `wlv` (:6122), `qsl` (:6636), `Ac_mf`/
    `Ac_strat`/`qc_mf` (:5840), and the `sgm`, `qc_bl1D_old` and `cldfra_bl1D_old`
    dummy arguments, plus the `F_QC`/`F_QI`/`F_QN*` `optional` flags — none of which
    the body references. The five number-concentration plume matrices `UPQNC`,
    `UPQNI`, `UPQNWFA`, `UPQNIFA`, `UPQNBCA` are computed unconditionally but their
    only source is the zero columns the driver passes, so they and their `s_awqn*`
    sums are zero everywhere.

### Running

```sh
julia --project=. tools/mynn_dump_columns.jl            # rewrites columns/ (needs the TC run for case 3)
tools/mynn_fortran_driver/run.sh                         # builds both, runs both, prints the gate lines
```

Re-running is only needed when a case changes; the checked-in `ref_driver_output_r8.txt`
is static and the test suite needs neither gfortran nor the TC run.

## Surface-roughness driver (`sfc_ref_driver.f90`, stage S1b)

Prints `znot_m_v7(U10)` and `znot_t_v7(U10)` — the HWRF/HAFS sea-surface roughness fits
(Bin Liu, NOAA/NCEP/EMC 2018) — at the 17 wind speeds `test/test_surface_layer.jl` checks
the Julia port in `src/mc_surface_layer.jl` against.

- `module_sf_exchcoef.f90` — **verbatim** copy of
  `ccpp-physics/physics/SFC_Layer/GFDL/module_sf_exchcoef.f90` at the same `72570a3f`
  (Apache-2.0 header retained; the file itself was last touched at `b7e3e94e`, 2025-06-05).
  Unmodified: the module has no `private` statement and no `use`, so it compiles alone.
- `sfc_ref_driver.f90` — the program. Output: a `## gfdl_v7 znot uref z0m z0t` header line,
  then one `uref z0m z0t` line per wind speed in `ES25.17E3`.
- `run_sfc.sh` — builds and runs it, leaving `sfc_ref_driver_output.txt`.
- Build flags: `gfortran -fdefault-real-8 -ffp-contract=off -O0`. That module declares its
  arguments and locals as **bare `real`**, so without `-fdefault-real-8` the printed values
  are single precision and useless as a double-precision reference;
  `-ffp-contract=off` for the same FMA reason as the MYNN build above.
- The values are transcribed into `test/reference/gfdl_sfc_refs.jl` as Float64 literals, so
  the test suite needs no Fortran. The Julia port currently reproduces all 34 of them
  BITWISE (the test asserts rtol 1e-12, which is the honest bound: it depends on gfortran
  and Julia agreeing on `exp` and on the expansion of `uref**n`).

```sh
tools/mynn_fortran_driver/run_sfc.sh
```
