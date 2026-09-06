# MYNN-EDMF Fortran reference driver

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

## Files

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
- `ref_driver_output_r8.txt` — the reference (checked in, 5.9 MB, ~184k lines).
- `ref_driver_output_native.txt` — UFS-precision run (gitignored; regenerate with `run.sh`).

## The two modes and the gate

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

## Build flags

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

## Fortran behaviours the port must know about (found while building the gate)

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

## Running

```sh
julia --project=. tools/mynn_dump_columns.jl            # rewrites columns/ (needs the TC run for case 3)
tools/mynn_fortran_driver/run.sh                         # builds both, runs both, prints the gate lines
```

Re-running is only needed when a case changes; the checked-in `ref_driver_output_r8.txt`
is static and the test suite needs neither gfortran nor the TC run.
