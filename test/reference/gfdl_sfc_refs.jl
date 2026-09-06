# GFDL/HWRF v7 sea-surface roughness reference values.
#
# Printed by `tools/mynn_fortran_driver/sfc_ref_driver.f90`, which calls a VERBATIM copy of
# `ccpp-physics/physics/SFC_Layer/GFDL/module_sf_exchcoef.f90` (Apache-2.0; the v7 fits are
# Bin Liu, NOAA/NCEP/EMC 2018), built with
#
#     gfortran -fdefault-real-8 -ffp-contract=off -O0
#
# (that module declares bare `real` arguments, so without the promotion the reference would
# be single precision) and formatted `ES25.17E3`. Regenerate with
# `tools/mynn_fortran_driver/run_sfc.sh`; transcribed here by hand so the test suite needs
# no Fortran. Checked by `test/test_surface_layer.jl` at rtol 1e-12 -- the residual is the
# difference between gfortran's binary expansion of `uref**n` plus its libm `exp`, and
# Julia's.
#
# Columns: U10 [m/s], z0m [m] (`znot_m_v7`), z0t [m] (`znot_t_v7`).

const GFDL_SFC_Z0_REFS = (
    (1.0, 5.993431392088234e-05, 0.00011),
    (2.0, 2.6084164512966597e-05, 0.00011),
    (5.0, 1.906476891108173e-05, 0.00011),
    (10.0, 0.00016614483657359012, 2.3759273207065736e-05),
    (15.0, 0.0008613156570502822, 6.308654343686339e-06),
    (20.0, 0.0022309564684849566, 3.5789553340564767e-06),
    (25.0, 0.0034949475166774417, 3.04623713258421e-06),
    (30.0, 0.003884842211730774, 3.3043317292574744e-06),
    (35.0, 0.003197391998918815, 5.023191793722271e-06),
    (40.0, 0.0019812175790551137, 1.1361514586241052e-05),
    (45.0, 0.0009831663325977204, 3.099688030284986e-05),
    (50.0, 0.00046692719467781, 7.533853397463652e-05),
    (55.0, 0.0003371427455376717, 0.0001013839682319495),
    (60.0, 0.0003371427455376717, 9.259285203561118e-05),
    (70.0, 0.0003371427455376717, 7.660912607541925e-05),
    (80.0, 0.0003371427455376717, 6.840803042777732e-05),
    (85.0, 0.0003371427455376717, 6.840803042788487e-05),
)
