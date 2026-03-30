# physics.cmake — intentionally empty.
#
# Physics code (SPH fluid, heat, EM field) has been decoupled from the
# numerics library and moved directly into each app that uses it:
#
#   apps/fluid_sim/    — 2D SPH + heat (fluid.cpp, spatial_hash.cpp, heat.cpp)
#   apps/fluid_sim_3d/ — 3D SPH        (fluid3d.cpp)
#   apps/em_demo/      — EM field       (field.cpp)
#
# Each app compiles its own physics sources and exposes them only locally.
# The numerics library (target: numerics) is purely numerical analysis.
