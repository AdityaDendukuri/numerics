# ODEs

Initial value problems: explicit, adaptive, symplectic and implicit integrators. The concepts are on the [Concepts](concepts.md) page.

## Integrators <ode/ode.hpp>
num::ode_euler num::ode_rk4 num::ode_rk45 num::ode_rk4_2nd num::ode_verlet num::ode_yoshida4 num::euler num::rk4 num::rk45 num::rk4_2nd num::verlet num::yoshida4

## Step ranges <ode/steps.hpp>
num::euler_steps num::rk4_steps num::rk45_steps num::rk4_2nd_steps num::verlet_steps num::yoshida4_steps num::basic_euler_steps num::basic_rk4_steps num::basic_rk45_steps num::basic_rk4_2nd_steps num::basic_verlet_steps num::basic_yoshida4_steps

## Implicit <ode/implicit.hpp>
num::ode::advance num::ode::implicit_params

## Types <ode/types.hpp>
num::ode_params num::ode_result num::ode_step num::step_end num::symplectic_result num::symplectic_step num::ode_rhs_fn num::accel_fn num::observer_fn num::symp_observer_fn

## Checks <ode/debug.hpp>
num::ode::debug::verify_order_of_accuracy num::ode::debug::verify_symplectic_2form
