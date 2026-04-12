/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "numerics", "index.html", [
    [ "Numerics Library", "index.html", "index" ],
    [ "Boltzmann Acceptance Table", "page_boltzmann_table.html", [
      [ "Motivation", "page_boltzmann_table.html#autotoc_md3", null ],
      [ "API", "page_boltzmann_table.html#autotoc_md5", null ],
      [ "Usage", "page_boltzmann_table.html#autotoc_md7", [
        [ "Direct replacement in Ising", "page_boltzmann_table.html#autotoc_md8", null ],
        [ "Building a flat lookup table", "page_boltzmann_table.html#autotoc_md9", null ]
      ] ]
    ] ],
    [ "Connected Components", "page_connected_components.html", [
      [ "Motivation", "page_connected_components.html#autotoc_md11", null ],
      [ "API", "page_connected_components.html#autotoc_md13", [
        [ "Labels", "page_connected_components.html#autotoc_md14", null ]
      ] ],
      [ "Algorithm", "page_connected_components.html#autotoc_md16", null ],
      [ "Usage – Ising nucleation", "page_connected_components.html#autotoc_md18", null ],
      [ "Generalizations", "page_connected_components.html#autotoc_md20", null ]
    ] ],
    [ "3D Field Types and Solvers", "page_fields.html", [
      [ "Types", "page_fields.html#autotoc_md37", [
        [ "ScalarField3D", "page_fields.html#autotoc_md38", null ],
        [ "VectorField3D", "page_fields.html#autotoc_md40", null ]
      ] ],
      [ "FieldSolver", "page_fields.html#autotoc_md42", null ],
      [ "MagneticSolver", "page_fields.html#autotoc_md44", null ],
      [ "Typical Workflow", "page_fields.html#autotoc_md46", null ],
      [ "Where Each Type Is Used", "page_fields.html#autotoc_md48", null ]
    ] ],
    [ "PBCLattice2D", "page_pbc_lattice.html", [
      [ "Motivation", "page_pbc_lattice.html#autotoc_md137", null ],
      [ "API", "page_pbc_lattice.html#autotoc_md139", null ],
      [ "Index Layout", "page_pbc_lattice.html#autotoc_md141", null ],
      [ "Usage", "page_pbc_lattice.html#autotoc_md143", null ]
    ] ],
    [ "PDE Module", "page_pde.html", [
      [ "Contents", "page_pde.html#autotoc_md145", null ],
      [ "Stencil operators (<tt>pde/stencil.hpp</tt>)", "page_pde.html#autotoc_md147", [
        [ "2D operators", "page_pde.html#autotoc_md148", null ],
        [ "2D grid utilities", "page_pde.html#autotoc_md149", null ],
        [ "2D fiber sweeps", "page_pde.html#autotoc_md150", null ],
        [ "3D operators", "page_pde.html#autotoc_md151", null ]
      ] ],
      [ "3D fields (<tt>pde/fields.hpp</tt>)", "page_pde.html#autotoc_md153", null ],
      [ "Crank-Nicolson ADI (<tt>pde/adi.hpp</tt>)", "page_pde.html#autotoc_md155", [
        [ "Sweep", "page_pde.html#autotoc_md156", null ],
        [ "Strang splitting (TDSE)", "page_pde.html#autotoc_md157", null ]
      ] ],
      [ "Implicit backward Euler (<tt>pde/diffusion.hpp</tt>)", "page_pde.html#autotoc_md159", null ],
      [ "Explicit diffusion (<tt>pde/stencil.hpp</tt>)", "page_pde.html#autotoc_md160", [
        [ "Typical usage (Navier-Stokes viscosity)", "page_pde.html#autotoc_md161", null ]
      ] ],
      [ "2D Poisson solver (<tt>pde/poisson.hpp</tt>)", "page_pde.html#sec_pde_poisson", null ]
    ] ],
    [ "2D Poisson Solver", "page_poisson.html", [
      [ "Problem formulation", "page_poisson.html#sec_poisson_problem", null ],
      [ "Eigenvalue decomposition", "page_poisson.html#sec_poisson_eigen", null ],
      [ "Algorithm (FD solver)", "page_poisson.html#sec_poisson_fd", null ],
      [ "Algorithm (spectral solver)", "page_poisson.html#sec_poisson_spectral", null ],
      [ "DST-I via complex FFT", "page_poisson.html#sec_poisson_dst", null ],
      [ "API reference", "page_poisson.html#sec_poisson_api", null ],
      [ "Worked example", "page_poisson.html#sec_poisson_example", null ]
    ] ],
    [ "SPH Kernels", "page_sph_kernel.html", [
      [ "Motivation", "page_sph_kernel.html#autotoc_md196", null ],
      [ "API", "page_sph_kernel.html#autotoc_md198", null ],
      [ "Kernels", "page_sph_kernel.html#autotoc_md200", [
        [ "Cubic Spline – density", "page_sph_kernel.html#autotoc_md201", null ],
        [ "Spiky Kernel – pressure gradient", "page_sph_kernel.html#autotoc_md202", null ]
      ] ],
      [ "Dimension specialization", "page_sph_kernel.html#autotoc_md204", null ],
      [ "Usage", "page_sph_kernel.html#autotoc_md206", null ],
      [ "Where Each Kernel Is Used", "page_sph_kernel.html#autotoc_md208", null ]
    ] ],
    [ "Stencil Higher-Order Functions", "page_stencil_hof.html", [
      [ "Motivation", "page_stencil_hof.html#autotoc_md210", null ],
      [ "2D Functions", "page_stencil_hof.html#autotoc_md212", [
        [ "laplacian_stencil_2d – Dirichlet boundaries", "page_stencil_hof.html#autotoc_md213", null ],
        [ "laplacian_stencil_2d_periodic – Periodic boundaries", "page_stencil_hof.html#autotoc_md215", null ],
        [ "col_fiber_sweep / row_fiber_sweep – ADI/CN sweeps", "page_stencil_hof.html#autotoc_md217", null ]
      ] ],
      [ "3D Functions", "page_stencil_hof.html#autotoc_md219", [
        [ "neg_laplacian_3d – 7-point stencil", "page_stencil_hof.html#autotoc_md220", null ],
        [ "gradient_3d", "page_stencil_hof.html#autotoc_md222", null ],
        [ "divergence_3d", "page_stencil_hof.html#autotoc_md224", null ],
        [ "curl_3d", "page_stencil_hof.html#autotoc_md226", null ]
      ] ],
      [ "Where Each Function Is Used", "page_stencil_hof.html#autotoc_md228", null ],
      [ "Adding a New App", "page_stencil_hof.html#autotoc_md230", null ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.html", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Concepts", "concepts.html", "concepts" ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", null ],
        [ "Typedefs", "functions_type.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Variables", "globals_vars.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"adi_8hpp.html",
"classnum_1_1CellList2D.html#a13fd593b2e5c3ef80c38056ba77e2751",
"classnum_1_1VerletList2D.html#a73660a818563a7ba6c2babba42d5dcf3",
"dir_fe92384f304924996d25ba21ed755d8c.html",
"math_8hpp.html#a2d62c88d7dff390703363e9444c42618",
"namespacenum.html#a8273696d8f563fdd52e64cc04ec9b54e",
"namespacenum_1_1spectral.html#aea71718dc75397727f9643146daa278c",
"page_week7.html#autotoc_md279",
"structnum_1_1BackwardEuler.html#aa078d229c0b0e5073df27e6ac708ba58",
"structnum_1_1RK45.html",
"structnum_1_1kernel_1_1subspace_1_1CallableOp.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';