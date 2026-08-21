# Optional third-party backends

Numerics does not vendor the following software. These components are detected
at configuration time and dynamically linked when available.

## SuiteSparse KLU

- KLU: LGPL-2.1-or-later
- AMD: BSD-3-Clause
- COLAMD: BSD-3-Clause
- BTF and SuiteSparse_config: see the licenses shipped with the installed
  SuiteSparse distribution

The KLU backend can be disabled with `-DNUMERICS_USE_SUITESPARSE=OFF`. Numerics
does not enable or link ParU, METIS, or the complete SuiteSparse distribution.
Redistributors of binaries with KLU enabled must preserve the applicable
SuiteSparse notices and comply with the LGPL requirements for relinking or
replacement of the dynamically linked KLU library.

Upstream license collection:
https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/LICENSE.txt
