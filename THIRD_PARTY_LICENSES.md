# Optional third-party backends

Numerics does not vendor the following software. These components are detected
at configuration time and dynamically linked when available.

## SuiteSparse KLU

- KLU: LGPL-2.1-or-later
- AMD: BSD-3-Clause
- COLAMD: BSD-3-Clause
- BTF and SuiteSparse_config: see the licenses shipped with the installed
  SuiteSparse distribution

KLU serves both real sparse LU and the complex shifted factorizations of `sparse_resolvent`.
The KLU backend can be disabled with `-DNUMERICS_USE_SUITESPARSE=OFF`. Numerics
does not enable or link ParU, METIS, or the complete SuiteSparse distribution.
Redistributors of binaries with KLU enabled must preserve the applicable
SuiteSparse notices and comply with the LGPL requirements for relinking or
replacement of the dynamically linked KLU library.

Upstream license collection:
https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/LICENSE.txt

## SuiteSparse UMFPACK (optional, off by default)

- UMFPACK: GPL-2.0-or-later

UMFPACK provides `umfpack_factorization` (real sparse LU) and `umfpack_shifted_lu` (complex
sI − A), each with ordinary and transposed solves. It is multifrontal with dense BLAS kernels and
factors high-fill matrices several times faster than KLU. Because it is GPL, it is enabled only
with `-DNUMERICS_USE_UMFPACK=ON`; numerics' own source stays MIT, and binaries built with
UMFPACK enabled are subject to the GPL when distributed. Numerics does not vendor UMFPACK.

