# Packaging

Linux packages for numerics. Each builds from the GitHub release tarball of the version it
names, with the optional backends (BLAS, LAPACK, OpenMP, FFTW, SuiteSparse, CUDA, MPI) and
the I/O target turned off. The Homebrew formula lives in its own tap,
`AdityaDendukuri/homebrew-numerics`.

| Directory | Builds | Package |
| :--- | :--- | :--- |
| `arch/` | Arch Linux, with `makepkg`; the AUR when registration reopens | `numerics` |
| `debian/` | Debian and Ubuntu | `libnumerics-dev` |
| `rpm/` | Fedora and openSUSE | `numerics-devel` |
| `obs/` | All of the above on the openSUSE Build Service | |
| `test/` | A consumer that runs `find_package(numerics 1.1)` against the installed package | |

`.github/workflows/packaging.yml` builds every package in its distribution's container,
installs it and runs `test/`, on each tag and on changes to this directory.

## Installing on Arch without the AUR

```sh
git clone https://github.com/AdityaDendukuri/numerics
cd numerics/packaging/arch && makepkg -si
```

## Publishing on the openSUSE Build Service

Once, create the package in your home project (`osc meta pkg -e home:<user> numerics`)
and enable the repositories to build for, such as Fedora, openSUSE Tumbleweed, Debian 12,
xUbuntu 24.04 and Arch. Then, for each release:

```sh
osc checkout home:<user> numerics
packaging/obs/stage.sh home:<user>/numerics
cd home:<user>/numerics && osc addremove && osc commit -m "numerics <version>"
```

## Releasing a new version

Bump the version in each of these, then tag:

- `arch/PKGBUILD`: `pkgver`, `pkgrel=1` and `sha256sums` (`curl -sL <tarball> | sha256sum`)
- `arch/.SRCINFO`: the same fields (`makepkg --printsrcinfo > .SRCINFO` on Arch)
- `rpm/numerics.spec`: `Version`, `Release`, and a `%changelog` entry
- `debian/changelog`: a new entry (`dch -v <version>-1`)
- `obs/_service` and `obs/numerics.dsc`: the version in the tarball path and fields
