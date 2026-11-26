# Packaging

numerics is header-only, so every package is architecture-independent: headers plus the
CMake package, built from the GitHub release tarball of the version it names.

| Directory | Builds | Package |
| :--- | :--- | :--- |
| `arch/` | Arch Linux (`makepkg -si`; the AUR once registration reopens) | `numerics` |
| `debian/` | Debian, Ubuntu | `libnumerics-dev` |
| `rpm/` | Fedora, openSUSE | `numerics-devel` |
| `obs/` | All of the above on the openSUSE Build Service (`obs/stage.sh`) | |
| `test/` | A `find_package(numerics)` consumer, run by CI against each package | |

For a release, bump the version in `arch/PKGBUILD`, `rpm/numerics.spec`, `debian/changelog`,
`obs/_service` and `obs/numerics.dsc`, then tag. `.github/workflows/packaging.yml` builds and
tests every package on each tag.
