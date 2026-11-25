#!/bin/sh
# Assemble the OBS package directory from the packaging files in this repository.
#
#   osc checkout home:<user> numerics       # once, after creating the package on OBS
#   packaging/obs/stage.sh home:<user>/numerics
#   cd home:<user>/numerics && osc addremove && osc commit -m "numerics <version>"
#
# OBS builds the spec for Fedora and openSUSE, the .dsc for Debian and Ubuntu, and the
# PKGBUILD for Arch, all from the tarball the _service downloads.
set -eu
dest=${1:?usage: stage.sh <osc package checkout>}
here=$(cd "$(dirname "$0")" && pwd)
pkg="$here/.."

cp "$here/_service" "$here/numerics.dsc" "$pkg/rpm/numerics.spec" "$pkg/arch/PKGBUILD" "$dest/"
# Debian files go in as one archive (debtransform). source/format is left out: OBS
# rebuilds the source package itself and the quilt format expects an orig tarball.
tar -czf "$dest/debian.tar.gz" -C "$pkg" --exclude debian/source debian
echo "Staged $(ls "$dest" | tr '\n' ' ')"
