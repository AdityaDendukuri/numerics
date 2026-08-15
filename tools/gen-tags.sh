#!/usr/bin/env sh
set -eu

if ! command -v ctags >/dev/null 2>&1; then
  echo "ctags not found. Install Universal Ctags to generate tags." >&2
  exit 1
fi

ctags \
  -R \
  --languages=C++ \
  --exclude=build \
  --exclude=build-* \
  --exclude=_deps \
  --exclude=.git \
  -f tags \
  include src tests benchmarks
