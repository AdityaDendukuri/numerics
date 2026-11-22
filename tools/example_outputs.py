#!/usr/bin/env python3
"""Run every example program and record what it prints, for the documentation.

    python3 tools/example_outputs.py --build <build-dir>

Each `examples/<name>.cpp` must already be built (`cmake --build <build-dir> --target
numerics_examples`). When it writes standard output, it goes to `examples/output/<name>.txt`,
which `refdoc.py` shows under the program on its reference page. Programs that save
`result.json` have that file recorded instead.

Plots are committed under `docs/assets/examples/`. This script runs without `--plot` and
discards gnuplot output, so refreshing JSON does not open windows or write image files.
"""

import argparse
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
OUTPUT = EXAMPLES / "output"

SHIM = """#!/bin/sh
# Discard plot commands during result recording.
script=$(cat)
case "$script" in
  *"set terminal dumb"*) printf '%s\\n' "$script" | grep -v 'pause mouse' | "{gnuplot}" ;;
  *) : ;;
esac
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=pathlib.Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    build = args.build.resolve()
    binaries = build / "examples"
    cache = build / "CMakeCache.txt"
    io_enabled = "NUMERICS_BUILD_IO:BOOL=OFF" not in cache.read_text() if cache.exists() else True
    OUTPUT.mkdir(exist_ok=True)
    failed = []
    with tempfile.TemporaryDirectory() as shim_dir:
        real_gnuplot = shutil.which("gnuplot")
        shim = pathlib.Path(shim_dir) / "gnuplot"
        shim.write_text(SHIM.format(gnuplot=real_gnuplot or "false"))
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
        env = dict(os.environ, PATH=f"{shim_dir}{os.pathsep}{os.environ['PATH']}")

        for source in sorted(EXAMPLES.glob("*.cpp")):
            binary = binaries / source.stem
            if not binary.exists():
                if not io_enabled and source.stem != "11_terminal_ascii_plot":
                    continue
                failed.append(f"{source.stem}: not built")
                continue
            with tempfile.TemporaryDirectory() as run_dir:
                result = subprocess.run([str(binary)], cwd=run_dir, env=env, capture_output=True,
                                        text=True, timeout=args.timeout)
                saved = pathlib.Path(run_dir, "result.json")
                json_output = saved.read_text() if saved.exists() else None
            text = "\n".join(line.rstrip() for line in result.stdout.splitlines()).strip("\n")
            recorded = OUTPUT / f"{source.stem}.txt"
            saved_output = OUTPUT / f"{source.stem}.json"
            if json_output is not None:
                saved_output.write_text(json_output)
                recorded.unlink(missing_ok=True)
            else:
                saved_output.unlink(missing_ok=True)
                if text:
                    recorded.write_text(text + "\n")
                else:
                    recorded.unlink(missing_ok=True)
            status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
            print(f"{source.stem:42} {status}")
            if result.returncode != 0:
                failed.append(f"{source.stem}: exit {result.returncode}\n{result.stderr}")

    for message in failed:
        print(message, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
