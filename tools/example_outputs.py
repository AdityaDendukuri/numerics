#!/usr/bin/env python3
"""Run every example program and record what it prints, for the documentation.

    python3 tools/example_outputs.py --build <build-dir>

Each `examples/<name>.cpp` must already be built (`cmake --build <build-dir> --target
numerics_examples`). Its standard output goes to `examples/output/<name>.txt`, which
`refdoc.py` shows under the program on its reference page.

Plots: a program draws only with `--plot`, which this script never passes. A `plt::show_dumb`
plot is text and is kept; anything else sent to gnuplot is discarded by a stand-in `gnuplot` on
the PATH, so no window opens and no file is written.
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
# Pass text (dumb terminal) plots to gnuplot; drop everything else.
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

    binaries = args.build / "examples"
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
                failed.append(f"{source.stem}: not built")
                continue
            with tempfile.TemporaryDirectory() as run_dir:
                result = subprocess.run([str(binary)], cwd=run_dir, env=env, capture_output=True,
                                        text=True, timeout=args.timeout)
            text = "\n".join(line.rstrip() for line in result.stdout.splitlines()).strip("\n")
            (OUTPUT / f"{source.stem}.txt").write_text(text + "\n")
            status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
            print(f"{source.stem:42} {status}")
            if result.returncode != 0:
                failed.append(f"{source.stem}: exit {result.returncode}\n{result.stderr}")

    for message in failed:
        print(message, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
