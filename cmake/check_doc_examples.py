#!/usr/bin/env python3
"""Compile every C++ example in docs/pages and report the ones that no longer build.

The docs carry ~1500 lines of example code across ~270 fenced blocks. Nothing compiled
any of it, so an API rename would silently leave the documentation describing a library
that no longer exists -- which is exactly what happened to the backend and concept
vocabularies before this script was written.

Two things make the check usable rather than noisy:

*Fragments.* Most blocks are excerpts that refer to a variable an earlier block
introduced. Each block is tried twice, wrapped at function scope and at namespace scope,
and is only reported when *both* wrappings fail with an error that is not a
fragment artifact (an undeclared identifier, a missing type name, and so on). A block
that compiles either way is fine.

*Deliberate failures.* The concepts page demonstrates what the compiler says when a law is
missing, so some blocks are supposed to fail. Mark those with a `DOES NOT COMPILE` comment
on the first line; they are skipped, and the marker tells the reader the same thing.

*Proposed API.* The refactor roadmap describes an architecture that does not exist yet.
Mark those blocks `PROPOSED API`, which likewise skips them and tells a reader that the
call is a design target rather than something they can write today.

Run it with `cmake --build <build> --target check-docs`. It is not part of `ctest`: it
compiles the umbrella header once per block and takes minutes, where the whole test suite
takes seconds.
"""

import argparse
import concurrent.futures
import pathlib
import re
import subprocess
import sys
import tempfile

FUNCTION_SCOPE = """#include <numerics.hpp>
#include <vector>
#include <iostream>
using namespace num;
void doc_example() {
%s
}
"""

NAMESPACE_SCOPE = """#include <numerics.hpp>
#include <vector>
#include <iostream>
using namespace num;
%s
"""

# Errors that mean "this block is an excerpt", not "this block is stale".
FRAGMENT = re.compile(
    r"undeclared identifier"
    r"|unknown type name"
    r"|a type specifier is required"
    r"|expected unqualified-id"
    r"|redefinition of"
    r"|non-local lambda"
    r"|expected '\)'"
    r"|expected expression"
    r"|use of class template"
    r"|cannot use dot operator on a type"
    r"|expected parameter declarator"
    r"|expected ';' after"
    r"|no type named '\w+' in namespace"
    r"|too few template arguments for concept"
    r"|use of concept '\w+' requires template arguments"
)

SKIP = re.compile(r"DOES NOT COMPILE|PROPOSED API")


def blocks(pages_dir):
    for path in sorted(pages_dir.glob("*.md")):
        source = path.read_text()
        for index, match in enumerate(re.finditer(r"```cpp\n(.*?)```", source, re.S)):
            line = source[: match.start()].count("\n") + 1
            yield path.name, index, line, match.group(1)


def check(job):
    name, index, line, code, include_dir, tmp, defines = job
    if not code.strip() or SKIP.search(code):
        return None

    unexplained = []
    for kind, template in (("fn", FUNCTION_SCOPE), ("ns", NAMESPACE_SCOPE)):
        path = tmp / f"{name}.{index}.{kind}.cpp"
        path.write_text(template % code)
        result = subprocess.run(
            ["c++", "-std=c++20", f"-I{include_dir}", *defines, "-fsyntax-only", str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return None
        errors = [l.split(": error: ")[-1] for l in result.stderr.splitlines() if ": error: " in l]
        unexplained.append([e for e in errors if not FRAGMENT.search(e)])

    if unexplained[0] and unexplained[1]:
        return name, line, unexplained[0][0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--define", action="append", default=[])
    args = parser.parse_args()

    root = pathlib.Path(args.source_dir)
    pages = root / "docs" / "pages"
    include = root / "include"
    defines = [f"-D{d}" for d in args.define]

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = pathlib.Path(raw_tmp)
        jobs = [(n, i, l, c, include, tmp, defines) for n, i, l, c in blocks(pages)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            broken = [r for r in pool.map(check, jobs) if r]

    print(f"documentation examples checked: {len(jobs)}")
    if not broken:
        print("all of them still build")
        return 0

    print(f"no longer building: {len(broken)}\n")
    for name, line, error in sorted(broken):
        print(f"  docs/pages/{name}:{line}\n      {error[:110]}")
    print(
        "\nEach block above names an API the documentation still describes but the library\n"
        "no longer provides. Fix the example, or the library if the example is the\n"
        "intended interface. If a block is meant to fail (demonstrating a diagnostic),\n"
        "put a `DOES NOT COMPILE` comment on its first line; if it describes API that\n"
        "does not exist yet, put `PROPOSED API`."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
