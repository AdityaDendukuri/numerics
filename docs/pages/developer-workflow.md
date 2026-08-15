# Developer Workflow {#page_developer_workflow}

Use CMake presets and `clangd` as the common interface for Neovim, VS Code,
and command-line builds.

## Configure

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The `dev` preset enables tests, Doxygen, C++23, and
`compile_commands.json`. The root `.clangd` file points `clangd` at
`build/dev/compile_commands.json`.

For sanitizer diagnostics:

```bash
cmake --preset asan
cmake --build --preset asan
ctest --preset asan
```

## Editor Indexing

Use `clangd` for semantic navigation:

```text
go to definition
find references
rename symbol
completion
signature help
diagnostics
```

VS Code users should install the recommended extensions from
`.vscode/extensions.json`. Neovim users should configure their LSP client to
run `clangd` from the repository root after `cmake --preset dev`.

## API Documentation

Build local Doxygen pages:

```bash
cmake --build --preset docs
```

Open `build/dev/docs/html/index.html` and use the generated search box for
files, namespaces, classes, functions, concepts, and pages. The docs build also
emits `build/dev/docs/numerics.tag` for external Doxygen projects.

## Tags

For Vim-style tag navigation:

```bash
cmake --build --preset tags
```

This writes a root `tags` file from `include/`, `src/`, `tests/`, and
`benchmarks/`. `clangd` remains the primary tool for C++23 templates and
concepts; tags are a lightweight fallback.
