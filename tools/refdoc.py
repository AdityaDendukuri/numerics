#!/usr/bin/env python3
"""Generate the reference pages from the `///` comments in the headers.

    tools/refdoc.py build    write docs/reference: one page per public name, one per directory
    tools/refdoc.py check    report comments that disagree with the declarations they document

A page documents one qualified name, with every overload, at
docs/reference/<namespaces>/<name>.md. The declarations come from libclang, so they are
what the compiler sees; the prose comes from the comments. Doxygen math, \\f$ and \\f[,
becomes Markdown math, which mkdocs renders with MathJax.

The compile flags come from build/compile_commands.json when it exists, so the headers are
parsed in the configuration they are built in. Set NUMERICS_BUILD to use another build
directory and NUMERICS_DOC_INCLUDE to add dependency include directories.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ROOT / "include"
EXAMPLES = ROOT / "examples"
OUT = ROOT / "docs" / "reference"
BACKENDS = ("seq", "omp", "blas", "lapack", "cuda", "mpi")
INDEXES = ROOT / "docs" / "indexes"
SKIPPED_DIRECTORIES = ("cuda/", "mpi/")


# ---------------------------------------------------------------------------------------
# Tools and flags
# ---------------------------------------------------------------------------------------

def llvm_bin():
    for directory in [os.environ.get("LLVM_BIN"), "/opt/homebrew/opt/llvm/bin",
                      "/usr/local/opt/llvm/bin"]:
        if directory and Path(directory).is_dir():
            return Path(directory)
    return None


def load_clang():
    """The libclang bindings: the `libclang` pip package, or those of an LLVM install."""
    try:
        import clang.cindex as cindex
        return cindex
    except ImportError:
        pass
    bindir = llvm_bin()
    if bindir is None:
        sys.exit("refdoc: install the bindings with `pip install libclang`")
    libdir = subprocess.run([str(bindir / "llvm-config"), "--libdir"], capture_output=True,
                            text=True).stdout.strip()
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.append(str(Path(libdir) / version / "site-packages"))
    import clang.cindex as cindex
    for name in ["libclang.dylib", "libclang.so"]:
        if (Path(libdir) / name).exists():
            cindex.Config.set_library_file(str(Path(libdir) / name))
    return cindex


def clang_format():
    found = shutil.which("clang-format")
    if found:
        return found
    bindir = llvm_bin()
    return str(bindir / "clang-format") if bindir and (bindir / "clang-format").exists() else None


def compile_flags():
    """The -I, -isystem, -D and -std flags of one translation unit of the build."""
    database = Path(os.environ.get("NUMERICS_BUILD", ROOT / "build")) / "compile_commands.json"
    flags = ["-x", "c++", "-std=c++20", f"-I{INCLUDE}"]
    for directory in os.environ.get("NUMERICS_DOC_INCLUDE", "").split(os.pathsep):
        if directory:
            flags.append(f"-I{directory}")
    if not database.exists():
        return flags
    entries = json.loads(database.read_text())
    entry = next((e for e in entries if "/tests/" in e["file"]), entries[0])
    words = shlex.split(entry["command"]) if "command" in entry else entry["arguments"]
    keep, i = [], 0
    while i < len(words):
        word = words[i]
        if word in ("-I", "-isystem", "-D", "-iquote"):
            keep += [word, words[i + 1]]
            i += 2
            continue
        if word.startswith(("-I", "-D", "-std=", "-isystem")):
            keep.append(word)
        i += 1
    return ["-x", "c++"] + flags[3:] + keep


def system_flags():
    """What libclang needs to find the headers the compiler finds on its own."""
    flags = []
    bindir = llvm_bin()
    compiler = str(bindir / "clang++") if bindir else shutil.which("clang++")
    if compiler:
        resource = subprocess.run([compiler, "-print-resource-dir"], capture_output=True,
                                  text=True).stdout.strip()
        if resource:
            flags += ["-resource-dir", resource]
    if sys.platform == "darwin":
        sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True)
        if sdk.returncode == 0:
            flags += ["-isysroot", sdk.stdout.strip()]
    return flags


# ---------------------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------------------

def comment_lines(raw):
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^(///<?|//!<?|/\*\*<?|/\*!|\*/|\*)\s?", "", line)
        line = re.sub(r"\*/$", "", line)
        lines.append(line.rstrip())
    return lines


COMMAND = re.compile(r"^[@\\](brief|param|tparam|returns?|throws?|exception|see|sa|note|"
                     r"warning|pre|post|todo|name|file|details|section)\b(\[[a-z, ]+\])?\s*(.*)$")


def parse_comment(raw):
    """Split a comment into its brief, prose paragraphs, and tagged sections."""
    doc = {"brief": "", "details": [], "param": [], "tparam": [], "returns": [],
           "throws": [], "see": [], "note": [], "pre": [], "post": []}
    if not raw:
        return doc
    current = None  # the list receiving continuation lines, or "ignored"
    paragraph = []
    in_code = False

    def flush():
        if paragraph:
            doc["details"].append("\n".join(paragraph) if in_code_block(paragraph)
                                  else " ".join(paragraph))
            paragraph.clear()

    def in_code_block(lines):
        return any(l.startswith("```") for l in lines)

    for line in comment_lines(raw):
        if line.startswith("```"):
            if not in_code:
                flush()
            in_code = not in_code
            paragraph.append(line)
            if not in_code:
                doc["details"].append("\n".join(paragraph))
                paragraph.clear()
            current = None
            continue
        if in_code:
            paragraph.append(line)
            continue
        if line in ("@{", "@}", "\\{", "\\}"):
            continue
        match = COMMAND.match(line)
        if match:
            flush()
            tag, text = match.group(1), match.group(3)
            if tag in ("file", "name", "todo", "section"):
                current = "ignored"
                continue
            if tag == "brief":
                doc["brief"] = text
                current = ["brief"]
                continue
            if tag == "details":
                paragraph.append(text)
                current = None
                continue
            if tag in ("param", "tparam", "throws", "throw", "exception"):
                name, _, rest = text.partition(" ")
                key = {"throw": "throws", "exception": "throws"}.get(tag, tag)
                entry = [name, rest.strip()]
                doc[key].append(entry)
                current = entry
                continue
            key = {"return": "returns", "sa": "see", "warning": "note"}.get(tag, tag)
            entry = [text]
            doc[key].append(entry)
            current = entry
            continue
        if not line:
            flush()
            current = None
            continue
        if current is None:
            paragraph.append(line)
        elif current == "ignored":
            continue
        elif current == ["brief"]:
            doc["brief"] += " " + line
        else:
            current[-1] = (current[-1] + " " + line).strip()
    flush()
    doc["brief"] = doc["brief"].strip()
    if not doc["brief"] and doc["details"] and not doc["details"][0].startswith("```"):
        first = doc["details"].pop(0)
        sentence, dot, rest = first.partition(". ")
        doc["brief"] = sentence + ("." if dot else "")
        if rest:
            doc["details"].insert(0, rest)
    return doc


def markup(text, link):
    """Doxygen inline markup to Markdown: math, and @ref as a link."""
    text = re.sub(r"\\f\[(.*?)\\f\]", lambda m: "\n\n$$\n" + m.group(1).strip() + "\n$$\n\n",
                  text, flags=re.S)
    text = re.sub(r"\\f\$(.*?)\\f\$", lambda m: "$" + m.group(1).strip() + "$", text,
                  flags=re.S)
    text = re.sub(r'[@\\]ref\s+([\w:]+)(?:\s+"([^"]*)")?',
                  lambda m: link(m.group(1), m.group(2)), text)
    text = re.sub(r"[@\\][cp]\s+(\w+)", r"`\1`", text)
    return text


def table_cell(text):
    """Make text safe inside a Markdown table: no bare pipes, inside math or out."""
    def math(m):
        return m.group(0).replace(r"\|", r"\Vert ").replace("|", r"\vert ")
    text = re.sub(r"\$[^$]*\$", math, text)
    return re.sub(r"(?<!\\)\|", r"\|", text).replace("\n", " ")


# ---------------------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------------------

def qualified(cursor, cindex):
    parts, parent = [cursor.spelling], cursor.semantic_parent
    while parent is not None and parent.kind != cindex.CursorKind.TRANSLATION_UNIT:
        if parent.kind != cindex.CursorKind.NAMESPACE:
            return None
        if parent.spelling:
            parts.append(parent.spelling)
        parent = parent.semantic_parent
    return "::".join(reversed(parts))


def is_private(name):
    parts = name.split("::")
    if not parts[-1] or parts[-1].startswith(("operator", "_", "<")) or parts[0] == "std" \
            or parts[-1] == "tag_invoke":
        return True
    return any(part == "detail" or part.endswith("_detail") or part == "experimental"
               for part in parts[:-1])


def source_text(cursor, start, end):
    return Path(cursor.extent.start.file.name).read_bytes()[start:end].decode(errors="replace")


FORMAT_STYLE = ("{BasedOnStyle: LLVM, ColumnLimit: 88, AlwaysBreakTemplateDeclarations: Yes, "
                "RequiresClausePosition: OwnLine, IndentRequiresClause: false}")


def format_code(text, formatter):
    if formatter is None:
        return text
    result = subprocess.run([formatter, f"--style={FORMAT_STYLE}"], input=text,
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else text


def signature(cursor, cindex, formatter):
    """A declaration without its body."""
    kind = cindex.CursorKind
    start, end = cursor.extent.start.offset, cursor.extent.end.offset
    if cursor.kind in (kind.STRUCT_DECL, kind.CLASS_DECL, kind.CLASS_TEMPLATE):
        for token in cursor.get_tokens():
            if token.spelling == "{":
                end = token.extent.start.offset
                break
    else:
        end = declaration_end(cursor, end)
    text = source_text(cursor, start, end)
    if cursor.kind == kind.VAR_DECL and "#" in text:
        text = text.split("=")[0]
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"\binline\s+|\[\[nodiscard\]\]\s*|\bNUM_K_\w+\s*", "", text)
    text = " ".join(text.split()).rstrip()
    if not text.endswith(";"):
        text += ";"
    return format_code(text, formatter)


def declaration_end(cursor, end):
    """Where a function's declaration ends: before its body or its initializer list."""
    base = cursor.spelling.split("<")[0]
    symbol = base[len("operator"):].strip() if base.startswith("operator") else None
    depth, parameters_at, done = 0, None, False
    armed, collected = False, None
    for token in cursor.get_tokens():
        spelling = token.spelling
        if parameters_at is None and depth == 0:
            if symbol is not None:
                if spelling == "operator":
                    collected = ""
                    continue
                if collected is not None and collected != symbol:
                    collected += spelling
                    armed = collected == symbol
                    continue
            elif spelling == base:
                armed = True
                continue
        if spelling in ("(", "["):
            if spelling == "(" and armed and parameters_at is None and depth == 0:
                parameters_at = 0
            depth += 1
        elif spelling in (")", "]"):
            depth -= 1
            if parameters_at is not None and depth == 0:
                done = True
        elif depth == 0 and done and spelling in ("{", ":"):
            return token.extent.start.offset
        if spelling != base:
            armed = armed and spelling == "("
    return end


def comment_of(cursor):
    """The comment libclang attaches, or one written between `template <...>` and the name."""
    if cursor.raw_comment:
        return cursor.raw_comment
    start, name = cursor.extent.start.line, cursor.location.line
    if name <= start:
        return None
    lines = Path(cursor.location.file.name).read_text(errors="replace").splitlines()
    inside = [l for l in lines[start - 1:name - 1] if l.strip().startswith("///")]
    return "\n".join(inside) or None


def parameters(cursor, cindex):
    return [c.spelling for c in cursor.get_children()
            if c.kind == cindex.CursorKind.PARM_DECL and c.spelling]


def template_parameters(cursor, cindex):
    kinds = (cindex.CursorKind.TEMPLATE_TYPE_PARAMETER,
             cindex.CursorKind.TEMPLATE_NON_TYPE_PARAMETER,
             cindex.CursorKind.TEMPLATE_TEMPLATE_PARAMETER)
    return [c.spelling for c in cursor.get_children() if c.kind in kinds and c.spelling]


def members(cursor, cindex):
    kind = cindex.CursorKind
    wanted = {kind.CXX_METHOD, kind.CONSTRUCTOR, kind.FUNCTION_TEMPLATE, kind.FIELD_DECL,
              kind.VAR_DECL, kind.TYPE_ALIAS_DECL, kind.TYPEDEF_DECL}
    found = []
    for child in cursor.get_children():
        if child.kind not in wanted:
            continue
        if child.access_specifier in (cindex.AccessSpecifier.PRIVATE,
                                      cindex.AccessSpecifier.PROTECTED):
            continue
        if child.spelling.startswith("_") or child.spelling.endswith("_"):
            continue
        text = signature(child, cindex, None)
        if "= delete" in text:
            continue
        found.append({"declaration": " ".join(text.split()),
                      "doc": parse_comment(comment_of(child))})
    return found


def entity_kinds(cindex):
    kind = cindex.CursorKind
    kinds = {kind.FUNCTION_DECL, kind.FUNCTION_TEMPLATE, kind.STRUCT_DECL, kind.CLASS_DECL,
             kind.CLASS_TEMPLATE, kind.TYPE_ALIAS_DECL, kind.VAR_DECL}
    for extra in ("CONCEPT_DECL", "TYPE_ALIAS_TEMPLATE_DECL"):
        if hasattr(kind, extra):
            kinds.add(getattr(kind, extra))
    return kinds


def parse(headers, cindex):
    """One translation unit including every header; headers that fail to parse are dropped."""
    flags = compile_flags() + system_flags()
    headers = list(headers)
    while True:
        unit = "\n".join(f'#include "{h}"' for h in headers)
        tu = cindex.Index.create().parse("refdoc.cpp", args=flags,
                                         unsaved_files=[("refdoc.cpp", unit)])
        failing = set()
        for d in tu.diagnostics:
            if d.severity >= cindex.Diagnostic.Error and d.location.file is not None:
                path = Path(d.location.file.name).resolve()
                if INCLUDE.resolve() in path.parents:
                    failing.add(str(path.relative_to(INCLUDE.resolve())))
        failing &= set(headers)
        if not failing:
            errors = [d for d in tu.diagnostics if d.severity >= cindex.Diagnostic.Error]
            if errors:
                sys.exit("refdoc: the headers do not parse:\n" +
                         "\n".join(str(e) for e in errors[:5]))
            return tu, headers
        for header in sorted(failing):
            print(f"refdoc: skipping {header}, which does not parse in this configuration")
        headers = [h for h in headers if h not in failing]


def collect(cindex):
    """Every public namespace-scope name, grouped with its overloads."""
    include = INCLUDE.resolve()
    headers = sorted(str(p.relative_to(INCLUDE)) for p in INCLUDE.rglob("*.hpp")
                     if not str(p.relative_to(INCLUDE)).startswith(SKIPPED_DIRECTORIES))
    tu, headers = parse(headers, cindex)
    formatter = clang_format()
    kinds, kind = entity_kinds(cindex), cindex.CursorKind
    class_kinds = (kind.STRUCT_DECL, kind.CLASS_DECL, kind.CLASS_TEMPLATE)
    names, usr_entry = {}, {}
    for cursor in tu.cursor.walk_preorder():
        if cursor.kind not in kinds or cursor.location.file is None:
            continue
        path = Path(cursor.location.file.name).resolve()
        if include not in path.parents:
            continue
        if cursor.semantic_parent is None or cursor.semantic_parent.kind != kind.NAMESPACE:
            continue
        if cursor.kind in class_kinds and not cursor.is_definition():
            continue
        name = qualified(cursor, cindex)
        if name is None or is_private(name):
            continue
        usr = cursor.get_usr()
        if usr in usr_entry:  # a redeclaration: keep the first text, take any comment
            entry = usr_entry[usr]
            if not entry["doc"]["brief"] and comment_of(cursor):
                entry["doc"] = parse_comment(comment_of(cursor))
            continue
        text = signature(cursor, cindex, formatter)
        if "= delete" in text or text.startswith(("extern template", "template class",
                                                  "template struct")):
            continue
        entry = {
            "header": str(path.relative_to(include)),
            "line": cursor.location.line,
            "start": cursor.extent.start.line,
            "declaration": text,
            "parameters": parameters(cursor, cindex),
            "template_parameters": template_parameters(cursor, cindex),
            "doc": parse_comment(comment_of(cursor)),
            "members": members(cursor, cindex) if cursor.kind in class_kinds else [],
        }
        usr_entry[usr] = entry
        names.setdefault(name, []).append(entry)
    for name in [n for n in names if n.endswith("_t") and n[:-2] in names]:
        del names[name]  # a customization point's type; its object is the public name
    return names, headers


def file_brief(header):
    """The @brief of a header's @file comment."""
    text = (INCLUDE / header).read_text(errors="replace")
    match = re.search(r"///\s*@file[^\n]*\n((?:[ \t]*///[^\n]*\n)*)", text)
    if not match:
        return ""
    return parse_comment(match.group(0))["brief"]


# ---------------------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------------------

def page_path(name):
    return OUT.joinpath(*name.split("::")).with_suffix(".md")


class Linker:
    """Resolves a name written in a comment to a page, relative to the page citing it."""

    def __init__(self, names):
        self.names = set(names)
        self.by_last = defaultdict(list)
        for name in names:
            self.by_last[name.split("::")[-1]].append(name)

    def resolve(self, word, scope):
        word = word.strip("`():,.")
        if word in self.names:
            return word
        parts = scope.split("::") if scope else []
        for i in range(len(parts), -1, -1):
            candidate = "::".join(parts[:i] + [word])
            if candidate in self.names:
                return candidate
        matches = self.by_last.get(word.split("::")[-1], [])
        return matches[0] if len(matches) == 1 else None

    def link(self, word, text, source_page, scope):
        target = self.resolve(word, scope)
        if target is None and "::" in word:
            target = self.resolve(word.rsplit("::", 1)[0], scope)
        label = text or f"`{word.strip('`')}`"
        if target is None:
            return label
        return f"[{label}]({os.path.relpath(page_path(target), source_page.parent)})"


def section(title, body):
    return f"\n## {title}\n\n{body.strip()}\n" if body.strip() else ""


def listing(pairs, convert):
    return "\n".join(f"- `{name}`: {convert(text)}" if text else f"- `{name}`"
                     for name, text in pairs)


def prose(doc, convert):
    parts = [doc["brief"]] + doc["details"]
    return "\n\n".join(convert(p) for p in parts if p)


def fallback(name, overload, linker, path):
    """What an undocumented name can still say: whose backend version or alias it is."""
    scope, _, last = name.rpartition("::")
    if scope.split("::")[-1] in BACKENDS:
        base = scope.rpartition("::")[0] + "::" + last
        if base in linker.names:
            return (f"The `{scope}` version of "
                    f"{linker.link(base, f'`{base}`', path, scope)}.")
    match = re.match(r"(?:template\s*<.*?>\s*)?using\s+\w+\s*=\s*([\w:]+)", overload["declaration"],
                     re.S)
    if match:
        target = linker.resolve(match.group(1), scope)
        if target and target != name:
            return f"An alias of {linker.link(target, f'`{target}`', path, scope)}."
    return ""


def render_page(name, overloads, linker):
    path = page_path(name)
    scope = name.rpartition("::")[0]

    def convert(text):
        return markup(text, lambda w, t: linker.link(w, t, path, scope))

    headers = sorted({o["header"] for o in overloads})
    numbered = len(overloads) > 1
    blocks = [o["declaration"] + (f"  // ({i})" if numbered else "")
              for i, o in enumerate(overloads, 1)]
    out = [f"# {name}", "",
           "*Defined in header* " + ", ".join(f"`<{h}>`" for h in headers), "",
           "```cpp", "\n\n".join(blocks), "```", ""]

    texts = [prose(o["doc"], convert) or fallback(name, o, linker, path) for o in overloads]
    if numbered and len({t for t in texts if t}) > 1:
        for i, text in enumerate(texts, 1):
            out += [f"({i}) " + (text or "*Undocumented.*"), ""]
    elif any(texts):
        out += [next(t for t in texts if t), ""]
    else:
        out += ["*Undocumented.*", ""]

    def union(key):
        seen, merged = set(), []
        for o in overloads:
            for pair in o["doc"][key]:
                if pair[0] not in seen:
                    seen.add(pair[0])
                    merged.append(pair)
        return merged

    def lines(key):
        return "\n\n".join(convert(e[0]) for o in overloads for e in o["doc"][key])

    returns = [convert(o["doc"]["returns"][0][0]) for o in overloads if o["doc"]["returns"]]
    out.append("".join([
        section("Template parameters", listing(union("tparam"), convert)),
        section("Parameters", listing(union("param"), convert)),
        section("Return value", "\n\n".join(dict.fromkeys(returns))),
        section("Exceptions", listing(union("throws"), convert)),
        section("Preconditions", lines("pre")),
        section("Notes", lines("note")),
    ]))

    member_list = [m for o in overloads for m in o["members"]]
    if member_list:
        out += ["", "## Members", "", "| member | description |", "| :--- | :--- |"]
        for m in member_list:
            code = m["declaration"].rstrip(";").replace("|", "\\|")
            text = table_cell(convert(" ".join([m["doc"]["brief"]] + m["doc"]["details"])))
            out.append(f"| `{code}` | {text} |")
        out.append("")

    see = [w for o in overloads for s in o["doc"]["see"] for w in re.split(r"[,\s]+", s[0]) if w]
    if see:
        out.append(section("See also", ", ".join(linker.link(w, None, path, scope)
                                                  for w in dict.fromkeys(see))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip() + "\n")


def summary(name, overloads, linker, source_page):
    doc = next((o["doc"] for o in overloads if o["doc"]["brief"]), None)
    if doc is None:
        return ""
    scope = name.rpartition("::")[0]
    return table_cell(markup(doc["brief"], lambda w, t: linker.link(w, t, source_page, scope)))


def plain(text):
    return markup(text, lambda w, t: t or f"`{w}`")


def directory_of(header):
    return header.split("/")[0] if "/" in header else "numerics"


def common_prefix(names):
    """The namespace every name shares, which the index need not repeat."""
    scopes = [n.split("::")[:-1] for n in names]
    prefix = []
    for parts in zip(*scopes):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    return "::".join(prefix) + "::" if prefix else ""


def grid(groups, page, linker, scope):
    """The multi-column index: a titled group per line of names, each linking to its page."""
    names = [n for _, _, listed in groups for n in listed]
    prefix = common_prefix([linker.resolve(n, scope) or n for n in names])
    out = ['<div class="sym-index" markdown>', ""]
    for title, header, listed in groups:
        hdr = f' <span class="hdr">&lt;{header}&gt;</span>' if header else ""
        links = []
        for name in listed:
            shown = name[len(prefix):] if name.startswith(prefix) else name
            links.append(linker.link(name, f"`{shown}`", page, scope))
        out += ['<div class="kidx-group" markdown>',
                f'<span class="kidx-title">{title}{hdr}</span>',
                '<p class="kidx-syms" markdown="span">' + " &ndash; ".join(links) + "</p>",
                "</div>", ""]
    out.append("</div>")
    return "\n".join(out)


def read_index(path):
    """A curated index: a title, an introduction, and `## Title <header>` groups of names."""
    title, intro, groups = "", [], []
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            match = re.match(r"## (.*?)\s*(?:<([^>]+)>)?\s*$", line)
            groups.append((match.group(1), match.group(2) or "", []))
        elif line.strip() and groups:
            groups[-1][2].extend(line.split())
        elif line.strip():
            intro.append(line)
    return title, " ".join(intro), groups


def render_curated(names, linker):
    """The hand-grouped indexes in docs/indexes, and what each is missing."""
    problems, pages = [], []
    for source in sorted(INDEXES.glob("*.md")):
        title, intro, groups = read_index(source)
        page = OUT / source.name
        listed = set()
        for _, _, group in groups:
            for name in group:
                target = linker.resolve(name, "num")
                if target is None and "::" in name:
                    target = linker.resolve(name.rsplit("::", 1)[0], "num")
                if target is None:
                    problems.append(f"{source.relative_to(ROOT)}: `{name}` is not a public name")
                listed.add(target)
        covered = {header for _, header, _ in groups if header}
        namespaces = {n.rpartition("::")[0] for n in listed if n}
        for name, overloads in sorted(names.items()):
            if overloads[0]["header"] in covered and name not in listed \
                    and name.rpartition("::")[0] in namespaces:
                problems.append(f"{source.relative_to(ROOT)}: `{name}` from "
                                f"<{overloads[0]['header']}> is not listed")
        page.write_text(f"# {title}\n\n{intro}\n\n{grid(groups, page, linker, 'num')}\n")
        pages.append((page, title, intro))
    return pages, problems


def render_indexes(names, headers, linker):
    by_header = defaultdict(list)
    for name, overloads in names.items():
        by_header[overloads[0]["header"]].append(name)
    directories = defaultdict(list)
    for header in headers:
        directories[directory_of(header)].append(header)

    curated, problems = render_curated(names, linker)
    top = ["# Reference", "",
           "One page per public name, generated from the comments in the headers.", ""]
    if curated:
        top += ["## By topic", "", "| index | contents |", "| :--- | :--- |"]
        for page, title, intro in curated:
            top.append(f"| [{title}]({page.name}) | {table_cell(intro)} |")
        top.append("")
    top += ["## By directory", "", "| directory | contents |", "| :--- | :--- |"]
    for directory in sorted(directories):
        if not any(by_header[h] for h in directories[directory]):
            continue
        umbrella = f"{directory}/{directory}.hpp"
        about = file_brief(umbrella) if (INCLUDE / umbrella).exists() else ""
        top.append(f"| [`{directory}/`]({directory}.md) | {table_cell(plain(about))} |")
        page = OUT / f"{directory}.md"
        groups = []
        for header in sorted(directories[directory]):
            listed = sorted(by_header[header])
            if listed:
                title = plain(file_brief(header)).rstrip(".") or header.rsplit("/", 1)[-1]
                groups.append((title, header, listed))
        body = [f"# {directory}/", ""]
        if about:
            body += [plain(about), ""]
        body.append(grid(groups, page, linker, "num"))
        page.write_text("\n".join(body).rstrip() + "\n")
    top += ["", "The example programs are listed under [Examples](examples/index.md)."]
    (OUT / "index.md").write_text("\n".join(top) + "\n")
    return problems


def render_examples():
    directory = OUT / "examples"
    directory.mkdir(parents=True, exist_ok=True)
    rows = ["# Examples", "", "Complete programs from `examples/`, built with "
            "`-DNUMERICS_BUILD_EXAMPLES=ON`.", "", "| program | summary |", "| :--- | :--- |"]
    for source in sorted(EXAMPLES.glob("*.cpp")):
        text = source.read_text(errors="replace")
        head = re.match(r"(\s*///[^\n]*\n)+", text)
        brief = plain(parse_comment(head.group(0))["brief"]) if head else ""
        rows.append(f"| [`{source.name}`]({source.stem}.md) | {table_cell(brief)} |")
        body = text[head.end():] if head else text
        (directory / f"{source.stem}.md").write_text(
            f"# {source.name}\n\n{brief}\n\n```cpp\n{body.strip()}\n```\n")
    (directory / "index.md").write_text("\n".join(rows) + "\n")


def build(cindex):
    names, headers = collect(cindex)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    linker = Linker(names)
    for name, overloads in names.items():
        render_page(name, overloads, linker)
    problems = render_indexes(names, headers, linker)
    render_examples()
    for problem in problems:
        print(f"refdoc: {problem}")
    documented = sum(1 for o in names.values() if any(e["doc"]["brief"] for e in o))
    print(f"refdoc: {len(names)} names from {len(headers)} headers, {documented} documented")
    return 0


def check(cindex):
    names, headers = collect(cindex)
    OUT.mkdir(parents=True, exist_ok=True)
    linker = Linker(names)
    problems = render_indexes(names, headers, linker)
    undocumented = defaultdict(list)
    for name, overloads in sorted(names.items()):
        if not any(o["doc"]["brief"] for o in overloads) \
                and not fallback(name, overloads[0], linker, page_path(name)):
            undocumented[overloads[0]["header"]].append(name)
        for o in overloads:
            where = f"{o['header']}:{o['line']} {name}"
            actual = set(o["parameters"]) | set(o["template_parameters"])
            for pname, _ in o["doc"]["param"] + o["doc"]["tparam"]:
                if pname not in actual:
                    problems.append(f"{where}: documents `{pname}`, which is not a parameter")
            raw = " ".join([o["doc"]["brief"]] + o["doc"]["details"])
            if re.search(r"\\f[{}]|[@\\](code|endcode|li|par)\b", raw):
                problems.append(f"{where}: markup that refdoc does not convert")
    for problem in problems:
        print(problem)
    total = sum(len(v) for v in undocumented.values())
    print(f"refdoc: {len(names)} names, {total} without a comment, {len(problems)} problems")
    for header, listed in sorted(undocumented.items()):
        print(f"  {header}: {', '.join(listed)}")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("build", "check"):
        sys.exit(__doc__)
    clang = load_clang()
    sys.exit(build(clang) if sys.argv[1] == "build" else check(clang))
