"""mkdocs hook: link every mention of a public name to its reference page.

refdoc writes docs/reference/symbols.json, mapping each qualified name to its page and kind.
After a page is rendered, this links

- inline code naming a symbol, such as `num::cg`, `cg()` or `with_law<mat, law::spd>`;
- in code blocks, every qualified chain starting at `num`, and short names of types and
  concepts. Short function and variable names in code are left alone, since code names its
  own variables `x`, `sum` or `factor`.

A qualified name always links. A short name resolves in the page's namespace first, then in
`num`, then anywhere it is unique. A reference page's namespace is its path; a narrative page
can set one with `scope: num::kernel` in its front matter. In prose, a short name that is also
a parameter name somewhere, such as `beta`, is not linked, nor is a short name that is also a
namespace, such as `seq`. Headings, existing links, and links
from a page to itself are skipped.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from mkdocs.utils import get_relative_url

SYMBOLS = {}
PARAMETERS = set()
NAMESPACES = set()
BY_LAST = defaultdict(list)


def on_config(config):
    SYMBOLS.clear()
    PARAMETERS.clear()
    NAMESPACES.clear()
    BY_LAST.clear()
    path = Path(config["docs_dir"]) / "reference" / "symbols.json"
    if path.exists():
        data = json.loads(path.read_text())
        SYMBOLS.update(data["names"])
        PARAMETERS.update(data["parameters"])
    for name in SYMBOLS:
        BY_LAST[name.rsplit("::", 1)[-1]].append(name)
        NAMESPACES.update(name.split("::")[:-1])
    return config


def page_scope(page):
    """The namespace a page is about: its front matter's `scope`, or a reference page's path."""
    if page.meta.get("scope"):
        return page.meta["scope"]
    parts = page.file.src_uri.split("/")
    if parts[0] != "reference" or len(parts) < 3:
        return ""
    return "::".join(parts[1:-1])


def resolve(word, scope, in_code):
    """The qualified name `word` refers to, or None when it is unknown, ambiguous, or unsafe."""
    if word.startswith("num::"):
        while word:
            if word in SYMBOLS:
                return word
            word = word.rpartition("::")[0]  # a member such as num::spmat::nnz: its class
        return None
    if word in NAMESPACES:
        return None  # `seq` names the namespace num::seq, not whichever symbol is called seq
    parts = scope.split("::") if scope else []
    candidates = ["::".join(parts[:i] + [word]) for i in range(len(parts), 0, -1)]
    candidates.append(f"num::{word}")
    found = next((c for c in candidates if c in SYMBOLS), None)
    if found is None and "::" not in word and len(BY_LAST.get(word, [])) == 1:
        found = BY_LAST[word][0]
    if found is None:
        return None
    if "::" not in word:
        if in_code and SYMBOLS[found]["kind"] != "type":
            return None
        if not in_code and word in PARAMETERS:
            return None
    return found


def href(name, page, files):
    target = files.get_file_from_path(SYMBOLS[name]["page"])
    if target is None or target.src_uri == page.file.src_uri:
        return None
    return get_relative_url(target.url, page.url)


INLINE = re.compile(r"(<a\b[^>]*>.*?</a>|<h[1-6]\b.*?</h[1-6]>|<pre\b.*?</pre>)|<code>([^<]+)</code>",
                    re.S)
NAME = re.compile(r"^((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)(?:\(\)|&lt;.*&gt;)?$")

# A chain of pygments name spans joined by `::`: num::kernel::gemm, or a single name.
CHAIN = re.compile(r'<span class="(n[a-z]*)">(\w+)</span>'
                   r'((?:<span class="o">::</span><span class="n[a-z]*">\w+</span>)*)')
PRE = re.compile(r"(<pre\b.*?</pre>)", re.S)


def link_inline(html, page, files, scope):
    def replace(m):
        if m.group(1):
            return m.group(1)
        text = m.group(2)
        name = NAME.match(text)
        target = resolve(name.group(1), scope, in_code=False) if name else None
        url = href(target, page, files) if target else None
        return f'<a href="{url}"><code>{text}</code></a>' if url else m.group(0)
    return INLINE.sub(replace, html)


def link_code(block, page, files, scope):
    def replace(m):
        words = [m.group(2)] + re.findall(r'<span class="n[a-z]*">(\w+)</span>', m.group(3))
        if len(words) > 1 and words[0] != "num":
            return m.group(0)
        target = resolve("::".join(words), scope, in_code=True)
        url = href(target, page, files) if target else None
        return f'<a href="{url}">{m.group(0)}</a>' if url else m.group(0)
    return CHAIN.sub(replace, block)


def on_page_content(html, page, config, files):
    if not SYMBOLS:
        return html
    scope = page_scope(page)
    html = link_inline(html, page, files, scope)
    return PRE.sub(lambda m: link_code(m.group(1), page, files, scope), html)
