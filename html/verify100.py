#!/usr/bin/env python3
"""Static verifier for the self-contained HTML gallery in html/100-html.

No browser exists on this host, so this checks what can be checked without one:
  - file pairs (NNN-*.html <-> NNN-*.txt)
  - the document closes with </html>
  - no leftover build sentinels
  - no external references (http(s), protocol-relative fonts, @import, url()/src= to a file)
  - exactly one <style>, at least one <script>, and every inline script parses under `node --check`
  - tag balance via html.parser
"""
import os
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser

VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
SENTINEL = re.compile(
    r"<!--\s*(?:P|CHUNK|CH|PART|SCRIPT|JS|SEC|SECTION)\d*\s*-->|__[A-Z0-9]{2,}__"
)

# Offline-safe, so never counted as an external dependency:
#  - W3C XML namespace URIs (http://www.w3.org/2000/svg, 1999/xlink, XML/1998, ...)
#  - CSS url(#fragment) pointing at inline <svg> <defs>
#  - data: URIs
ALLOWED_URI = re.compile(
    r"""https?://(?:www\.w3\.org|\(www\.w3\.org\))/[A-Za-z0-9_.:/+\-]*""", re.I
)
# Only these contexts can actually pull in an outside resource.
SCHEME = re.compile(r"https?://", re.I)
CSS_BAD = re.compile(r"""@import|(?<![A-Za-z-])url\(\s*(?=["']?(?!#|data:))""", re.I)
TAG_BAD = re.compile(
    r"""(?:src|href)\s*=\s*["'](?![\#d]|javascript:|\{\{)[^"']*["']""", re.I
)


class Balance(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"line {self.getpos()[0]}: stray </{tag}>")
            return
        if self.stack[-1][0] == tag:
            self.stack.pop()
            return
        names = [t for t, _ in self.stack]
        if tag in names:
            while self.stack and self.stack[-1][0] != tag:
                t, ln = self.stack.pop()
                self.errors.append(
                    f"line {self.getpos()[0]}: </{tag}> closes unclosed <{t}> from line {ln}"
                )
            if self.stack:
                self.stack.pop()
        else:
            self.errors.append(f"line {self.getpos()[0]}: unmatched </{tag}>")


def check_script(js, line_no):
    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(js)
        path = fh.name
    try:
        proc = subprocess.run(
            ["node", "--check", path], capture_output=True, text=True, timeout=60
        )
        if proc.returncode:
            first = (proc.stderr or "").strip().splitlines()
            return f"script@L{line_no} node --check: {first[0] if first else 'syntax error'}"
    except FileNotFoundError:
        return "node not available"
    except subprocess.TimeoutExpired:
        return f"script@L{line_no} node --check timed out"
    finally:
        os.unlink(path)
    return None


def external_refs(raw):
    """Return genuine outside-world references, ignoring offline-safe constructs."""
    found = []
    body = ALLOWED_URI.sub("\x00", raw)
    if SCHEME.search(body):
        hit = SCHEME.search(body)
        found.append("scheme " + body[hit.start(): hit.start() + 42].replace("\n", " "))
    # CSS: only the <style> block(s) can load an outside stylesheet or font.
    for m in re.finditer(r"<style[^>]*>(.*?)</style>", raw, re.S | re.I):
        css = ALLOWED_URI.sub("\x00", m.group(1))
        bad = CSS_BAD.findall(css)
        if bad:
            at = css[CSS_BAD.search(css).start(): CSS_BAD.search(css).start() + 40]
            found.append("css " + at.replace("\n", " "))
    # Markup: src/href on any tag (skip data:, fragments, JS hooks).
    for m in TAG_BAD.finditer(raw):
        frag = m.group(0)
        if "data:" in frag or ".svg" in frag or re.match(
            r"""(?:src|href)\s*=\s*["'](?:mailto:|tel:|about:|blob:)""", frag, re.I
        ):
            continue
        found.append("attr " + frag[:48].replace("\n", " "))
    return sorted(set(found))


def dead_id_refs(raw):
    """Literal element ids the JS looks up but the document never defines.

    Catches the bug class where a page renders and throws nothing, yet a whole
    panel silently never updates because getElementById() returned null.
    """
    have = set(re.findall(r"""id\s*=\s*["']([\w:-]+)["']""", raw))
    have |= set(re.findall(r"""\.id\s*=\s*["']([\w-]+)["']""", raw))
    have |= set(
        re.findall(r"""setAttribute\(\s*["']id["']\s*,\s*["']([\w-]+)["']""", raw)
    )
    want = set(re.findall(r"""getElementById\(\s*["']([\w:-]+)["']""", raw))
    for m in re.finditer(r"""querySelector(?:All)?\(\s*["']([^"']+)["']""", raw):
        sel = m.group(1).strip()
        if sel.startswith("#"):
            want.add(re.split(r"[.\s\[:>]", sel[1:])[0])
    return sorted(want - have - {""})


def check_html(path):
    problems = []
    raw = open(path, encoding="utf-8", errors="replace").read()
    if not re.search(r"</html>\s*$", raw):
        problems.append("does not end with </html>")
    sent = SENTINEL.findall(raw)
    if sent:
        problems.append(f"leftover sentinel(s): {sorted(set(sent))[:3]}")
    dead = dead_id_refs(raw)
    if dead:
        problems.append(f"dead id ref(s): {dead[:4]}")
    ext = external_refs(raw)
    if ext:
        problems.append(f"external ref(s): {ext[:3]}")
    n_style = len(re.findall(r"<style[\s>]", raw, re.I))
    n_script = len(re.findall(r"<script[\s>]", raw, re.I))
    if n_style != 1:
        problems.append(f"{n_style} <style> blocks (expected 1)")
    if n_script < 1:
        problems.append("no <script> block")
    for m in re.finditer(r"<script[^>]*>(.*?)</script>", raw, re.S | re.I):
        if "src=" in m.group(0):
            continue
        err = check_script(m.group(1), raw[: m.start()].count("\n") + 1)
        if err:
            problems.append(err)
    bal = Balance()
    try:
        bal.feed(raw)
        bal.close()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"html.parse raised {type(exc).__name__}: {exc}")
    problems += bal.errors[:4]
    for tag, line in bal.stack[:4]:
        problems.append(f"unclosed <{tag}> from line {line}")
    return problems, len(raw.splitlines())


def main(target):
    names = sorted(os.listdir(target))
    htmls = [n for n in names if n.endswith(".html")]
    txts = {n for n in names if n.endswith(".txt")}
    bad = 0
    for n in htmls:
        base = n[:-5]
        problems, lines = check_html(os.path.join(target, n))
        has_txt = (base + ".txt") in txts
        ok = not problems and has_txt
        if not ok:
            bad += 1
        detail = "; ".join(problems) if problems else ("" if has_txt else "missing .txt")
        print(
            f"[{'ok  ' if ok else 'FAIL'}] {n:44s} {lines:5d} lines  "
            f"txt={'y' if has_txt else 'N'}  {detail}"
        )
    orphans = sorted(t[:-4] for t in txts if t[:-4] + ".html" not in set(htmls))
    print(f"\nhtml={len(htmls)} txt={len(txts)} failing={bad} orphan_txt={len(orphans)}")
    if orphans:
        print("orphans:", ", ".join(orphans))
    expected = {f"{i:03d}" for i in range(1, 101)}
    have = {n[:3] for n in htmls}
    print("numbers still missing:", len(expected - have))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
