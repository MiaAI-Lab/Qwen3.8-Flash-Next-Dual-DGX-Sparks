#!/usr/bin/env python3
"""Headless render smoke-test for the gallery in html/100-html.

An ARM aarch64 Chromium ships with Playwright at
~/.cache/ms-playwright/chromium-1208/chrome-linux/chrome, so pages CAN be
executed for real here — static checks alone miss runtime errors.

For every page it captures:
  - uncaught runtime exceptions / console errors (the real signal)
  - a 1440x900 screenshot in html/shots/ for visual review
Usage: python3 render100.py [only ...]   e.g. python3 render100.py 001 008
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "100-html")
SHOTS = os.path.join(HERE, "shots")
CHROME = os.path.expanduser(
    "~/.cache/ms-playwright/chromium-1208/chrome-linux/chrome"
)
NOISE = re.compile(
    r"dbus|bluez|vaapi|libva|gpu|GLES|sandbox|Fontconfig|fontconfig|DevTools|"
    r"jemalloc|ozone|EGL|ContextResult|swiftshader|egl",
    re.I,
)
CONSOLE = re.compile(r'CONSOLE:\d+\]\s*"(.*?)"', re.S)
IGNORE_CONSOLE = re.compile(r"favicon", re.I)


def render(path, shot):
    url = "file://" + path
    cmd = [
        CHROME, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
        "--enable-logging=stderr", "--v=0", "--virtual-time-budget=6000",
        "--window-size=1440,900", "--screenshot=" + shot, url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        log = proc.stderr or ""
    except subprocess.TimeoutExpired:
        return ["render timed out after 120s"], False
    msgs = []
    for m in CONSOLE.finditer(log):
        text = " ".join(m.group(1).split())
        if text and not IGNORE_CONSOLE.search(text):
            msgs.append(text[:220])
    fatal = [m for m in msgs if re.search(r"Uncaught|ReferenceError|TypeError|SyntaxError", m)]
    return list(dict.fromkeys(msgs)), bool(fatal)


def main():
    if not os.path.exists(CHROME):
        print("chromium not found at", CHROME)
        return 2
    os.makedirs(SHOTS, exist_ok=True)
    only = sys.argv[1:]
    names = sorted(n for n in os.listdir(TARGET) if n.endswith(".html"))
    if only:
        names = [n for n in names if n[:3] in {o.zfill(3) for o in only}]
    bad = 0
    for n in names:
        shot = os.path.join(SHOTS, n[:-5] + ".png")
        msgs, fatal = render(os.path.join(TARGET, n), shot)
        ok = not fatal and os.path.exists(shot) and os.path.getsize(shot) > 3000
        if not ok:
            bad += 1
        print(f"[{'ok  ' if ok else 'FAIL'}] {n:44s} shot={(os.path.getsize(shot) if os.path.exists(shot) else 0)//1024:4d}KB  console={len(msgs)}")
        for m in msgs[:6]:
            print("        ! " + m)
    print(f"\nrendered={len(names)} failing={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
