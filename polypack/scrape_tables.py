"""Scrape all 24 Erich Friedman packing tables to CSVs and diff against a prior scrape.

Usage: python scrape_tables.py <out_dir> [prior_dir]
Fetches via urllib, with a PowerShell fallback for Windows setups whose
Python TLS stack cannot reach the site.
"""
import csv
import io
import os
import re
import subprocess
import sys

CATS = [f"{a}_in_{b}" for a in ("tri", "squ", "pen", "hex", "oct")
        for b in ("tri", "squ", "pen", "hex", "oct")
        if f"{a}_in_{b}" != "squ_in_squ"]  # squ_in_squ hosted externally

BASE = "https://erich-friedman.github.io/packing/{slug}/"

NUM_RE = re.compile(
    r"size=\+3>(\d+)\s*\.?\s*(?:(?:-|&ndash;|&#8211;|–)\s*(\d+))?\s*\.?")


def fetch(url):
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "polypack-scraper"})
        with urllib.request.urlopen(req, timeout=60) as r:
            html = r.read().decode("utf-8", errors="replace")
        if len(html) >= 500:
            return html
    except Exception:
        pass  # fall back to PowerShell below
    cmd = ["powershell", "-NoProfile", "-Command",
           f"(Invoke-WebRequest -Uri '{url}' -UseBasicParsing).Content"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0 or len(p.stdout) < 500:
        raise RuntimeError(f"fetch failed for {url}: {p.stderr[:200]}")
    return p.stdout


def parse(html):
    out = []
    for block in re.split(r"<TABLE", html, flags=re.I)[1:]:
        groups, caps = parse_block(block)
        if len(groups) != len(caps):
            snippet = re.sub(r"\s+", " ", re.sub(r"<img[^>]+>", "", block))[:400]
            raise RuntimeError(
                f"block mismatch: groups={groups} caps={len(caps)}\n{snippet}")
        for (lo, hi), cap in zip(groups, caps):
            for n in range(lo, hi + 1):
                out.append((n, *cap))
    return out


def parse_block(html):
    # Each size=+3 cell numbers one packing; "41-42." means the same packing
    # (and caption) covers both n.
    groups = [(int(m.group(1)), int(m.group(2) or m.group(1)))
              for m in NUM_RE.finditer(html)]
    caps = []
    for cell in re.split(r"<td", html, flags=re.I)[1:]:
        cell = cell.split(">", 1)[-1]  # drop the td tag's own attributes
        cell = re.sub(NUM_RE, "", cell)
        cap = parse_caption(cell)
        if cap:
            caps.append(cap)
    return groups, caps


def parse_caption(cell):
    lines = [re.sub(r"<[^>]+>", "", l).replace("&nbsp;", " ").strip()
             for l in re.split(r"<br>", cell, flags=re.I)]
    hidx = next((i for i, l in enumerate(lines)
                 if re.search(r"Found by|Proved by|Trivial|^in \w+ \d{4}", l)),
                None)
    if hidx is None or hidx == 0:
        return None  # spacer, image-only, or intro cell
    val = " ".join(l for l in lines[:hidx] if l)
    val = re.sub(r"^s = ", "", re.sub(r"^>", "", val).strip()).strip()
    if not re.search(r"\d", val) or len(val) > 120:
        return None
    rest = " ".join(lines[hidx:])
    holder, year = "", ""
    hm = re.search(r"Found by (.+?)(?= in \w+ \d{4}|\s*\.\s*$|$)", rest)
    if hm:
        holder = hm.group(1).strip().rstrip(".")
    ym = re.search(r"in (\w+ \d{4})", rest)
    if ym:
        year = ym.group(1)
    if not holder and "Trivial" in rest:
        holder = "Trivial"
    return (val, holder, year)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="directory to write per-category CSVs")
    ap.add_argument("prior_dir", nargs="?", default=None,
                    help="previous scrape directory to diff against")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = args.out_dir
    prior_dir = args.prior_dir
    os.makedirs(out_dir, exist_ok=True)
    changes = []
    for cat in CATS:
        slug = cat.replace("_", "")
        rows = parse(fetch(BASE.format(slug=slug)))
        with io.open(os.path.join(out_dir, f"{cat}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["category", "n", "s", "holder", "year"])
            for n, s, holder, year in rows:
                w.writerow([cat, n, s, holder, year])
        if prior_dir:
            old = {}
            opath = os.path.join(prior_dir, f"{cat}.csv")
            if os.path.exists(opath):
                for r in csv.DictReader(io.open(opath, encoding="utf-8-sig")):
                    old[int(r["n"])] = (r["s"], r["holder"], r["year"])
            for n, s, holder, year in rows:
                o = old.get(n)
                if o and (o[0] != s or o[1] != holder):
                    changes.append(f"{cat} n={n}: {o[0]} ({o[1]}) -> {s} ({holder}, {year})")
        print(f"{cat}: {len(rows)} rows", flush=True)
    if prior_dir:
        print(f"\n=== {len(changes)} CHANGED ROWS since {prior_dir} ===")
        for c in changes:
            print(c)


if __name__ == "__main__":
    main()
