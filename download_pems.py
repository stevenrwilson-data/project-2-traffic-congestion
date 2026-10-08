#!/usr/bin/env python3
"""
Download PeMS D7 station 5-minute files for Feb-May 2026.

Reads a text file of lines in the form:
    /?download=522690&dnode=Clearinghouse | d07_text_station_5min_2026_05_17.txt.gz

Requires the PeMS session cookie in the environment:
    export PEMS_COOKIE=<PHPSESSID value>

Usage:
    python3 download_pems.py pems_file_list.txt /path/to/output_dir
"""

import os
import re
import subprocess
import sys
import time

BASE = "https://pems.dot.ca.gov"
MONTHS = ("2026_02", "2026_03", "2026_04", "2026_05")
DELAY = 2.0  # seconds between requests


def parse_list(path):
    """Return [(url_path, filename), ...] for target months."""
    entries = []
    with open(path) as f:
        for line in f:
            if "|" not in line:
                continue
            href, name = (p.strip() for p in line.split("|", 1))
            if not href.startswith("/?download="):
                continue
            if not re.search(r"_(%s)_\d{2}\.txt\.gz$" % "|".join(MONTHS), name):
                continue
            entries.append((href, name))
    return entries


def download(href, name, outdir, cookie):
    dest = os.path.join(outdir, name)
    if os.path.exists(dest):
        return "skip"

    tmp = dest + ".part"
    result = subprocess.run(
        [
            "curl", "-sS", "--fail", "--max-time", "300",
            "-b", "PHPSESSID=" + cookie,
            "-o", tmp,
            BASE + href,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return "FAILED: curl %d %s" % (result.returncode, result.stderr.strip())

    if os.path.getsize(tmp) < 1_000_000:
        os.remove(tmp)
        return "too small"

    os.rename(tmp, dest)
    return "ok"


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: download_pems.py <file_list.txt> <output_dir>")

    list_path, outdir = sys.argv[1], sys.argv[2]

    cookie = os.environ.get("PEMS_COOKIE")
    if not cookie:
        sys.exit("PEMS_COOKIE not set")

    os.makedirs(outdir, exist_ok=True)
    entries = parse_list(list_path)
    print("%d files matched %s" % (len(entries), ", ".join(MONTHS)))

    failures = []
    for i, (href, name) in enumerate(entries, 1):
        status = download(href, name, outdir, cookie)
        if status.startswith("FAILED") or status == "too small":
            failures.append(name)
        print("[%3d/%d] %s  %s" % (i, len(entries), name, status))
        if status == "ok":
            time.sleep(DELAY)

    print("\ndone. %d failures" % len(failures))
    for name in failures:
        print("  " + name)


if __name__ == "__main__":
    main()
