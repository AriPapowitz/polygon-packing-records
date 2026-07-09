"""Parallel chunked HTTPS downloader for hostile networks.

Combines two workarounds:
  - TCP_MAXSEG clamping (survives MTU black holes)
  - many short-lived connections fetching small Range chunks in parallel
    (defeats per-flow token-bucket shaping: every fresh connection gets a
    fast burst before the shaper clamps it)

Usage:
    python chunked_get.py URL OUT [--workers 24] [--chunk-mb 2] [--mss 1200]
    python chunked_get.py --list wheel_urls.txt --dir wheels/ [...]
"""
import argparse
import os
import socket
import ssl
import sys
import threading
import time
from queue import Queue
from urllib.parse import urlparse

CTX = ssl.create_default_context()


def open_conn(host, port, mss, timeout=30):
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, mss)
    except OSError:
        pass
    raw.settimeout(timeout)
    raw.connect((host, port))
    return CTX.wrap_socket(raw, server_hostname=host)


def http_get(host, port, path, mss, range_hdr=None, timeout=30):
    """One request on one fresh connection. Returns (status, headers, body_reader)."""
    s = open_conn(host, port, mss, timeout)
    req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: chunked-get/1.0\r\n"
           "Accept: */*\r\n" + (f"Range: {range_hdr}\r\n" if range_hdr else "")
           + "Connection: close\r\n\r\n")
    s.sendall(req.encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        c = s.recv(65536)
        if not c:
            raise ConnectionError("closed in headers")
        buf += c
    head, rest = buf.split(b"\r\n\r\n", 1)
    lines = head.decode("latin1").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return s, status, headers, rest


def resolve_redirects(url, mss, depth=0):
    if depth > 5:
        raise RuntimeError("redirect loop")
    u = urlparse(url)
    s, status, headers, _ = http_get(u.hostname, u.port or 443, u.path or "/", mss,
                                     range_hdr="bytes=0-0")
    s.close()
    if status in (301, 302, 303, 307, 308):
        loc = headers["location"]
        if loc.startswith("/"):
            loc = f"https://{u.hostname}{loc}"
        return resolve_redirects(loc, mss, depth + 1)
    if status not in (200, 206):
        raise RuntimeError(f"HTTP {status}")
    if status == 206:
        total = int(headers["content-range"].split("/")[-1])
    else:
        total = int(headers.get("content-length", 0))
    return url, total


def fetch_range(url, a, b, mss, timeout=45):
    """Fetch bytes [a, b] inclusive on a fresh connection."""
    u = urlparse(url)
    path = (u.path or "/") + (("?" + u.query) if u.query else "")
    s, status, headers, rest = http_get(u.hostname, u.port or 443, path, mss,
                                        range_hdr=f"bytes={a}-{b}", timeout=timeout)
    if status != 206:
        s.close()
        raise RuntimeError(f"HTTP {status} for range")
    want = b - a + 1
    data = bytearray(rest)
    deadline = time.time() + timeout * 3
    while len(data) < want:
        if time.time() > deadline:
            s.close()
            raise TimeoutError("range fetch too slow")
        c = s.recv(1 << 18)
        if not c:
            break
        data.extend(c)
    s.close()
    if len(data) < want:
        raise ConnectionError(f"short read {len(data)}/{want}")
    return bytes(data[:want])


def download(url, out, workers=24, chunk_mb=2.0, mss=1200):
    url, total = resolve_redirects(url, mss)
    chunk = int(chunk_mb * 1024 * 1024)
    ranges = [(a, min(a + chunk, total) - 1) for a in range(0, total, chunk)]
    q = Queue()
    for i, r in enumerate(ranges):
        q.put((i, r))
    results = {}
    lock = threading.Lock()
    done_bytes = [0]
    errors = []
    t0 = time.time()

    def worker():
        while not q.empty() and not errors:
            try:
                i, (a, b) = q.get_nowait()
            except Exception:
                return
            for attempt in range(6):
                try:
                    data = fetch_range(url, a, b, mss)
                    with lock:
                        results[i] = data
                        done_bytes[0] += len(data)
                        dt = time.time() - t0
                        print(f"\r{done_bytes[0]/1e6:.0f}/{total/1e6:.0f} MB "
                              f"{done_bytes[0]/1e6/max(dt,.01):.2f} MB/s "
                              f"({len(results)}/{len(ranges)} chunks)", end="", flush=True)
                    break
                except Exception as e:
                    if attempt == 5:
                        errors.append(f"chunk {i}: {e}")
                    else:
                        time.sleep(1 + attempt)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("; ".join(errors[:3]))
    with open(out, "wb") as f:
        for i in range(len(ranges)):
            f.write(results[i])
    dt = time.time() - t0
    print(f"\n{os.path.basename(out)}: {total/1e6:.1f} MB in {dt:.0f}s = {total/1e6/max(dt,.01):.2f} MB/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--list", help="file with one URL per line")
    ap.add_argument("--dir", default=".", help="output dir for --list mode")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk-mb", type=float, default=2.0)
    ap.add_argument("--mss", type=int, default=1200)
    args = ap.parse_args()

    if args.list:
        os.makedirs(args.dir, exist_ok=True)
        urls = [u.strip() for u in open(args.list) if u.strip()]
        for k, u in enumerate(urls):
            name = u.rsplit("/", 1)[-1]
            out = os.path.join(args.dir, name)
            if os.path.exists(out):
                print(f"[{k+1}/{len(urls)}] {name} already present, skipping")
                continue
            print(f"[{k+1}/{len(urls)}] {name}")
            download(u, out, args.workers, args.chunk_mb, args.mss)
    else:
        download(args.url, args.out, args.workers, args.chunk_mb, args.mss)


if __name__ == "__main__":
    main()
