"""MSS-clamped HTTPS downloader: works around MTU-black-hole networks by
forcing a small TCP Maximum Segment Size before connecting.

Usage: python mss_get.py URL OUT_PATH [MSS]
"""
import socket
import ssl
import sys
import time
from urllib.parse import urlparse


def fetch(url, out_path, mss=1200, depth=0):
    if depth > 5:
        raise RuntimeError("too many redirects")
    u = urlparse(url)
    host, port = u.hostname, u.port or 443
    path = (u.path or "/") + (("?" + u.query) if u.query else "")

    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, mss)
    except OSError as e:
        print(f"warning: TCP_MAXSEG failed: {e}")
    raw.settimeout(60)
    raw.connect((host, port))
    ctx = ssl.create_default_context()
    s = ctx.wrap_socket(raw, server_hostname=host)
    req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
           "User-Agent: mss-get/1.0\r\nAccept: */*\r\nConnection: close\r\n\r\n")
    s.sendall(req.encode())

    # read headers
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(65536)
        if not chunk:
            raise RuntimeError("connection closed in headers")
        buf += chunk
    head, body = buf.split(b"\r\n\r\n", 1)
    head_txt = head.decode("latin1")
    status = int(head_txt.split()[1])
    headers = {}
    for line in head_txt.split("\r\n")[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    if status in (301, 302, 303, 307, 308):
        s.close()
        loc = headers["location"]
        if loc.startswith("/"):
            loc = f"https://{host}{loc}"
        return fetch(loc, out_path, mss, depth + 1)
    if status != 200:
        raise RuntimeError(f"HTTP {status}")

    total = int(headers.get("content-length", 0))
    t0 = time.time()
    got = len(body)
    with open(out_path, "wb") as f:
        f.write(body)
        while True:
            if total and got >= total:
                break
            chunk = s.recv(1 << 18)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got % (1 << 22) < (1 << 18):  # progress every ~4MB
                dt = time.time() - t0
                print(f"\r{got/1e6:.1f}/{total/1e6:.1f} MB  {got/1e6/max(dt,0.01):.2f} MB/s",
                      end="", flush=True)
    dt = time.time() - t0
    print(f"\nDONE {got} bytes in {dt:.1f}s = {got/1e6/max(dt,0.01):.2f} MB/s")
    s.close()


if __name__ == "__main__":
    url, out = sys.argv[1], sys.argv[2]
    mss = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    fetch(url, out, mss)
