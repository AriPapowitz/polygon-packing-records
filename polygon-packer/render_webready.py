"""Render packing solutions as web-ready GIFs matching Erich Friedman's site
style: same container orientation, fill color, and pixel scale as the page's
existing images, thin black outlines, white background, .gif extension.

Style is learned per category from a sample incumbent GIF (the ones the sweep
downloaded): fill color = dominant saturated/gray shade, orientation = fitted
container rotation, scale = fitted px-per-unit.

Usage:
    python render_webready.py <solution.json> <sample_incumbent.gif> <out.gif>
"""

import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

SS = 8  # supersampling factor for antialiasing

# canonical container orientation per side-count (matches the site's pages):
# pentagon apex-up/flat base, hexagon flat top+bottom, octagon flat bottom,
# triangle apex-up, square axis-aligned
CANON_PHI = {3: np.pi / 2, 4: np.pi / 4, 5: np.pi / 2, 6: 0.0, 8: np.pi / 8}


def learn_style(gif_path, nsc):
    img = np.asarray(Image.open(gif_path).convert("RGB")).astype(int)
    H, W, _ = img.shape
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    sat = img.max(-1) - img.min(-1)
    fill = (sat >= 15) & (img.max(-1) >= 200)
    if fill.sum() < 100:                        # gray-fill page
        ink0 = ~((R > 235) & (G > 235) & (B > 235))
        vals, counts = np.unique(img.max(-1)[ink0 & (sat < 10)], return_counts=True)
        shade = int(vals[counts.argmax()])
        fill = (sat < 8) & (np.abs(img.max(-1) - shade) <= 6)
    rows, cols = np.nonzero(fill)
    color = tuple(int(np.median(img[rows, cols, c])) for c in range(3))

    ink = ~((R > 235) & (G > 235) & (B > 235))
    rows, cols = np.nonzero(ink)
    pts = np.column_stack([cols.astype(float), (H - 1 - rows).astype(float)])
    hull = pts[ConvexHull(pts).vertices]
    corners = [hull[hull[:, 0].argmin()]]
    for _ in range(nsc - 1):
        d = np.min([np.hypot(*(hull - c).T) for c in corners], axis=0)
        corners.append(hull[d.argmax()])
    corners = np.array(corners)
    cc = corners.mean(0)
    ang = np.arctan2(*(corners - cc).T[::-1])
    # container rotation: offset of corner angles from engine convention (0, 2pi/nsc, ...)
    phi = np.angle(np.mean(np.exp(1j * nsc * ang))) / nsc
    circ_px = np.hypot(*(corners - cc).T).mean()
    return color, float(phi), float(circ_px), (W, H)


def render(sol_path, sample_gif, out_path):
    sol = json.load(open(sol_path))
    n, nsi, nsc = sol["inner_polygons"], sol["inner_sides"], sol["container_sides"]
    S = sol["container_circumradius"]
    color, phi, circ_px, (W, H) = learn_style(sample_gif, nsc)
    phi = CANON_PHI.get(nsc, phi)                    # exact orientation, no fit noise
    k = circ_px / S                                  # px per engine unit

    # canvas sized to the container at the sample's scale + small margin
    cont = S * np.exp(1j * (2 * np.pi * np.arange(nsc) / nsc + phi))
    pad = 4
    xs, ys = cont.real * k, cont.imag * k
    W2 = int(np.ceil(xs.max() - xs.min())) + 2 * pad
    H2 = int(np.ceil(ys.max() - ys.min())) + 2 * pad
    ox, oy = pad - xs.min(), pad - ys.min()

    def to_px(z):
        return ((z.real * k + ox) * SS, (H2 - 1 - (z.imag * k + oy)) * SS)

    im = Image.new("RGB", (W2 * SS, H2 * SS), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    lw = max(1, SS)  # ~1px final line width
    for p in sol["placements"]:
        z = (p["x"] + 1j * p["y"]) * np.exp(1j * phi)
        verts = z + np.exp(1j * (p["angle"] + phi + 2 * np.pi * np.arange(nsi) / nsi))
        dr.polygon([to_px(v) for v in verts], fill=color, outline=(0, 0, 0), width=lw)
    dr.polygon([to_px(c) for c in cont], outline=(0, 0, 0), width=lw)
    im = im.resize((W2, H2), Image.LANCZOS)
    # tight crop: no white border (per Erich's request)
    import numpy as _np
    arr = _np.asarray(im.convert("L"))
    nz = _np.argwhere(arr < 250)
    (y0, x0), (y1, x1) = nz.min(0), nz.max(0) + 1
    im = im.crop((int(x0), int(y0), int(x1), int(y1)))
    im.convert("P", palette=Image.ADAPTIVE, colors=64).save(out_path)
    return out_path


if __name__ == "__main__":
    print(render(sys.argv[1], sys.argv[2], sys.argv[3]))
