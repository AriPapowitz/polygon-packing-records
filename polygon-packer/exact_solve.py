"""Exact-value pipeline: refine a converged packing to arbitrary precision and
hunt its minimal polynomial.

Steps: extract active contacts (vertex-edge, vertex-wall) -> build the KKT
system for minimizing container size S on the contact manifold (contacts +
force balance) -> mpmath Gauss/Newton from the float64 start to ~60 digits ->
PSLQ the refined S against powers of itself (minimal polynomial) and against
relevant number-field bases.

Usage: python exact_solve.py results/sweep_WIN_squ_in_oct_26_polished.json
Writes findings to results/exact_<name>.md
"""

import json
import os
import sys

import numpy as np
from mpmath import mp, mpf, matrix, cos, sin, sqrt, qr_solve, lu_solve, norm, pslq

mp.dps = 160
TOUCH = 1e-7


def load(path):
    sol = json.load(open(path))
    n, nsi, nsc = sol["inner_polygons"], sol["inner_sides"], sol["container_sides"]
    assert nsi == 4, "squares only for now"
    vals = [(p["x"], p["y"], p["angle"]) for p in sol["placements"]]
    return sol, n, nsc, vals, sol["container_circumradius"]


def f64_contacts(n, nsc, vals, S):
    """Enumerate active contacts once in float64: returns
    walls = [(i, vertex_k, wall_c)], pairs = [(i, vk, j, edge_e)] with the
    convention vertex of first shape touches edge of second."""
    import validate_packing as vp
    geom = vp.build_geometry(4, nsc)
    x = np.array(vals).ravel()
    verts, normals = vp.transform_all(x, geom["inner_vertices"], geom["inner_normals"])
    limit = S * geom["container_apothem"]
    walls = []
    for i in range(n):
        proj = verts[i] @ geom["container_normals"].T
        for k in range(4):
            for c in range(nsc):
                if limit - proj[k, c] < TOUCH:
                    walls.append((i, k, c))
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # vertex k of i against edge e of j (edge e has normal normals[j][e])
            for k in range(4):
                for e in range(4):
                    d = (verts[i][k] - verts[j][e]) @ normals[j][e]
                    # vertex touches edge plane from outside, and within edge span
                    if abs(d) < TOUCH:
                        t = (verts[i][k] - verts[j][e]) @ (verts[j][(e + 1) % 4] - verts[j][e])
                        L = np.sum((verts[j][(e + 1) % 4] - verts[j][e]) ** 2)
                        if -1e-9 <= t <= L + 1e-9:
                            pairs.append((i, k, j, e))
    # dedupe: one constraint per (vertex, target square) - a vertex sitting on
    # a corner of j otherwise yields two coincident rows
    seen, uniq = set(), []
    for (i, k, j, e) in pairs:
        if (i, k, j) not in seen:
            seen.add((i, k, j))
            uniq.append((i, k, j, e))
    return walls, uniq


def solve(path):
    sol, n, nsc, vals64, S64 = load(path)
    walls, pairs = f64_contacts(n, nsc, vals64, S64)
    active = sorted(set([w[0] for w in walls] + [p[0] for p in pairs] + [p[2] for p in pairs]))
    idx = {i: t for t, i in enumerate(active)}
    m = len(active)
    print(f"{n} squares, {m} load-bearing, {len(walls)} wall + {len(pairs)} pair contacts")

    # unknowns: u = [x_i, y_i, th_i for active] + [S]; lambdas for each contact
    nu = 3 * m + 1
    nc = len(walls) + len(pairs)

    wall_n = [(cos(2 * mpf(c) * mp.pi / nsc + mp.pi / nsc),
               sin(2 * mpf(c) * mp.pi / nsc + mp.pi / nsc)) for c in range(nsc)]
    apo = cos(mp.pi / mpf(nsc))

    def vert(u, i, k):
        t = idx[i] * 3
        a = u[t + 2] + mp.pi / 4 + k * mp.pi / 2   # square vertex angles: th+45+90k
        # engine convention: vertices at angle th + k*pi/2 from +x, circumradius 1
        a = u[t + 2] + k * mp.pi / 2
        return u[t] + cos(a), u[t + 1] + sin(a)

    def normal(u, j, e):
        t = idx[j] * 3
        a = u[t + 2] + e * mp.pi / 2 + mp.pi / 4   # edge normals bisect vertices
        return cos(a), sin(a)

    def constraints(u):
        F = []
        S = u[-1]
        for (i, k, c) in walls:
            vx, vy = vert(u, i, k)
            F.append(vx * wall_n[c][0] + vy * wall_n[c][1] - apo * S)
        for (i, k, j, e) in pairs:
            vx, vy = vert(u, i, k)
            ex, ey = vert(u, j, e)
            nx, ny = normal(u, j, e)
            F.append((vx - ex) * nx + (vy - ey) * ny)
        return F

    def jac_num(u, F0, h=None):
        h = h or mpf(10) ** (-mp.dps // 2)
        J = matrix(nc, nu)
        for a in range(nu):
            u2 = list(u)
            u2[a] = u2[a] + h
            F1 = constraints(u2)
            for b in range(nc):
                J[b, a] = (F1[b] - F0[b]) / h
        return J

    u = []
    for i in active:
        u += [mpf(vals64[i][0]), mpf(vals64[i][1]), mpf(vals64[i][2])]
    u.append(mpf(S64))

    # select a maximal linearly-independent constraint subset (f64 pivoted QR)
    uf = np.array([float(x) for x in u])
    def F64(uv):
        sv = list(map(mpf, uv))
        return np.array([float(f) for f in constraints(sv)])
    F0f = F64(uf)
    Jf = np.zeros((nc, nu))
    hh = 1e-7
    for a in range(nu):
        u2 = uf.copy(); u2[a] += hh
        Jf[:, a] = (F64(u2) - F0f) / hh
    from scipy.linalg import qr as sqr
    _, _, piv = sqr(Jf.T, pivoting=True, mode='economic')
    import numpy.linalg as nl
    keep = []
    for r in piv:
        trial = keep + [r]
        if nl.matrix_rank(Jf[trial], tol=1e-9) == len(trial):
            keep.append(r)
    keep = sorted(keep)
    print(f"independent constraints: {len(keep)} of {nc}")
    allc = [('w',) + w for w in walls] + [('p',) + pr for pr in pairs]
    walls = [c[1:] for t, c in enumerate(allc) if t in keep and c[0] == 'w']
    pairs = [c[1:] for t, c in enumerate(allc) if t in keep and c[0] == 'p']
    nc = len(walls) + len(pairs)

    if nc < nu:
        print(f"WARNING: underdetermined ({nc} constraints, {nu} unknowns) - manifold not a point")
    # plain Gauss-Newton on F(u) = 0 (jammed packing => isolated solution)
    for it in range(30):
        F0 = constraints(u)
        res = max(abs(x) for x in F0)
        print(f"iter {it}: |F| = {mp.nstr(res, 5)}")
        if res < mpf(10) ** (-(mp.dps - 15)):
            break
        J = jac_num(u, F0)
        rhs = matrix(nc, 1)
        for b in range(nc):
            rhs[b] = -F0[b]
        try:
            JJt = J * J.T
            y = lu_solve(JJt, rhs)
            d = J.T * y                      # minimum-norm step
        except Exception as ex:
            print("linear solve failed:", ex)
            return None
        step = max(abs(d[a]) for a in range(nu))
        damp = mpf(1) if step < mpf('0.05') else mpf('0.05') / step
        for a in range(nu):
            u[a] = u[a] + damp * d[a]
    else:
        print("no convergence")
        return None

    S_hi = u[-1]
    ratio = S_hi * sin(mp.pi / mpf(nsc)) / sin(mp.pi / 4)
    name = os.path.basename(path).replace(".json", "")
    out = [f"# Exact-solve: {name}", "",
           f"high-precision s = {mp.nstr(ratio, 65)}",
           f"(f64 was {sol['side_length']:.15f})", ""]
    v = ratio
    found = False
    for deg in range(2, 9):
        rel = pslq([v ** k for k in range(deg + 1)], maxcoeff=10 ** 8, maxsteps=10 ** 6)
        if rel:
            out.append(f"minimal polynomial candidate (deg {deg}): {rel}")
            found = True
            break
    if not found:
        out.append("no minimal polynomial up to degree 8 / coeffs 1e8 at 60+ digits")
    al = sqrt(2 + sqrt(2))
    rel = pslq([v, mpf(1), al, al ** 2, al ** 3], maxcoeff=10 ** 6, maxsteps=10 ** 6)
    out.append(f"Q(sqrt(2+sqrt2)) relation: {rel}")
    txt = "\n".join(out)
    open(f"results/exact_{name}.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    solve(sys.argv[1])
