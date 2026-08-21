# polypack API reference

Everything lives in the `polypack` package. Conventions used throughout:

- The **container** is a regular `nsc`-gon centered at the origin with a
  vertex on the +x axis and circumradius `S`.
- Each **inner shape** is a unit-circumradius regular `nsi`-gon placed by
  `(x, y, angle)`; a packing of `n` shapes is a flat vector of length `3n`.
- The **side ratio** (the number the Packing Center tables list) is
  `s = S · sin(π/nsc) / sin(π/nsi)` — container side over inner side.
- **Solution JSONs** store `inner_polygons`, `inner_sides`,
  `container_sides`, `container_circumradius`, `side_length`, and
  `placements: [{x, y, angle}, …]`.

Lazy top-level re-exports (`from polypack import …`): `Engine`,
`load_solution`, `save_solution`, `build_geometry`, `exact_margins`,
`pair_separation`, `make_penalty`, `certify`, `polish`, `squeeze`.
`import polypack` itself is cheap; JAX/numba load on first use.

---

## polypack.pack_core — the batched search engine

### `Engine(n, nsi, nsc, dtype=jnp.float32)`

Batched JAX engine for `n` unit `nsi`-gons in an `nsc`-gon container.
Useful attributes: `ratio` (S→s conversion), `apothem` (container apothem
for circumradius 1), `lowest_S` (area lower bound on S), `D = 3n`.

- **`Engine.lbfgs(x, S, max_iters, vg=None)`** — natively batched L-BFGS
  (memory 8, Armijo backtracking, per-instance masks) on the packing
  penalty. `x: (B, 3n)`, `S: (B,)`; runs in `x.dtype`. Returns `(x, penalty)`.
- **`Engine.make_anneal(tol, final_step=1e-4, kicks=10, kick_size=0.1,
  lbfgs_iters=250, max_outer=8000)`** — returns a jitted
  `anneal(key, x, S) -> (best_S, best_x)`: alternately minimize the penalty
  and shrink the container, with random rescue kicks for stalled instances;
  instances that fail to re-converge retire with `S = inf`.
- **`Engine.refine64(x, S, iters=400, grow_rounds=60, squeeze_rounds=60,
  squeeze_step=3e-5, grow_rate=5e-5)`** — float64 polish → grow-to-feasible
  → backtracking squeeze, all on device. Returns numpy `(S, x, valid)`;
  `valid` means the penalty (sum of squared violations) is below `1e-24`,
  i.e. residual violations of order `1e-12`. To traverse a known slack of
  `r`, give it `squeeze_rounds × squeeze_step ≳ r`.

The penalty is the squared pairwise separating-axis overlap plus squared
containment violation — differentiable, and zero exactly on valid packings.

### `load_solution(path) -> (dict, x, S)` / `save_solution(path, n, nsi, nsc, S, x, extra=None)`

Solution JSON I/O in the conventions above.

---

## polypack.validate_packing — the independent certifier

Independent of the search stack by design (numpy + numba + scipy only).

- **`build_geometry(nsi, nsc) -> dict`** — unit vertices/normals and
  apothems for the shape/container pair.
- **`transform_all(values, inner_vertices, inner_normals)`** — world-space
  vertices `(n, nsi, 2)` and edge normals for all shapes.
- **`pair_separation(verts_i, verts_j, normals_i, normals_j) -> float`** —
  exact SAT margin for one pair (positive = disjoint by that gap, negative =
  penetration depth). Exact for convex polygons.
- **`exact_margins(values, S, geom) -> (worst_pair, worst_containment,
  touching, centers_inside)`** — worst margins over all pairs and all
  vertex/wall checks; `touching` counts pairs with `|sep| < 1e-7`.
- **`make_penalty(geom, n, nsi, nsc)`** — numba-compiled scalar penalty
  matching the upstream packer's semantics.
- **`polish(values, S, penalty_fn, max_rounds=8)`** — L-BFGS-B re-minimize
  at fixed size until the penalty stalls or drops below `1e-26`.
- **`squeeze(values, S, penalty_fn, initial_step=1e-5, min_step=1e-11)`** —
  backtracking multiplicative container shrink while a machine-precision
  valid packing survives.
- **`certify(values, S, geom, nsi, nsc) -> dict`** — the rigorous bound:
  shrink inner shapes to erase any residual overlap, dilate the container to
  erase any poking, and report `certified_side_length` — a size at which the
  packing is **provably** valid — plus the exact margins and the
  certification cost. A claim is a record only if `certified_side_length`
  is strictly below the incumbent's displayed floor.

CLI: `polypack-validate solution.json [--polish] [--squeeze]`.

---

## polypack.packer_bh — structured basin-hopping (CLI)

`polypack-search n nsi nsc [options]` — population search over the batched
anneal: an elite pool sampled rank-weighted, perturbed by structured moves
(vacancy teleport, coherent cluster tilt, row shear, local shake, angle
scramble, global jiggle), refreshed by lattice and random immigrants, with
float64 certification of each round's best. Key options: `--batch`,
`--rounds`, `--elites`, `--refine-top`, `--immigrants`, `--seed-json GLOB`
(warm-start elites from solution files), `--target s` (early exit),
`--x64`. Writes `{n}_{nsi}_in_{nsc}_bh_top{k}.json` as records improve.

## polypack.drop_one — n → n−1 propagation (CLI)

`polypack-drop-one solution.json [--out-prefix P] [--save-top K]
[--iters I] [--squeeze-rounds R] [--squeeze-step X]` — delete each of the
`n` shapes in turn and float64-squeeze all `n` removal variants as one
batch; writes the best `(n−1)`-packings. One strong `(n+1)` basin can
cascade into records at `n` and below.

## polypack.reconstruct_gif — image → coordinates (CLI)

`polypack-reconstruct image.gif n nsi nsc s_claim [--out F]
[--inflate 0.005] [--debug]` — recover placements from a published packing
image: segment fills (colored, gray, or pastel pages), fit the container
similarity transform by minimizing the packing penalty, then subpixel
per-shape boundary fits (~0.1–0.5 px center accuracy at site resolution).
The output is inflated by `--inflate` so pixel noise stays feasible;
squeeze it back down with `refine64` or `polypack-validate --squeeze`.

## polypack.exact_solve — contact solving + PSLQ (CLI)

`polypack-exact solution.json` (squares only for now) — extract active
vertex–edge and vertex–wall contacts, select independent constraints by
pivoted QR, solve the contact system by minimum-norm Gauss–Newton in
160-digit mpmath arithmetic, then hunt the minimal polynomial of the
refined `s` with PSLQ. **Candidate relations count only if their residual
vanishes at ≥140 digits** — at realistic coefficient budgets PSLQ emits
spurious candidates, and the residual gate is what rejects them. Writes
`exact_<name>.md` next to the input. Library functions: `load(path)`,
`f64_contacts(n, nsc, vals, S)`, `solve(path)`.

## polypack.build_lattice / polypack.scan_removals

`polypack-lattice [--remove i j …]` — the exact 24-triangle tiling of the
side-2 hexagon and its removal variants (warm starts). Library:
`lattice_placements()`, `S`.
`polypack-scan-removals --singles 0-23 | --pairs "0,1;…" | --auto-pairs A B
| --auto-triples A B` — squeeze lattice-minus-holes configurations hunting
sub-plateau rearrangements.

## polypack.render_packing / polypack.render_webready

`polypack-render solution.json [-o out.png] [--no-title]` — quick
matplotlib rendering.
`polypack-render-webready solution.json sample.gif out.gif` — render in the
Packing Center's page style (fill color, orientation, and pixel scale
learned from a sample incumbent image; tight-cropped, no white border).

## polypack.scrape_tables

`polypack-scrape out_dir [prior_dir]` — scrape all 24 record tables to CSV
(`category, n, s, holder, year`; range rows like "41.-42." are expanded)
and print a diff against a prior scrape. Fetches via urllib with a
PowerShell fallback. Archived scrapes are how claims stay verifiable after
the live tables move.
