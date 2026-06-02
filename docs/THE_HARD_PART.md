# Stages 03 (label) & 04 (render) — design notes

These two stages are now **implemented**, not stubbed. This doc explains *how* they work,
the approximations they make, and what to tune when the masks look wrong. The thesis lives
or dies on **label quality of rendered novel-view masks**, so read this before trusting a
result.

## Coordinate frame (why both stages load the nerfstudio pipeline)

nerfstudio re-orients and re-scales the COLMAP poses, so the trained Gaussians live in a
*normalized* frame — not COLMAP's. Anything that mixes raw COLMAP poses with the Gaussians
will be silently misaligned. Both stages therefore load the trained pipeline via
`src/common/nerfstudio_io.py` and use its **training cameras** (same frame as the
Gaussians) for projection and intrinsics.

## Stage 03 — labeling

**(a) Seed** (`src/label/sam2_seed.py`): SAM2 automatic mask generator proposes
class-agnostic masks on every Nth training view; open_clip classifies each masked crop
against your `label.classes` (+ a "background" prompt that rejects junk). Output per view
is a semantic label map (0 bg, 1..C).

**(b) Vote** (`src/label/lift.py::vote_classes`): every Gaussian center is projected into
every seeded view; the class covering it accumulates a vote. Each Gaussian's class =
argmax votes, gated by `vote.min_votes` and required to beat background.

**(c) Split into instances** (`lift.py::cluster_instances`): within each class, DBSCAN on
Gaussian positions separates spatially-distinct objects into instances (two chairs across
the room → two instances). This avoids fragile cross-frame 2D instance tracking — identity
comes from 3D geometry.

### Known approximations (tune these first if masks are bad)
- **No occlusion test in voting.** A Gaussian behind a wall still projects into a
  foreground mask, so it can get mislabeled. Mitigations: rely on multi-view voting +
  `vote.min_votes`; or add a depth test (render depth per seed view, reject Gaussians far
  behind the surface). The depth test is the highest-value upgrade.
- **CLIP crop classification is noisy.** Raise `clip.score_threshold` for precision (fewer
  but cleaner labels) or lower it for recall. For a tightly-scoped prototype, swapping CLIP
  for manual prompts or Grounded-SAM (text→box→mask with a real label) is more reliable.
- **DBSCAN `eps`** is `instance.dbscan_eps_scale * scene_radius`. Too small → one object
  splits into many; too large → adjacent objects merge. Tune per scene.

### Alternative lift (if voting boundaries are too rough)
Feature-field optimization (Gaussian Grouping / SAGA): give each Gaussian a learnable
instance embedding and optimize rendered instance maps to match the seed masks (with SAM2
video tracking for cross-frame id consistency). Cleaner boundaries, more work. Swap it in
behind the same output contract below.

## Stage 04 — instance-aware rendering

`src/render/engine.py` does two gsplat passes per view with identical geometry:
1. **RGB** from SH colors (matches splatfacto appearance).
2. **Instance**: a one-hot-per-instance feature buffer; per pixel we take `argmax` over the
   instance channels above `render.instance_alpha_thresh`. Argmax (not alpha-blend) keeps
   masks crisp. Background where coverage/alpha is below threshold.

`src/common/coco.py` turns `(instance_map, instance→class)` into COCO boxes + RLE masks.

### Known approximations
- **One-hot buffer is (N_gaussians × N_instances).** Fine for a room (tens of instances) on
  96 GB. If instance count explodes, render instances in batches and merge argmax.
- **Trajectory `up = +y`.** Assumes nerfstudio's default orientation put world-up near +y.
  Usually true; if novel views look tilted, switch `render.trajectory` to `jitter_input`
  (perturbs real poses — safest, stays in well-reconstructed regions) or fix the up vector.

## Output contract (the seam between 03 and 04)
```
data/<scene>/labels/instances/gaussian_instance_ids.npy   int[num_gaussians]  0 = bg
data/<scene>/labels/instances/instance_classes.json       {instance_id: class_index}
data/<scene>/labels/classes.json                           [class names]
```

## Validate BEFORE training — non-negotiable
```
SCENE=<scene> bash scripts/qa.sh 40        # tools/qa_overlay.py
```
Eyeball `data/<scene>/render/qa/*.jpg`: masks should hug object boundaries with no
floaters, and class labels should be right. If they're not, fix stage 03 (thresholds /
occlusion / eps) before spending GPU-hours on stage 05.

## The differentiator (next step after it works)
Close the loop: have stage 06 surface low-AP viewpoints, then bias `render.trajectory`
sampling toward those failure modes and regenerate — active data generation, not just
"render more views."
