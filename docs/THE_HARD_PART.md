# The two stages that need real work (03 label & 04 render)

Everything else in this repo is plumbing. The thesis lives or dies on **label quality of
rendered novel-view masks**. These two stages are scaffolded with the exact I/O contract
wired; you fill the model-specific middle.

## Stage 03 — lifting 2D masks into the 3D scene

Goal: assign a persistent **instance id** to each Gaussian, plus a global
`{instance_id -> class_id}` table.

Two approaches, pick one for the prototype:

**A) Geometric voting (start here — no retraining).**
For each Gaussian center, project into every seed view that has masks. The instance whose
2D mask covers the projection most often wins. Cheap, runs in minutes, good enough to
prove the pipeline. Weakness: bleeds at occlusion boundaries.

**B) Feature-field optimization (Gaussian Grouping / SAGA).**
Give each Gaussian a learnable instance embedding, render it, and optimize so the rendered
instance map matches the SAM2 seed masks (with SAM2 video tracking giving cross-frame id
consistency). Cleaner boundaries, more work. This is the "real" version.

Output contract (what stage 04 reads):
```
labels/instances/gaussian_instance_ids.npy   # int[num_gaussians]
labels/instances/instance_classes.json       # {instance_id: class_id}
labels/classes.json                           # [class names]   (written by the scaffold)
```

## Stage 04 — instance-aware rendering

Goal: a `render(c2w, W, H) -> (rgb, instance_map)` that rasterizes both appearance and the
per-Gaussian instance id.

gsplat can rasterize arbitrary per-Gaussian channels. Render the instance-id channel with
nearest / max-alpha compositing (not alpha-blending — you want a crisp argmax per pixel,
not a blended id). Then `src.common.coco.CocoWriter` turns `(instance_map, inst_to_class)`
into COCO boxes + RLE masks automatically.

Load the trained model from `splat/LATEST_CONFIG` via nerfstudio's `eval_setup`.

## Validate masks BEFORE training anything

The #1 failure mode is silently-wrong masks from floaters/imperfect lifting poisoning the
detector. Before stage 05, overlay a sample of rendered masks on rendered RGB and eyeball
them. A tiny `tools/qa_overlay.py` that saves N overlays is the cheapest insurance in the
whole project.

## Suggested order of attack

1. Stage 03 approach **A** (voting) + stage 04 render → get *any* labeled views out.
2. QA overlay. If masks look clean, run 05/06 and read the real-frame mAP.
3. Only then upgrade stage 03 to approach **B** if boundaries are too rough.
4. Differentiator: close the loop — have stage 06 surface low-AP viewpoints and feed them
   back as targeted `render.trajectory` samples (active data generation).
