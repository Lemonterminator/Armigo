# Brick surface anomaly detection on simulated crops (v1)

Simulation only. Crops of the held brick at the expert's first show hold and at random virtual show poses (MuJoCo, `scripts/anomaly/render_dataset.py`), detectors from anomalib trained on normal crops only (`scripts/anomaly/train_eval.py`). Full numbers: `report-v1.json`.

Device: cuda, torch 2.11.0+cu128, anomalib 2.7.0; input 224 px; threshold = 0.95 quantile of validation normal scores.

Data (crops): fit 4734 / val 845 normal (171 / 30 layouts); test 5070 (184 layouts).

Test crops by label: test/abnormal=871, test/hidden_defect=1733, test/normal=2466, train/normal=5579

## Overall (test: normal vs abnormal)

| model / scope | n normal | n abnormal | AUROC | F1 @thr | precision | recall | FPR | best F1 (oracle thr) |
|---|---|---|---|---|---|---|---|---|
| padim/top | 919 | 293 | 0.500 | 0.102 | 0.305 | 0.061 | 0.045 | 0.394 |
| padim/wrist_cam | 1547 | 578 | 0.806 | 0.581 | 0.816 | 0.452 | 0.038 | 0.639 |
| padim/pooled | 2466 | 871 | 0.616 | 0.121 | 0.403 | 0.071 | 0.037 | 0.455 |
| patchcore/top | 919 | 293 | 0.629 | 0.261 | 0.520 | 0.174 | 0.051 | 0.427 |
| patchcore/wrist_cam | 1547 | 578 | 0.828 | 0.571 | 0.767 | 0.455 | 0.052 | 0.637 |
| patchcore/pooled | 2466 | 871 | 0.748 | 0.344 | 0.634 | 0.236 | 0.048 | 0.535 |
| patchcore_fine/top | 919 | 293 | 0.640 | 0.161 | 0.433 | 0.099 | 0.041 | 0.431 |
| patchcore_fine/wrist_cam | 1547 | 578 | 0.813 | 0.428 | 0.678 | 0.313 | 0.056 | 0.626 |
| patchcore_fine/pooled | 2466 | 871 | 0.721 | 0.289 | 0.609 | 0.189 | 0.043 | 0.512 |

## AUROC by defect kind (abnormal of that stratum vs all test normals)

| model / scope | chip | cross | discolor | scuff | tape |
|---|---|---|---|---|---|
| padim/top | 0.528 (n=53) | 0.478 (n=67) | 0.524 (n=82) | 0.452 (n=40) | 0.499 (n=51) |
| padim/wrist_cam | 0.712 (n=85) | 0.996 (n=102) | 0.835 (n=119) | 0.652 (n=136) | 0.851 (n=136) |
| padim/pooled | 0.569 (n=138) | 0.827 (n=169) | 0.609 (n=201) | 0.501 (n=176) | 0.577 (n=187) |
| patchcore/top | 0.551 (n=53) | 0.799 (n=67) | 0.569 (n=82) | 0.582 (n=40) | 0.621 (n=51) |
| patchcore/wrist_cam | 0.739 (n=85) | 1.000 (n=102) | 0.775 (n=119) | 0.755 (n=136) | 0.872 (n=136) |
| patchcore/pooled | 0.669 (n=138) | 0.961 (n=169) | 0.698 (n=201) | 0.639 (n=176) | 0.769 (n=187) |
| patchcore_fine/top | 0.586 (n=53) | 0.795 (n=67) | 0.580 (n=82) | 0.589 (n=40) | 0.627 (n=51) |
| patchcore_fine/wrist_cam | 0.717 (n=85) | 0.993 (n=102) | 0.725 (n=119) | 0.780 (n=136) | 0.848 (n=136) |
| patchcore_fine/pooled | 0.660 (n=138) | 0.943 (n=169) | 0.635 (n=201) | 0.648 (n=176) | 0.725 (n=187) |

## AUROC by defect face (0 +x, 1 -x, 2 +y, 3 -y, 4 top, 5 bottom) (abnormal of that stratum vs all test normals)

| model / scope | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| padim/top | 0.476 (n=50) | 0.481 (n=48) | 0.530 (n=29) | 0.465 (n=40) | 0.485 (n=79) | 0.581 (n=47) |
| padim/wrist_cam | 0.949 (n=51) | 0.747 (n=68) | 0.790 (n=85) | - | 0.815 (n=357) | 0.500 (n=17) |
| padim/pooled | 0.701 (n=101) | 0.583 (n=116) | 0.565 (n=114) | 0.733 (n=40) | 0.598 (n=436) | 0.682 (n=64) |
| patchcore/top | 0.663 (n=50) | 0.576 (n=48) | 0.670 (n=29) | 0.624 (n=40) | 0.642 (n=79) | 0.606 (n=47) |
| patchcore/wrist_cam | 0.952 (n=51) | 0.689 (n=68) | 0.841 (n=85) | - | 0.850 (n=357) | 0.482 (n=17) |
| patchcore/pooled | 0.875 (n=101) | 0.635 (n=116) | 0.750 (n=114) | 0.825 (n=40) | 0.752 (n=436) | 0.671 (n=64) |
| patchcore_fine/top | 0.716 (n=50) | 0.609 (n=48) | 0.714 (n=29) | 0.637 (n=40) | 0.617 (n=79) | 0.585 (n=47) |
| patchcore_fine/wrist_cam | 0.940 (n=51) | 0.634 (n=68) | 0.810 (n=85) | - | 0.846 (n=357) | 0.463 (n=17) |
| patchcore_fine/pooled | 0.841 (n=101) | 0.621 (n=116) | 0.748 (n=114) | 0.769 (n=40) | 0.720 (n=436) | 0.636 (n=64) |

## AUROC by visible defect pixels (abnormal of that stratum vs all test normals)

| model / scope | 150+ | 25-60 | 60-150 |
|---|---|---|---|
| padim/top | 0.525 (n=26) | 0.469 (n=128) | 0.523 (n=139) |
| padim/wrist_cam | 0.840 (n=510) | 0.454 (n=17) | 0.577 (n=51) |
| padim/pooled | 0.577 (n=536) | 0.694 (n=145) | 0.668 (n=190) |
| patchcore/top | 0.796 (n=26) | 0.563 (n=128) | 0.659 (n=139) |
| patchcore/wrist_cam | 0.865 (n=510) | 0.403 (n=17) | 0.598 (n=51) |
| patchcore/pooled | 0.764 (n=536) | 0.718 (n=145) | 0.725 (n=190) |
| patchcore_fine/top | 0.758 (n=26) | 0.589 (n=128) | 0.664 (n=139) |
| patchcore_fine/wrist_cam | 0.851 (n=510) | 0.354 (n=17) | 0.585 (n=51) |
| patchcore_fine/pooled | 0.730 (n=536) | 0.695 (n=145) | 0.715 (n=190) |

## AUROC by camera (abnormal of that stratum vs all test normals)

| model / scope | top | wrist_cam |
|---|---|---|
| padim/top | 0.500 (n=293) | - |
| padim/wrist_cam | - | 0.806 (n=578) |
| padim/pooled | 0.479 (n=293) | 0.721 (n=578) |
| patchcore/top | 0.629 (n=293) | - |
| patchcore/wrist_cam | - | 0.828 (n=578) |
| patchcore/pooled | 0.618 (n=293) | 0.826 (n=578) |
| patchcore_fine/top | 0.640 (n=293) | - |
| patchcore_fine/wrist_cam | - | 0.813 (n=578) |
| patchcore_fine/pooled | 0.619 (n=293) | 0.795 (n=578) |

## Hidden defects (defective brick, < 25 defect pixels in view)

| model / scope | n | AUROC vs normal | frac above thr | normal frac above thr | median (normal median) |
|---|---|---|---|---|---|
| padim/top | 730 | 0.510 | 0.059 | 0.045 | 21.552 (21.340) |
| padim/wrist_cam | 1003 | 0.503 | 0.057 | 0.038 | 22.038 (21.936) |
| padim/pooled | 1733 | 0.520 | 0.062 | 0.037 | 21.444 (20.873) |
| patchcore/top | 730 | 0.502 | 0.071 | 0.051 | 56.436 (56.328) |
| patchcore/wrist_cam | 1003 | 0.483 | 0.059 | 0.052 | 45.848 (46.201) |
| patchcore/pooled | 1733 | 0.511 | 0.063 | 0.048 | 52.873 (52.747) |
| patchcore_fine/top | 730 | 0.516 | 0.056 | 0.041 | 41.932 (41.768) |
| patchcore_fine/wrist_cam | 1003 | 0.487 | 0.049 | 0.056 | 35.513 (35.669) |
| patchcore_fine/pooled | 1733 | 0.510 | 0.053 | 0.043 | 40.107 (39.919) |

Exported model: `patchcore/pooled` -> `models/best.pt` (see `score.py`).


## How the data was made

- 600 layouts from `bench_physics_expert.sample_layouts` (seed 2027) with defect face uniform over 6 faces, defect kind
  uniform over cross / tape / chip / scuff / discolor, defective with p = 0.5. Split BY LAYOUT (70 / 30); train
  layouts keep normal crops only (defective train layouts are not rendered). Threshold validation normals are a
  further by-layout split of the train layouts.
- Per rendered layout: both cameras at the expert's first real show hold plus 16 random virtual show poses
  (flex in [-1.2, 1.6] rad, roll in [-150, 150] deg, kept only if arm and brick stay >= 2 cm above the table),
  rendered on a kinematic copy (`HeldBrickPoser`). Crops 128 px, upsampled to 224 px for the backbones.
- Regenerated after the `HeldBrickPoser.at()` fix (it now calls `mj_camlight`): an earlier dataset rendered every
  virtual wrist_cam view from the camera pose of the real show hold, so its virtual wrist crops did not match their
  pose metadata. All numbers here are from the regenerated dataset.
- Crop box: the renderer computes its own brick bounding box (`render_dataset._brick_bbox`: brick mask from a
  default-groups segmentation render, closed with a 5x5 kernel, components with >= 20 % of the largest one's area
  kept), NOT the `bbox` of `Cell.face_visibility`. Pixel counts (defect_px, faces_px, brick_px) do come from
  `face_visibility` (with its speck filter: components < max(10 px, 10 % of the largest) dropped).
- Labels: abnormal = defective and >= 25 defect pixels visible in the full camera image (not in the 128 px crop);
  hidden_defect = defective but < 25 px (excluded from AUROC); crops with < 150 brick px are skipped.

## Caveats

- Simulation only, synthetic defects drawn as flat visual-only patches (MuJoCo box / cylinder geoms 0.3 mm off the
  face), MuJoCo's simple renderer (no texture noise, no motion blur, no sensor noise). Nothing here measures
  performance on real D415 / wrist camera images or real defects.
- The task is much harder than the Junction-era cross-only setup: the brick appears at any pose, partly occluded by
  the jaws, against varied backgrounds, and a defect may cover only 25 px of a full camera frame.
- top camera: the brick is small (often ~1000-2000 px) and seen at arbitrary orientation. PaDiM, which models each
  image position separately, is at chance there (crops not aligned); PatchCore reaches only ~0.63.
- wrist_cam: the camera rides on the gripper, so the held brick sits at a nearly fixed place in the image (crops
  are close to aligned, which is why PaDiM works here) and most abnormal wrist crops show the defect large (150+
  px). That gives ~0.81-0.83 AUROC; wrist crops with 25-150 defect px stay near chance.
- The cross (dark, high contrast) is trivially detectable from the wrist camera (AUROC 0.99-1.0) and still the
  easiest kind overall; chip and scuff are the hardest. A cross-only benchmark therefore overstates what a detector
  can do; report the per-kind rows. AUROC per stratum is over ~20-500 abnormal crops; a few points are noise.
- Pooled models: per-stratum AUROC compares that stratum's abnormal crops with ALL test normals of both cameras, and
  the two cameras have different score scales (e.g. PatchCore normal median 56 top vs 46 wrist). Strata dominated by
  one camera (the 25-60 px bin is mostly top) are confounded by that; read the per-camera models for those.
- The F1 threshold (0.95 quantile of validation normals) gives FPR ~5 % on test normals, so recall is low wherever
  AUROC is modest. "best F1 (oracle thr)" is chosen on the test set and is only an upper bound.
- Leakage checks: split is by layout; normal and defective layouts come from the same sampler; hidden defects of a
  normal brick sit in a geom group that is never rendered in colour (a test asserts the colour image is identical
  across hidden kinds). hidden_defect crops score like normals (AUROC ~ 0.5), which is what one expects if nothing
  but the visible defect separates the classes.
- Virtual poses dominate (~88 % of crops); they are rigid kinematic copies of the held brick and do not include the
  motion of a real show move.



## Supervised v6 brick-level fusion and physics planner

The v6 / v6def comparison uses independent MuJoCo layouts, with 212 normal training bricks for score calibration
and 304 defective plus 84 normal test bricks. The detector uses its classification and segmentation heads.
The historical pipeline views cover at least 25 visible defect pixels on at least one frame of 304/304 defective
bricks. Max fusion detected all 304, giving a Wilson lower bound of 0.9875 and a one-sided exact 95% miss upper
bound of 0.98%. False alarms range from about 2% to 8% across the operating points. A rate of 7/84 remains
imprecise. k2 and top2 also detected 304/304 at the tested thresholds. k3 reduced observed recall to about
0.96–0.97. The offline set covers defect faces 0–4. See [fusion rows](sup-v6/fuse-pipeline-both.json).

On Isaac v4-isaac2 the brick AUROC was 0.994–0.998. Transferring the MuJoCo threshold increased false alarms to
43–62%. Recalibration on 136 Isaac normal bricks yielded 64/64 detection and about 8–10% false alarms on the 60
normal test bricks. The calibration set included 116 layouts whose MuJoCo render had been used for training.
With only 64 positive bricks, the one-sided 95% miss upper bound is about 4.6%.
See [Isaac fusion rows](sup-v6/fuse-isaac-both.json).

In the n=24 seed 23 MuJoCo planner experiment, supervised heuristic / heuristic_fast achieved 24/24 label
correctness and 23/24 audited physical successes, matching both oracle variants. PatchCore on the same strategies
reached 15/24 correct labels and 14/24 audited successes. Supervised show reached 13/24 physical successes.
Its recorded final confusion has 19 known decisions, all correct, and five normal bricks marked unknown.
See [episode summary](../heuristic-inspection/sup-n24/summary.json).

A separate six-face project measures the bottom view and online six-face coverage under its own camera profile.
The experiments above do not validate the new bottom view.
