# Brick surface anomaly detection on simulated crops (v1)

Simulation only. Crops of the held brick at the expert's first show hold and at random virtual show poses (MuJoCo, `scripts/anomaly/render_dataset.py`), detectors from anomalib trained on normal crops only (`scripts/anomaly/train_eval.py`). Full numbers: `report-v1.json`.

Device: cuda, torch 2.11.0+cu128, anomalib 2.7.0; input 256 px; threshold = 0.99 quantile of validation normal scores.

Data (crops): fit 4310 / val 746 normal (116 / 20 layouts); test 4579 (124 layouts).

Test crops by label: test/abnormal=960, test/hidden_defect=1391, test/normal=2228, train/normal=5056

## Overall (test: normal vs abnormal)

| model / scope | n normal | n abnormal | AUROC | F1 @thr | precision | recall | FPR | best F1 (oracle thr) |
|---|---|---|---|---|---|---|---|---|
| patchcore/top | 990 | 323 | 0.641 | 0.160 | 0.744 | 0.090 | 0.010 | 0.433 |
| patchcore/wrist_cam | 1238 | 637 | 0.853 | 0.589 | 0.964 | 0.424 | 0.008 | 0.752 |
| patchcore/pooled | 2228 | 960 | 0.788 | 0.463 | 0.908 | 0.310 | 0.013 | 0.622 |

## AUROC by defect kind (abnormal of that stratum vs all test normals)

| model / scope | cross_var |
|---|---|
| patchcore/top | 0.641 (n=323) |
| patchcore/wrist_cam | 0.853 (n=637) |
| patchcore/pooled | 0.788 (n=960) |

## AUROC by defect face (0 +x, 1 -x, 2 +y, 3 -y, 4 top, 5 bottom) (abnormal of that stratum vs all test normals)

| model / scope | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| patchcore/top | 0.711 (n=87) | 0.612 (n=91) | 0.634 (n=68) | 0.665 (n=44) | 0.518 (n=33) |
| patchcore/wrist_cam | 0.867 (n=67) | 0.785 (n=165) | 0.751 (n=105) | 0.548 (n=35) | 0.974 (n=265) |
| patchcore/pooled | 0.784 (n=154) | 0.726 (n=256) | 0.709 (n=173) | 0.636 (n=79) | 0.929 (n=298) |

## AUROC by visible defect pixels (abnormal of that stratum vs all test normals)

| model / scope | 150+ | 25-60 | 60-150 |
|---|---|---|---|
| patchcore/top | 0.856 (n=17) | 0.519 (n=161) | 0.751 (n=145) |
| patchcore/wrist_cam | 0.970 (n=447) | 0.489 (n=117) | 0.721 (n=73) |
| patchcore/pooled | 0.955 (n=464) | 0.536 (n=278) | 0.754 (n=218) |

## AUROC by camera (abnormal of that stratum vs all test normals)

| model / scope | top | wrist_cam |
|---|---|---|
| patchcore/top | 0.641 (n=323) | - |
| patchcore/wrist_cam | - | 0.853 (n=637) |
| patchcore/pooled | 0.640 (n=323) | 0.853 (n=637) |

## AUROC by stage (abnormal of that stage vs normals of that stage)

| model / scope | closeup | prescan | show | virtual_show |
|---|---|---|---|---|
| patchcore/top | 0.612 (n_abn=172, n_norm=480) | - | 0.731 (n_abn=18, n_norm=58) | 0.655 (n_abn=133, n_norm=452) |
| patchcore/wrist_cam | - | 0.745 (n_abn=247, n_norm=458) | 0.908 (n_abn=30, n_norm=60) | 0.883 (n_abn=360, n_norm=720) |
| patchcore/pooled | 0.614 (n_abn=172, n_norm=480) | 0.757 (n_abn=247, n_norm=458) | 0.853 (n_abn=48, n_norm=118) | 0.829 (n_abn=493, n_norm=1172) |

## Hidden defects (defective brick, < 25 defect pixels in view)

| model / scope | n | AUROC vs normal | frac above thr | normal frac above thr | median (normal median) |
|---|---|---|---|---|---|
| patchcore/top | 712 | 0.480 | 0.007 | 0.010 | 57.463 (58.052) |
| patchcore/wrist_cam | 679 | 0.527 | 0.012 | 0.008 | 54.341 (53.747) |
| patchcore/pooled | 1391 | 0.524 | 0.015 | 0.013 | 57.971 (57.412) |

Exported model: `patchcore/pooled` -> `models/best.pt` (see `score.py`).

