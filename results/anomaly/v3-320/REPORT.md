# Brick surface anomaly detection on simulated crops (v1)

Simulation only. Crops of the held brick at the expert's first show hold and at random virtual show poses (MuJoCo, `scripts/anomaly/render_dataset.py`), detectors from anomalib trained on normal crops only (`scripts/anomaly/train_eval.py`). Full numbers: `report-v1.json`.

Device: cuda, torch 2.11.0+cu128, anomalib 2.7.0; input 320 px; threshold = 0.99 quantile of validation normal scores.

Data (crops): fit 4310 / val 746 normal (116 / 20 layouts); test 4579 (124 layouts).

Test crops by label: test/abnormal=960, test/hidden_defect=1391, test/normal=2228, train/normal=5056

## Overall (test: normal vs abnormal)

| model / scope | n normal | n abnormal | AUROC | F1 @thr | precision | recall | FPR | best F1 (oracle thr) |
|---|---|---|---|---|---|---|---|---|
| patchcore/top | 990 | 323 | 0.675 | 0.214 | 0.800 | 0.124 | 0.010 | 0.452 |
| patchcore/wrist_cam | 1238 | 637 | 0.861 | 0.677 | 0.939 | 0.529 | 0.018 | 0.754 |
| patchcore/pooled | 2228 | 960 | 0.797 | 0.456 | 0.918 | 0.303 | 0.012 | 0.632 |

## AUROC by defect kind (abnormal of that stratum vs all test normals)

| model / scope | cross_var |
|---|---|
| patchcore/top | 0.675 (n=323) |
| patchcore/wrist_cam | 0.861 (n=637) |
| patchcore/pooled | 0.797 (n=960) |

## AUROC by defect face (0 +x, 1 -x, 2 +y, 3 -y, 4 top, 5 bottom) (abnormal of that stratum vs all test normals)

| model / scope | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| patchcore/top | 0.755 (n=87) | 0.640 (n=91) | 0.694 (n=68) | 0.717 (n=44) | 0.464 (n=33) |
| patchcore/wrist_cam | 0.875 (n=67) | 0.799 (n=165) | 0.758 (n=105) | 0.596 (n=35) | 0.972 (n=265) |
| patchcore/pooled | 0.816 (n=154) | 0.738 (n=256) | 0.706 (n=173) | 0.666 (n=79) | 0.926 (n=298) |

## AUROC by visible defect pixels (abnormal of that stratum vs all test normals)

| model / scope | 150+ | 25-60 | 60-150 |
|---|---|---|---|
| patchcore/top | 0.861 (n=17) | 0.548 (n=161) | 0.794 (n=145) |
| patchcore/wrist_cam | 0.974 (n=447) | 0.506 (n=117) | 0.737 (n=73) |
| patchcore/pooled | 0.955 (n=464) | 0.551 (n=278) | 0.775 (n=218) |

## AUROC by camera (abnormal of that stratum vs all test normals)

| model / scope | top | wrist_cam |
|---|---|---|
| patchcore/top | 0.675 (n=323) | - |
| patchcore/wrist_cam | - | 0.861 (n=637) |
| patchcore/pooled | 0.654 (n=323) | 0.859 (n=637) |

## AUROC by stage (abnormal of that stage vs normals of that stage)

| model / scope | closeup | prescan | show | virtual_show |
|---|---|---|---|---|
| patchcore/top | 0.656 (n_abn=172, n_norm=480) | - | 0.740 (n_abn=18, n_norm=58) | 0.685 (n_abn=133, n_norm=452) |
| patchcore/wrist_cam | - | 0.784 (n_abn=247, n_norm=458) | 0.896 (n_abn=30, n_norm=60) | 0.882 (n_abn=360, n_norm=720) |
| patchcore/pooled | 0.628 (n_abn=172, n_norm=480) | 0.776 (n_abn=247, n_norm=458) | 0.842 (n_abn=48, n_norm=118) | 0.840 (n_abn=493, n_norm=1172) |

## Hidden defects (defective brick, < 25 defect pixels in view)

| model / scope | n | AUROC vs normal | frac above thr | normal frac above thr | median (normal median) |
|---|---|---|---|---|---|
| patchcore/top | 712 | 0.498 | 0.007 | 0.010 | 58.618 (58.683) |
| patchcore/wrist_cam | 679 | 0.526 | 0.016 | 0.018 | 54.018 (53.474) |
| patchcore/pooled | 1391 | 0.525 | 0.014 | 0.012 | 58.152 (57.680) |

Exported model: `patchcore/pooled` -> `models/best.pt` (see `score.py`).

