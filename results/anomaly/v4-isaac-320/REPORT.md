# Brick surface anomaly detection on simulated crops (v1)

Simulation only. Crops of the held brick at the expert's first show hold and at random virtual show poses (MuJoCo, `scripts/anomaly/render_dataset.py`), detectors from anomalib trained on normal crops only (`scripts/anomaly/train_eval.py`). Full numbers: `report-v1.json`.

Device: cuda, torch 2.11.0+cu128, anomalib 2.7.0; input 320 px; threshold = 0.99 quantile of validation normal scores.

Data (crops): fit 3976 / val 715 normal (116 / 20 layouts); test 4246 (124 layouts).

Test crops by label: test/abnormal=960, test/hidden_defect=1391, test/normal=2228, train/normal=5056

## Overall (test: normal vs abnormal)

| model / scope | n normal | n abnormal | AUROC | F1 @thr | precision | recall | FPR | best F1 (oracle thr) |
|---|---|---|---|---|---|---|---|---|
| patchcore/top | 840 | 260 | 0.529 | 0.114 | 0.800 | 0.061 | 0.005 | 0.383 |
| patchcore/wrist_cam | 1238 | 637 | 0.746 | 0.644 | 0.969 | 0.482 | 0.008 | 0.660 |
| patchcore/pooled | 2078 | 897 | 0.692 | 0.511 | 0.914 | 0.354 | 0.014 | 0.541 |

## AUROC by defect kind (abnormal of that stratum vs all test normals)

| model / scope | cross_var |
|---|---|
| patchcore/top | 0.529 (n=260) |
| patchcore/wrist_cam | 0.746 (n=637) |
| patchcore/pooled | 0.692 (n=897) |

## AUROC by defect face (0 +x, 1 -x, 2 +y, 3 -y, 4 top, 5 bottom) (abnormal of that stratum vs all test normals)

| model / scope | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| patchcore/top | 0.457 (n=67) | 0.565 (n=78) | 0.585 (n=50) | 0.537 (n=36) | 0.495 (n=29) |
| patchcore/wrist_cam | 0.765 (n=67) | 0.664 (n=165) | 0.697 (n=105) | 0.619 (n=35) | 0.829 (n=265) |
| patchcore/pooled | 0.654 (n=134) | 0.630 (n=243) | 0.655 (n=155) | 0.613 (n=71) | 0.798 (n=294) |

## AUROC by visible defect pixels (abnormal of that stratum vs all test normals)

| model / scope | 150+ | 25-60 | 60-150 |
|---|---|---|---|
| patchcore/top | 0.655 (n=9) | 0.502 (n=136) | 0.551 (n=115) |
| patchcore/wrist_cam | 0.812 (n=447) | 0.505 (n=117) | 0.733 (n=73) |
| patchcore/pooled | 0.791 (n=456) | 0.534 (n=253) | 0.664 (n=188) |

## AUROC by camera (abnormal of that stratum vs all test normals)

| model / scope | top | wrist_cam |
|---|---|---|
| patchcore/top | 0.529 (n=260) | - |
| patchcore/wrist_cam | - | 0.746 (n=637) |
| patchcore/pooled | 0.533 (n=260) | 0.749 (n=637) |

## AUROC by stage (abnormal of that stage vs normals of that stage)

| model / scope | closeup | prescan | show | virtual_show |
|---|---|---|---|---|
| patchcore/top | 0.495 (n_abn=124, n_norm=363) | - | 0.678 (n_abn=16, n_norm=52) | 0.552 (n_abn=120, n_norm=425) |
| patchcore/wrist_cam | - | 0.820 (n_abn=247, n_norm=458) | 0.873 (n_abn=30, n_norm=60) | 0.685 (n_abn=360, n_norm=720) |
| patchcore/pooled | 0.498 (n_abn=124, n_norm=363) | 0.824 (n_abn=247, n_norm=458) | 0.811 (n_abn=46, n_norm=112) | 0.662 (n_abn=480, n_norm=1145) |

## Hidden defects (defective brick, < 25 defect pixels in view)

| model / scope | n | AUROC vs normal | frac above thr | normal frac above thr | median (normal median) |
|---|---|---|---|---|---|
| patchcore/top | 592 | 0.528 | 0.029 | 0.005 | 52.046 (51.148) |
| patchcore/wrist_cam | 679 | 0.504 | 0.015 | 0.008 | 41.600 (41.546) |
| patchcore/pooled | 1271 | 0.525 | 0.025 | 0.014 | 47.712 (46.621) |

Exported model: `patchcore/pooled` -> `models/best.pt` (see `score.py`).

