# Brick surface anomaly detection on simulated crops (v1)

Simulation only. Crops of the held brick at the expert's first show hold and at random virtual show poses (MuJoCo, `scripts/anomaly/render_dataset.py`), detectors from anomalib trained on normal crops only (`scripts/anomaly/train_eval.py`). Full numbers: `report-v1.json`.

Device: cuda, torch 2.11.0+cu128, anomalib 2.7.0; input 320 px; threshold = 0.99 quantile of validation normal scores.

Data (crops): fit 3976 / val 715 normal (116 / 20 layouts); test 4246 (124 layouts).

Test crops by label: test/abnormal=960, test/hidden_defect=1391, test/normal=2228, train/normal=5056

## Overall (test: normal vs abnormal)

| model / scope | n normal | n abnormal | AUROC | F1 @thr | precision | recall | FPR | best F1 (oracle thr) |
|---|---|---|---|---|---|---|---|---|
| patchcore/top | 840 | 260 | 0.663 | 0.203 | 0.674 | 0.119 | 0.018 | 0.431 |
| patchcore/wrist_cam | 1238 | 637 | 0.859 | 0.663 | 0.942 | 0.512 | 0.016 | 0.758 |
| patchcore/pooled | 2078 | 897 | 0.802 | 0.475 | 0.923 | 0.320 | 0.011 | 0.647 |

## AUROC by defect kind (abnormal of that stratum vs all test normals)

| model / scope | cross_var |
|---|---|
| patchcore/top | 0.663 (n=260) |
| patchcore/wrist_cam | 0.859 (n=637) |
| patchcore/pooled | 0.802 (n=897) |

## AUROC by defect face (0 +x, 1 -x, 2 +y, 3 -y, 4 top, 5 bottom) (abnormal of that stratum vs all test normals)

| model / scope | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| patchcore/top | 0.740 (n=67) | 0.646 (n=78) | 0.661 (n=50) | 0.712 (n=36) | 0.472 (n=29) |
| patchcore/wrist_cam | 0.876 (n=67) | 0.792 (n=165) | 0.761 (n=105) | 0.593 (n=35) | 0.971 (n=265) |
| patchcore/pooled | 0.822 (n=134) | 0.743 (n=243) | 0.708 (n=155) | 0.661 (n=71) | 0.926 (n=294) |

## AUROC by visible defect pixels (abnormal of that stratum vs all test normals)

| model / scope | 150+ | 25-60 | 60-150 |
|---|---|---|---|
| patchcore/top | 0.855 (n=9) | 0.527 (n=136) | 0.808 (n=115) |
| patchcore/wrist_cam | 0.973 (n=447) | 0.505 (n=117) | 0.733 (n=73) |
| patchcore/pooled | 0.957 (n=456) | 0.544 (n=253) | 0.776 (n=188) |

## AUROC by camera (abnormal of that stratum vs all test normals)

| model / scope | top | wrist_cam |
|---|---|---|
| patchcore/top | 0.663 (n=260) | - |
| patchcore/wrist_cam | - | 0.859 (n=637) |
| patchcore/pooled | 0.642 (n=260) | 0.861 (n=637) |

## AUROC by stage (abnormal of that stage vs normals of that stage)

| model / scope | closeup | prescan | show | virtual_show |
|---|---|---|---|---|
| patchcore/top | 0.626 (n_abn=124, n_norm=363) | - | 0.810 (n_abn=16, n_norm=52) | 0.679 (n_abn=120, n_norm=425) |
| patchcore/wrist_cam | - | 0.780 (n_abn=247, n_norm=458) | 0.873 (n_abn=30, n_norm=60) | 0.884 (n_abn=360, n_norm=720) |
| patchcore/pooled | 0.617 (n_abn=124, n_norm=363) | 0.782 (n_abn=247, n_norm=458) | 0.840 (n_abn=46, n_norm=112) | 0.838 (n_abn=480, n_norm=1145) |

## Hidden defects (defective brick, < 25 defect pixels in view)

| model / scope | n | AUROC vs normal | frac above thr | normal frac above thr | median (normal median) |
|---|---|---|---|---|---|
| patchcore/top | 592 | 0.498 | 0.010 | 0.018 | 58.277 (58.178) |
| patchcore/wrist_cam | 679 | 0.518 | 0.012 | 0.016 | 53.805 (53.442) |
| patchcore/pooled | 1271 | 0.528 | 0.015 | 0.011 | 57.615 (57.178) |

Exported model: `patchcore/pooled` -> `models/best.pt` (see `score.py`).

