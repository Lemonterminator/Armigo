# Brick surface anomaly detection on simulated crops (v1)

Simulation only. Crops of the held brick at the expert's first show hold and at random virtual show poses (MuJoCo, `scripts/anomaly/render_dataset.py`), detectors from anomalib trained on normal crops only (`scripts/anomaly/train_eval.py`). Full numbers: `report-v1.json`.

Device: cuda, torch 2.11.0+cu128, anomalib 2.7.0; input 320 px; threshold = 0.99 quantile of validation normal scores.

Data (crops): fit 3976 / val 715 normal (116 / 20 layouts); test 4246 (124 layouts).

Test crops by label: test/abnormal=960, test/hidden_defect=1391, test/normal=2228, train/normal=5056

## Overall (test: normal vs abnormal)

| model / scope | n normal | n abnormal | AUROC | F1 @thr | precision | recall | FPR | best F1 (oracle thr) |
|---|---|---|---|---|---|---|---|---|
| patchcore/top | 840 | 260 | 0.652 | 0.206 | 0.756 | 0.119 | 0.012 | 0.427 |
| patchcore/wrist_cam | 1238 | 637 | 0.860 | 0.660 | 0.947 | 0.507 | 0.015 | 0.762 |
| patchcore/pooled | 2078 | 897 | 0.798 | 0.482 | 0.918 | 0.327 | 0.013 | 0.635 |

## AUROC by defect kind (abnormal of that stratum vs all test normals)

| model / scope | cross_var |
|---|---|
| patchcore/top | 0.652 (n=260) |
| patchcore/wrist_cam | 0.860 (n=637) |
| patchcore/pooled | 0.798 (n=897) |

## AUROC by defect face (0 +x, 1 -x, 2 +y, 3 -y, 4 top, 5 bottom) (abnormal of that stratum vs all test normals)

| model / scope | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| patchcore/top | 0.739 (n=67) | 0.633 (n=78) | 0.628 (n=50) | 0.687 (n=36) | 0.499 (n=29) |
| patchcore/wrist_cam | 0.879 (n=67) | 0.792 (n=165) | 0.759 (n=105) | 0.597 (n=35) | 0.972 (n=265) |
| patchcore/pooled | 0.811 (n=134) | 0.733 (n=243) | 0.707 (n=155) | 0.647 (n=71) | 0.930 (n=294) |

## AUROC by visible defect pixels (abnormal of that stratum vs all test normals)

| model / scope | 150+ | 25-60 | 60-150 |
|---|---|---|---|
| patchcore/top | 0.825 (n=9) | 0.514 (n=136) | 0.802 (n=115) |
| patchcore/wrist_cam | 0.972 (n=447) | 0.506 (n=117) | 0.741 (n=73) |
| patchcore/pooled | 0.950 (n=456) | 0.542 (n=253) | 0.775 (n=188) |

## AUROC by camera (abnormal of that stratum vs all test normals)

| model / scope | top | wrist_cam |
|---|---|---|
| patchcore/top | 0.652 (n=260) | - |
| patchcore/wrist_cam | - | 0.860 (n=637) |
| patchcore/pooled | 0.639 (n=260) | 0.858 (n=637) |

## AUROC by stage (abnormal of that stage vs normals of that stage)

| model / scope | closeup | prescan | show | virtual_show |
|---|---|---|---|---|
| patchcore/top | 0.616 (n_abn=124, n_norm=363) | - | 0.834 (n_abn=16, n_norm=52) | 0.661 (n_abn=120, n_norm=425) |
| patchcore/wrist_cam | - | 0.786 (n_abn=247, n_norm=458) | 0.876 (n_abn=30, n_norm=60) | 0.882 (n_abn=360, n_norm=720) |
| patchcore/pooled | 0.616 (n_abn=124, n_norm=363) | 0.770 (n_abn=247, n_norm=458) | 0.861 (n_abn=46, n_norm=112) | 0.836 (n_abn=480, n_norm=1145) |

## Hidden defects (defective brick, < 25 defect pixels in view)

| model / scope | n | AUROC vs normal | frac above thr | normal frac above thr | median (normal median) |
|---|---|---|---|---|---|
| patchcore/top | 592 | 0.507 | 0.008 | 0.012 | 58.306 (58.280) |
| patchcore/wrist_cam | 679 | 0.524 | 0.013 | 0.015 | 53.889 (53.473) |
| patchcore/pooled | 1271 | 0.527 | 0.016 | 0.013 | 57.853 (57.169) |

Exported model: `patchcore/pooled` -> `models/best.pt` (see `score.py`).

