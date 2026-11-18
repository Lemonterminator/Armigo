# Brick surface anomaly detection on simulated crops (v1)

Simulation only. Crops of the held brick at the expert's first show hold and at random virtual show poses (MuJoCo, `scripts/anomaly/render_dataset.py`), detectors from anomalib trained on normal crops only (`scripts/anomaly/train_eval.py`). Full numbers: `report-v1.json`.

Device: cuda, torch 2.11.0+cu128, anomalib 2.7.0; input 224 px; threshold = 0.99 quantile of validation normal scores.

Data (crops): fit 3177 / val 551 normal (116 / 20 layouts); test 3388 (124 layouts).

Test crops by label: test/abnormal=729, test/hidden_defect=1011, test/normal=1648, train/normal=3728

## Overall (test: normal vs abnormal)

| model / scope | n normal | n abnormal | AUROC | F1 @thr | precision | recall | FPR | best F1 (oracle thr) |
|---|---|---|---|---|---|---|---|---|
| patchcore/top | 770 | 272 | 0.696 | 0.171 | 0.812 | 0.096 | 0.008 | 0.501 |
| patchcore/wrist_cam | 878 | 457 | 0.815 | 0.584 | 0.933 | 0.424 | 0.016 | 0.678 |
| patchcore/pooled | 1648 | 729 | 0.770 | 0.414 | 0.920 | 0.268 | 0.010 | 0.602 |

## AUROC by defect kind (abnormal of that stratum vs all test normals)

| model / scope | cross_var |
|---|---|
| patchcore/top | 0.696 (n=272) |
| patchcore/wrist_cam | 0.815 (n=457) |
| patchcore/pooled | 0.770 (n=729) |

## AUROC by defect face (0 +x, 1 -x, 2 +y, 3 -y, 4 top, 5 bottom) (abnormal of that stratum vs all test normals)

| model / scope | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| patchcore/top | 0.764 (n=71) | 0.698 (n=94) | 0.650 (n=47) | 0.720 (n=29) | 0.579 (n=31) |
| patchcore/wrist_cam | 0.739 (n=55) | 0.690 (n=111) | 0.729 (n=75) | 0.646 (n=29) | 0.972 (n=187) |
| patchcore/pooled | 0.755 (n=126) | 0.689 (n=205) | 0.683 (n=122) | 0.674 (n=58) | 0.929 (n=218) |

## AUROC by visible defect pixels (abnormal of that stratum vs all test normals)

| model / scope | 150+ | 25-60 | 60-150 |
|---|---|---|---|
| patchcore/top | 0.826 (n=20) | 0.610 (n=113) | 0.747 (n=139) |
| patchcore/wrist_cam | 0.933 (n=309) | 0.481 (n=81) | 0.675 (n=67) |
| patchcore/pooled | 0.907 (n=329) | 0.580 (n=194) | 0.730 (n=206) |

## AUROC by camera (abnormal of that stratum vs all test normals)

| model / scope | top | wrist_cam |
|---|---|---|
| patchcore/top | 0.696 (n=272) | - |
| patchcore/wrist_cam | - | 0.815 (n=457) |
| patchcore/pooled | 0.701 (n=272) | 0.816 (n=457) |

## AUROC by stage (abnormal of that stage vs normals of that stage)

| model / scope | closeup | prescan | show | virtual_show |
|---|---|---|---|---|
| patchcore/top | 0.673 (n_abn=180, n_norm=480) | - | 0.758 (n_abn=18, n_norm=58) | 0.739 (n_abn=74, n_norm=232) |
| patchcore/wrist_cam | - | 0.720 (n_abn=247, n_norm=458) | 0.880 (n_abn=30, n_norm=60) | 0.882 (n_abn=180, n_norm=360) |
| patchcore/pooled | 0.664 (n_abn=180, n_norm=480) | 0.726 (n_abn=247, n_norm=458) | 0.840 (n_abn=48, n_norm=118) | 0.848 (n_abn=254, n_norm=592) |

## Hidden defects (defective brick, < 25 defect pixels in view)

| model / scope | n | AUROC vs normal | frac above thr | normal frac above thr | median (normal median) |
|---|---|---|---|---|---|
| patchcore/top | 536 | 0.529 | 0.007 | 0.008 | 58.503 (57.866) |
| patchcore/wrist_cam | 475 | 0.514 | 0.032 | 0.016 | 53.053 (52.784) |
| patchcore/pooled | 1011 | 0.537 | 0.016 | 0.010 | 57.758 (57.102) |

Exported model: `patchcore/pooled` -> `models/best.pt` (see `score.py`).

