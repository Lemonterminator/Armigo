# Bowl Convex Decomposition Experiment Decision

**Result:** No candidate in R1–R4 passed all gates; no `src/` files were modified.

## Candidate Summary

- **R1_coacd_full**: FAIL at `G0`
  - `G0` (FAIL, 0.04s): `{"missing_area_frac": 0.0, "flat_floor_r_mm": 60.0, "flat_min_mm": 3.0001017772111895, "flat_max_mm": 5.193777856142889, "basin_min_mm": 3.0001017772111895, "rim_min_mm": 21.921760102525514, "rim_max_mm": 22.00000001488657, "wall_devs_mm": {"6": 7.375364637872323, "12": 5.501289369306278, "18": 4.2751673920750335}, "max_wall_dev_mm": 7.375364637872323, "wall_miss": 0, "protrusion_mm": 0.0025612763108284753}`
- **R2_wedge_full**: FAIL at `G1`
  - `G0` (PASS, 0.06s): `{"missing_area_frac": 0.0, "flat_floor_r_mm": 60.0, "flat_min_mm": 2.999999994085095, "flat_max_mm": 3.0000000791764596, "basin_min_mm": 2.9999999935924753, "rim_min_mm": 21.929257298159158, "rim_max_mm": 21.991981254261496, "wall_devs_mm": {"6": 0.1520431503768691, "12": 0.16902859791174762, "18": 0.1790423098084517}, "max_wall_dev_mm": 0.1790423098084517, "wall_miss": 0, "protrusion_mm": 0.0005700468908429457}`
  - `G1` (FAIL, 5.88s): `{"f552_worst_dist_mm": -3.3482093256501706, "f552_bad_normals": 0, "f552_ok": true, "worst_dist_mm": -7.091316800162605, "bad_normals": 417}`
- **R3_strips_floor_wedge_walls**: FAIL at `G1`
  - `G0` (PASS, 0.05s): `{"missing_area_frac": 0.0, "flat_floor_r_mm": 60.0, "flat_min_mm": 3.0000000000000027, "flat_max_mm": 3.0000000000000027, "basin_min_mm": 3.0000000000000027, "rim_min_mm": 21.929257298159158, "rim_max_mm": 21.991981254261496, "wall_devs_mm": {"6": 0.1520431503768691, "12": 0.16902859791174762, "18": 0.1790423098084517}, "max_wall_dev_mm": 0.1790423098084517, "wall_miss": 0, "protrusion_mm": 4.097333779496964}`
  - `G1` (FAIL, 4.66s): `{"f552_worst_dist_mm": -3.3333599614682217, "f552_bad_normals": 1, "f552_ok": false, "worst_dist_mm": -7.091316800162605, "bad_normals": 121}`
- **R4_strips_floor_box_walls**: FAIL at `G0`
  - `G0` (FAIL, 0.03s): `{"missing_area_frac": 0.0, "flat_floor_r_mm": 60.0, "flat_min_mm": 3.0000000000000027, "flat_max_mm": 3.0000000000000027, "basin_min_mm": 3.0000000000000027, "rim_min_mm": 3.0000000000000027, "rim_max_mm": 22.019388337463724, "wall_devs_mm": {"6": Infinity, "12": Infinity, "18": Infinity}, "max_wall_dev_mm": Infinity, "wall_miss": 3, "protrusion_mm": 6.043956171548857}`
- **R5_sdf** (follow-up): FAIL at `G0` (and diagnostic `G1` / `baseline_2x2` seed 1)
  - `G0` (FAIL, 0.28s): `{"missing_area_frac": 0.025905349794238683, "flat_min_mm": 0.634488793760199, "flat_max_mm": 3.0025918097766473, "basin_min_mm": 0.04059257090799351, "rim_min_mm": 20.757118532084576, "rim_max_mm": 21.872884726137563, "wall_devs_mm": {"6": Infinity, "12": 0.08759805361088979, "18": 0.06491255055803435}, "wall_miss": 72, "protrusion_mm": 0.0}`
  - Diagnostic `G1` (FAIL): `{"f552_worst_dist_mm": -0.9699171124796974, "f552_bad_normals": 4, "f552_ok": false, "worst_dist_mm": -2.146320941304034, "bad_normals": 396}`
  - Diagnostic `G2` (20/cond): PASS (`0` violations)
  - Diagnostic `baseline_2x2` seed 1: `FAILURE ['settle_timeout']`, `wall_s = 1.791 s` (2.43x baseline)
