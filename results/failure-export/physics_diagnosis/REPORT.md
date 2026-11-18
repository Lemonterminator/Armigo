# Physics diagnosis of the three failing episodes

MuJoCo 4.0.1, simulation only, nominal assumed physics parameters. Nothing under `src/`, `scripts/` or `assets/` was
changed; every variant below is applied in memory to the compiled model by `exp.py` / `idx32_open.py`.
The scripts add the repository `src` and `scripts` folders to `sys.path` from `E:\digital_twin_armigo`.

## 1. `high_mass_seed29` (bounce_out, 2.66 m/s): collision detection returns a 115 mm penetration

Substep log: `log_high_mass_seed29_substeps.txt` (240 Hz, frames 549-553).

- The jaws open, the brick drops 5 mm onto the bowl floor, tilts and sinks 3.2-3.6 mm into it. No jaw touches it any more.
- In one substep the contacts between the brick box and the bowl floor cylinder (`pink_bowl/g118`) report
  `dist = -115.31 mm` with a horizontal normal `(-0.69, -0.73, 0.01)`. The brick is 32 mm wide.
- Two contacts of about 5 N each then give the brick 2.7 m/s in that one substep (5 N x 4.17 ms / 11.2 g, twice).
- MuJoCo reports no warnings (`mjData.warning` all zero); the solver converged in at most 8 iterations.
  This is not a solver divergence.
- 115 mm = the floor radius (100 mm) + the brick half width (15 mm): the box was pushed out through the rim.

Isolated reproduction, `ccd_iso.py` / `log_ccd_isolated_repro.txt`: one static floor geom, one free box at the exact pose
from the log, `mj_forward` only.

| floor geom | contact |
|---|---|
| cylinder R=100 mm, 3 mm thick (the scene) | `-115.31 mm`, horizontal normal |
| the same cylinder, 40 mm thick, same top face | `-115.32 mm`, same |
| cylinder R=30 mm | `-45.3 mm` (30 + 15), same |
| box 200x200x3 mm | `-3.35 mm`, vertical normal (correct) |
| cylinder with `nativeccd` disabled | `-3.35 mm`, vertical normal (correct) |
| cylinder with `ccd_iterations=500`, `ccd_tolerance=1e-10` | `-115.31 mm` (no help) |

So it is the native box-vs-cylinder collision of 4.0.1, and it is independent of the cylinder thickness. Thickening the
cylinder in the full episode changes nothing (release peak 2.6622 m/s either way).

## 2. `low_friction_seed111` (bounce_out, 5.00 m/s): a resting brick spins up and hops with no external force

Substep log: `log_low_friction_seed111_substeps.txt`.

- Gripper fully open (0.849 rad), jaw force 0, the only contacts are four brick-corner contacts with the same bowl floor
  cylinder, penetration 0.07-0.6 mm, normal force 0.02-0.4 N.
- From frame 602 the brick yaw angular velocity goes 0.1 -> 5 -> 16 rad/s, it hops 35 mm, and in one substep
  (frame 604) goes 2.5 -> 110 rad/s; at frame 609 it reaches 5 m/s and 669 rad/s.
- The contact wrench at the 16 rad/s step has a torsional torque of 1.0e-3 N m per contact, which equals the cone limit
  (torsional friction 0.02 m x 0.0535 N). The contacts are condim 4 (bowl floor torsional friction on).
- No NaN, no warning, solver iterations 1-9. Energy appears from nothing: this is a contact-model instability on the bowl
  floor cylinder, not an agent action.
- I did not isolate the exact mechanism in a minimal model (only the box-vs-cylinder depth in section 1 was).

## 3. What fixes them (experiment, not applied)

Single episodes (`exp.py`, `release_peak` in m/s):

| variant | high_mass 29 | low_friction 111 |
|---|---|---|
| none (as recorded) | FAIL bounce_out 2.66 | FAIL bounce_out 5.00 |
| bowl floor 40 mm thick | FAIL 2.66 | SUCCESS 0.11 |
| bowl floor condim 3 (no torsion) | FAIL 2.15 | SUCCESS 0.11 |
| native CCD disabled | FAIL settle_timeout 0.19 | FAIL settle_timeout 0.26 |
| bowl floor as a box instead of a cylinder | SUCCESS 0.045 | SUCCESS 0.11 |

Box floor over the full seed sets of these two conditions (`log_boxfloor_*`): high_mass seeds 0-29: 30/30 (recorded 29/30),
max release peak 0.080 m/s; low_friction seeds 100-129: 30/30 (recorded 29/30), max release peak 0.300 m/s.
Not run: the other five matrix conditions, the held-out conditions, the layout sets. The box in this experiment is a
200x200 mm slab, so its corners stick out past the circular bowl; a real change needs a floor shaped to the bowl.
With native CCD disabled the brick no longer bounces but keeps spinning on the cylinder and ends in `settle_timeout`,
the same bowl-floor cylinder problem seen on 3.12.0.

## 4. `layouts_seed1000_idx32` (pick_miss): the moving jaw is blocked by the green bowl rim

Log: `log_idx32_jaw_env_contacts.txt`.

- When the closing ramp starts (frame ~101) the moving jaw is already in contact with `green_bowl/g76` (a rim wall box):
  10.4 N at first, then a steady 6.7 N. The measured gripper angle stalls at 0.76 rad while the command goes to 0.06 rad.
- The brick sees 0 N the whole time. The fixed jaw is fine.
- Both retries re-open to 0.85 rad and re-close on the same spot (frames 156-167, 266-...), so the retries repeat the failure.
- The chosen grasp had `clearance_m = -4.2 mm` to the bowl: the expert picked the least bad of candidates that were all blocked.

Narrower pre-grasp opening on this layout (`idx32_open.py`, `log_idx32_open_g_sweep.txt`; `open_g` is the gripper command with the
jaws open, default 0.85 rad):

| open_g | verdict | jaw-env max | clearance | frames |
|---|---|---|---|---|
| 0.85 | FAIL pick_miss (no_clamp, 2 retries) | 23.1 N | -4.2 mm | 540 |
| 0.70 | FAIL pick_miss | 55.5 N | -6.2 mm | 621 |
| 0.55 | FAIL pick_miss, torque_stall, tracking_lag | 48.5 N | -5.1 mm | 630 |
| 0.40 | SUCCESS, no retries | 0.0 N | +2.1 mm | 818 |
| 0.30 | FAIL place_miss (hit the 900-frame cap) | 0.0 N | +7.7 mm | 900 |

One layout only. A fixed 0.40 would change every other episode; the opening should follow from the grasp candidate
(brick width plus margin, checked against the bowl rim) and be regression-tested on the 120 layouts and the matrix.
