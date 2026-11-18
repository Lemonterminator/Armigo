# Agent overhaul and motion smoothness, 2026-11-18

Two pieces of work, measured on the same fixture suites:

- **Agent overhaul** (`agent-overhaul`, 4a4f357) against the code before it (245a774). This covers LLM fallback
  reasons, structured decoding, compact context, successive-halving policy search, bounded recovery variants,
  worker processes, the SQLite job store, the vLLM settings, and the MCP job tools.
- **Motion smoothness** (`motion-smoothness`, de065af) against `agent-overhaul`. This adds the smoothness metrics
  and fixes the jerk at the end of every episode.

All runs used the supervised LoRA detector (`sup-local.json`, the committed `sup.json` with an absolute
checkpoint path) and the rule-based skill choices unless the row says LLM. The LLM is Qwen3.8-27B-NVFP4 on vLLM
0.33 in WSL, port 8038. Compact summaries are under `summary/`. The full run folders stay local
(`results/agent-bench/before`, `final`, about 210 MB).

## Agent overhaul

| Run | Before | After |
|---|---|---|
| dev4, rule choices | 2/4 sorted, 4/4 labels | 4/4, 4/4 |
| dev4, LLM choices | 4/4, 4/4 | 4/4, 4/4 |
| bench16, rule choices | 12/16 sorted, 14/16 labels | 14/16, 14/16 |
| bench16, LLM choices | 12/16, 13/16 | 14/16, 14/16 |
| Prompt size, same decision context | 2,239 chars, 1,643 tokens | 470 chars, 198 tokens |
| Prompt tokens in bench16 LLM run | not recorded | median 371, max 734 |
| LLM decisions adopted | 19/19 | 22/22, 0 fallbacks |
| LLM decision latency, median | about 2.7 s | 0.94 s (1 worker), 1.23 s (4 workers) |
| bench16 throughput, rule | 6.08 episodes/min (1 worker) | 5.52 (1 worker), 16.39 (8 workers) |
| bench16 throughput, LLM | 5.56 episodes/min | 5.33 (1 worker), 12.15 (4 workers) |
| vLLM, 1 request at a time | 0.39 req/s, p50 2.76 s | 0.37 req/s, p50 2.92 s |
| vLLM, 4 concurrent | 0.38 req/s, p50 10.96 s | 1.16 req/s, p50 3.20 s |
| Policy search | LLM proposals: 12 episodes, 161 s | successive halving, budget 8: 352 episodes, 736 s, 8 workers |

Notes:

- The two extra sorts on bench16 and dev4 come from recovery. A torque stall in the second wrist scan used to be
  rewound to a checkpoint after the stall had started, which kept the stall in the health history and failed again.
  Recovery now offers the checkpoint before the stall, plus a bounded slower or higher variant.
- With one worker the new code is slightly slower per episode. The two recovered episodes now run to the end
  instead of failing early (26,375 against 24,419 simulated frames). The 8-worker and 1-worker bench16 runs gave
  identical per-episode outcomes.
- Single-request vLLM latency rose about 6% with `--max-num-seqs 4` and prefix caching. Four concurrent requests
  are three times faster.
- The search kept the default policy ("the default ranked first among the finalists"). That policy then sorted
  14/16 on bench16. The old search also kept the default.
- The files named `*_w1.json` were originally saved as `*_oldsrv`, but they ran against the new vLLM settings.

## Motion smoothness

Jerk is the third finite difference of arm joints 0-4 times 30³, in rad/s³. "Measured" uses the simulated joint
positions; "commanded" uses the expert's targets. A stop is a move that comes to rest (below 0.05 rad/s for three
frames) and then starts again.

There were two causes of the jerk at the end of every episode:

1. The rest pose has joints 1 and 3 at -1.757 and -1.777 rad, just outside their limits (-1.745 and -1.658).
   The expert planned a min-jerk return to the rest pose and then clipped each target, so those joints stopped
   dead partway through the move.
2. The 1.2 s return needed about 5.2 rad/s peak joint speed. Under the 2.94 N·m torque limit the wrist flex tops
   out near 4.6 rad/s, so it fell behind and then snapped onto the decelerating target.

The fix plans every move to the clipped target and sizes the return so its peak stays under 4.4 rad/s, capped at
one 1.4 s segment (longer moves are split with pauses). Four reference trajectories are identical frame for frame
up to the return; the return is 6 frames longer.

bench16, 8 workers (`summary/smoothness/`):

| | Before | After |
|---|---|---|
| Commanded jerk RMS, median | 22.95 | 5.55 |
| Measured jerk RMS, median | 11.01 | 6.40 |
| Worst measured peak | 504.7, in the return | 111.9, in the first frames of rest |
| Clipped target frames | 198 | 92 |
| Stops, median per episode | 36.5 | 36.5 |
| Sorted | 14/16 | 14/16, same per-episode outcomes |
| Simulated frames | 26,375 | 26,479 |

### Recorded episodes

Four bench16 fixtures were recorded on both versions (`config/agent_bench/video4.json`). Both sorted 4/4. The
peak column is the largest per-frame jerk over joints 0-4 from the recorded joint states.

| Case | Fixture | Result | Return peak before | Return peak after | Episode peak after |
|---|---|---|---|---|---|
| case-0000 | 600004, normal | green | 504.7 | 60.6 | 142.0, frame 3 (rest) |
| case-0001 | 600005, defect on face 4 | pink | 493.8 | 60.6 | 142.0, frame 3 (rest) |
| case-0002 | 600007, defect on face 0 | pink | 503.0 | 60.6 | 142.0, frame 3 (rest) |
| case-0003 | 600008, normal, one rewind | green | 407.9 | 60.5 | 86.8, approach |

Case-0003 compares branch 1, the branch after the rewind.

`videos/` has a jerk plot for each case. The videos are not in git (about 8 MB); they are in
`results/agent-bench/videos/` of the main checkout:

- `case-000N_full.mp4`: the whole episode, front camera, before on the left and after on the right, with the frame
  number. The two sides stay in step until the return.
- `case-000N_return_quarter_speed.mp4`: from 1 s before the return to the end, at quarter speed.

Rebuild them with:

```bash
python -m digital_twin_armigo.agent.twin_mcp run --detector results/agent-bench/sup-local.json --workers 4 --out <run> --suite config/agent_bench/video4.json
```

```bash
PYTHONPATH=src python scripts/compare_smoothness_videos.py <before-run> <after-run> results/agent-bench/videos
```

## Still open

- **Stop-and-go during approach and scanning.** The stop count did not change. The regular spikes in the first
  half of each jerk plot come from moves longer than 1.4 s being cut into chunks with 0.1 s pauses, and from holds.
  The health rule fails continuous torque saturation over 1.5 s, which is why the pauses exist. The next step is a
  jerk-limited planner that keeps the speed under the saturation limit without stopping.
- **The rest pose lies outside the joint limits.** It causes the remaining peak in the first frames and most of
  the clipped frames that are left. It is shared by the whole project, so it was left alone.
- The motion health check runs only after placing the brick.
- Two services can share one `--out` folder; nothing locks it. A second owner marks the first one's jobs
  interrupted.
- The committed `results/anomaly/models-sup/sup.json` uses a relative checkpoint path that breaks from the main
  checkout.
- Qwen's written failure analysis in the policy search is too vague to act on. The search itself does not rely on it.
