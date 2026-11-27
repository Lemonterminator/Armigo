#!/usr/bin/env bash
# Paired trajectory throughput / GPU VRAM / detector & vLLM latency benchmark (scheduled separately on GPU).
# Plan: docs/superpowers/plans/2026-11-24-paired-trajectory-governance.md Task P6 Step 6.
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONPATH=src
PY=E:/digital_twin_armigo/.venv/Scripts/python
ROOT=E:/Armigo_data_v3/governance/capacity_runs
OUT_MD=E:/Armigo_data_v3/governance/capacity.md
BENCH_SUITE=$(mktemp -t p2_dev8_XXXXXX.json)
trap 'rm -f "$BENCH_SUITE"' EXIT

"$PY" -c "
import json
from pathlib import Path
src = Path('config/agent_bench/v2/p2_dev12.json')
data = json.loads(src.read_text(encoding='utf-8-sig'))
if isinstance(data, dict) and 'cases' in data:
    data['cases'] = data['cases'][:8]
elif isinstance(data, list):
    data = data[:8]
Path(r'$BENCH_SUITE').write_text(json.dumps(data, indent=2), encoding='utf-8')
"

mkdir -p "$ROOT" "$(dirname "$OUT_MD")"
cat > "$OUT_MD" << 'HEADER'
# Paired Run Capacity Benchmark

| batch | workers | wall_s | eps_per_hr | detector_elapsed_med_ms | detector_elapsed_p90_ms | broker_wait_med_ms | broker_wait_p90_ms | peak_vram_mib |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
HEADER

for w in 2 4 6 8; do
	batch="cap-w${w}"
	vram_log="$ROOT/${batch}_vram.csv"
	nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$vram_log" 2>/dev/null &
	nv_pid=$!
	t0=$("$PY" -c "import time; print(time.time())")
	"$PY" -m digital_twin_armigo.experiments.runner run-pair \
		--root "$ROOT" --batch "$batch" --engine mujoco \
		--suite "$BENCH_SUITE" --real-cameras fixed --strategy B --detector v7bot --workers "$w"
	t1=$("$PY" -c "import time; print(time.time())")
	kill "$nv_pid" 2>/dev/null || true
	wait "$nv_pid" 2>/dev/null || true

	"$PY" - "$ROOT" "$batch" "$w" "$t0" "$t1" "$vram_log" >> "$OUT_MD" << 'PYSTAT'
import json, statistics, sys
from pathlib import Path

root, batch, workers, t0, t1, vram_log = (
    Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), Path(sys.argv[6])
)
wall = max(t1 - t0, 1e-3)
eps = list((root / "runs" / batch).rglob("episode.json"))
det_ms, waits_ms = [], []
for ep in eps:
    ev_path = ep.parent / "events.jsonl"
    if not ev_path.is_file():
        continue
    started = {}
    for line in ev_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "detector_started":
            started[e.get("step")] = float(e.get("t_wall", 0.0))
        elif e.get("kind") == "view":
            if e.get("detector_elapsed_ms") is not None:
                det_ms.append(float(e["detector_elapsed_ms"]))
            if e.get("step") in started and e.get("t_wall") is not None:
                waits_ms.append(max(0.0, (float(e["t_wall"]) - started[e["step"]]) * 1000.0))

def pct90(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.9 * (len(s) - 1)))]

vram_vals = [int(x.strip()) for x in vram_log.read_text(errors="ignore").splitlines() if x.strip().isdigit()] if vram_log.is_file() else []
med_d = statistics.median(det_ms) if det_ms else 0.0
p90_d = pct90(det_ms)
med_w = statistics.median(waits_ms) if waits_ms else 0.0
p90_w = pct90(waits_ms)
ep_hr = len(eps) * 3600.0 / wall
peak_v = max(vram_vals) if vram_vals else 0
print(f"| {batch} | {workers} | {wall:.1f} | {ep_hr:.1f} | {med_d:.1f} | {p90_d:.1f} | {med_w:.1f} | {p90_w:.1f} | {peak_v} |")
PYSTAT
done

# Strategy C with Qwen (workers=4)
"$PY" -m digital_twin_armigo.experiments.runner run-pair \
	--root "$ROOT" --batch "cap-c-qwen-w4" --engine mujoco \
	--suite "$BENCH_SUITE" --real-cameras fixed --strategy C --detector v7bot --workers 4

"$PY" - "$ROOT" "cap-c-qwen-w4" >> "$OUT_MD" << 'PYLLM'
import json, statistics, sys
from pathlib import Path

root, batch = Path(sys.argv[1]), sys.argv[2]
lats = []
for ev in (root / "runs" / batch).rglob("events.jsonl"):
    for line in ev.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") in ("llm", "vlm", "llm_call") and e.get("elapsed_ms") is not None:
            lats.append(float(e["elapsed_ms"]))
s = sorted(lats)
med = statistics.median(s) if s else 0.0
p90 = s[min(len(s) - 1, int(0.9 * (len(s) - 1)))] if s else 0.0
print(f"\n## Strategy C + Qwen (workers=4)\n\n- llm_requests: {len(s)}\n- median_ms: {med:.1f}\n- p90_ms: {p90:.1f}\n")
PYLLM
