from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from collections import Counter, defaultdict
from pathlib import Path

from digital_twin_armigo.experiments import report, runner
from digital_twin_armigo.experiments.contract import case_id
from digital_twin_armigo.experiments.counterfactual import VARIANTS, verdict
from digital_twin_armigo.experiments.stats import paired_comparison
from make_suites import register_suite

ROOT = Path("D:/Armigo_data_v2")
ORIG = ROOT / "reports/drift-dev-20261126-7be6abd"
TAG = "loop-readiness-20261126-43c9039"
OUT = ROOT / "reports" / TAG
CF = f"{TAG}-cf"
PROBE = f"{TAG}-probe"
BASE = f"{TAG}-off"
NEW = f"{TAG}-on"
GROUP_OFF = "realistic-drift-B-v7bot-loc-vote"
GROUP_ON = GROUP_OFF + "-alarm"
SUITES = OUT / "suite-source"
DETECTOR = "E:/digital_twin_armigo/results/anomaly/models-sup/sup-v7bot.json"
REPO = Path("E:/scratch/armigo-loop-readiness-worker")
SEED = 0
STATE_NAME = "state.json"
WORKERS = int(os.environ.get("ARMIGO_LOOP_WORKERS", "8"))
_NATIVE_RUN_JOB = runner.run_job
_NATIVE_SERVICE = runner._service
_SCORERS = {}

def shared_service(config):
    service = _NATIVE_SERVICE(config)
    if service.scorer is None:
        service.scorer = _SCORERS.get(config["detector"])
    return service

def run_worker_job(config, job):
    runner._service = shared_service
    result = _NATIVE_RUN_JOB(config, job)
    service = _NATIVE_SERVICE(config)
    scorer = service.scorer
    if scorer is not None:
        if result["state"] == "done":
            _SCORERS[config["detector"]] = scorer
        elif result["state"] == "infra_failed":
            _SCORERS.pop(config["detector"], None)
            for cached in runner._SERVICES.values():
                if cached.scorer is scorer:
                    cached.scorer = None
            scorer.close()
    return result

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    runner._write_json(path, data)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

def suite(name, rows):
    path = SUITES / (name + ".json")
    write_json(path, rows)
    register_suite(SUITES / "manifest.json", name, path, rows)
    return path

def jobs(batch, suite_name, camera="drift", *, alarm=True, localise=None, oracle=()):
    args = [
        "run", "--root", str(ROOT), "--batch", batch, "--suite", suite_name, "--engine", "mujoco",
        "--sensing", "realistic", "--camera", camera, "--strategy", "B", "--detector-spec", DETECTOR,
    ]
    if not alarm:
        args.append("--no-drift-alarm")
    if localise is not None:
        args.extend(("--localise", localise))
    if oracle:
        args.extend(("--oracle", ",".join(oracle)))
    result = []
    original = runner.cmd_run
    try:
        runner.cmd_run = lambda a: result.append(runner.plan_group(a)) or 0
        runner.main(args)
    finally:
        runner.cmd_run = original
    cfg, planned, _ = result[0]
    return [(cfg, j) for j in planned]

def execute(stage, todo, workers=WORKERS):
    write_json(OUT / STATE_NAME, {"state": stage, "pending_at_start": len(todo), "workers": workers})
    print(stage, len(todo), flush=True)
    runner.run_job = run_worker_job
    for retry in range(2):
        code = runner._execute_jobs(todo, ROOT, "mujoco", workers, 12.0, 28.0)
        if code == 0:
            return
        if code != 2 or retry:
            break
        retries = []
        for cfg, j in todo:
            path = Path(j["path"])
            status_path = path / "run_status.json"
            if status_path.exists() and read_json(status_path)["state"] == "done":
                continue
            if not status_path.exists():
                retries.append((cfg, j))
                continue
            exp = j["experiment"]
            attempt = exp["attempt"] + 1
            if attempt <= runner.MAX_ATTEMPTS:
                rp = runner._attempt_dir(path.parent, exp["case_id"], attempt)
                rex = {**exp, "attempt": attempt, "parent_run": exp["run_id"],
                       "run_id": __import__("uuid").uuid4().hex}
                retries.append((cfg, {**j, "path": str(rp), "experiment": rex}))
        if not retries:
            break
        todo = retries
    raise RuntimeError(f"{stage}: runner exit {code}; resume the coordinator after diagnosing the error")


def report_path(batch, compare=None):
    return report.write_report(ROOT, batch, compare=compare)

def full_rows(batch):
    return report.scan_batch(ROOT / "runs" / batch)[0]

def checked(rows, ids, title):
    actual = {r["case_id"] for r in rows if r["has_episode"] and not r["infra_failed"]}
    if len(rows) != len(ids) or actual != set(ids):
        raise RuntimeError(f"{title}: incomplete, {len(actual)}/{len(ids)}; missing={sorted(set(ids) - actual)[:10]}")
    return {r["case_id"]: r for r in rows}

def status_for_config(rows, batch, ids, group):
    return checked([r for r in rows if r["group"] == group], ids, batch + "/" + group)

def original_bank():
    selected = read_json(ORIG / "round-01/ledger.json")["clusters"][0]
    assert selected["key"] == ["grasp", "no_clamp", "calibration_stale"] and selected["n"] == 8
    scores = {Path(r["path"]).as_posix().split("/runs/")[1]: r for r in read_jsonl(ORIG / "scores.jsonl")}
    rows = []
    for tid in selected["examples"]:
        r = {**scores[tid], "traj_id": tid}
        r["case"] = read_json(Path(r["path"], "task.json"))["case"]
        r["localise"] = "vote" if "-loc-vote" in r["group"] else "top"
        rows.append(r)
    assert len({r["case_id"] for r in rows}) == len(rows) == 8
    return rows

def counterfactual_plan(selected):
    planned = []
    for mode in ("top", "vote"):
        name = f"cluster1_{mode}"
        suite(name, [r["case"] for r in selected if r["localise"] == mode])
        for variant in VARIANTS:
            planned.extend(jobs(CF, name, localise=mode, oracle=variant, alarm=False))
    return planned

def publish_cf(selected):
    path = report_path(CF)
    rows = read_jsonl(path / "scores.jsonl")
    indexed = {(r["group"], r["case_id"]): r for r in rows}
    output = []
    dist = Counter()
    for base in selected:
        vrows = {
            v: indexed[(runner.group_name("realistic", "drift", "B", "v7bot", oracle=v,
                                           localise=base["localise"]), base["case_id"])]
            for v in VARIANTS
        }
        outcomes = {v: r["physical_success"] for v, r in vrows.items()
                    if r["has_episode"] and not r["infra_failed"]}
        if len(outcomes) != len(VARIANTS):
            raise RuntimeError(f"incomplete counterfactual for {base['traj_id']}")
        result = verdict(base["physical_success"], outcomes)
        if base["physical_success"] and outcomes[()]:
            result = {"verdict": "already_pass", "minimal": []}
        display = " / ".join("+".join(v) for v in result["minimal"]) or "-"
        dist[result["verdict"], display] += 1
        output.append({
            "traj_id": base["traj_id"], "case_id": base["case_id"], "group": base["group"],
            "path": base["path"], "symptom": base["root_cause"], "mechanism": base["mechanism"],
            "upstream": base["upstream"], "original_success": base["physical_success"],
            "variants": {"repeat" if not v else "+".join(v): {
                "physical_success": r["physical_success"], "label_correct": r["label_correct"], "path": r["path"],
            } for v, r in vrows.items()}, **result,
        })
    write_jsonl(path / "counterfactual.jsonl", output)
    lines = [
        "# 第 1 簇反事实", "",
        "旧台账的 8 条记录均以原定位模式、原案例和同一检测器重跑。表中成功按 replay 的物理分拣评分。",
        "原本物理成功且复跑通过的案例标为 already_pass。", "",
        "| 案例 | 定位 | 原始 | repeat | camera | brick | bowls | 全替换 | 判定 | 最小集合 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    fmt = lambda ok: "✓" if ok else "×"
    for r, base in zip(output, selected, strict=True):
        vals = [fmt(d["physical_success"]) for d in r["variants"].values()]
        minimal = " / ".join("+".join(v) for v in r["minimal"]) or "–"
        lines.append(f"| {r['case_id']} | {base['localise']} | {fmt(r['original_success'])} | "
                     + " | ".join(vals) + f" | {r['verdict']} | {minimal} |")
    lines += ["", "## 最小集合分布", ""]
    lines += [f"- {kind}，{m}：{n} 例" for (kind, m), n in sorted(dist.items())]
    confirmed = [r for r in output if r["verdict"] == "confirmed"]
    if any(len(r["minimal"]) > 1 for r in confirmed):
        lines += ["", "有多个单项能救回的案例，单项替换无法区分哪一项是主因。可在开发集逐项减小校正量，"
                  "对相机、积木和碗分别测达到恢复的最小幅度，并在首个抓取窗口重复测量。"]
    (path / "counterfactual.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("counterfactual:", dict(dist), flush=True)

def probe_report():
    path = report_path(PROBE)
    subprocess.run([sys.executable, str(REPO / "scripts/experiments/failure_ledger.py"),
                    "--root", str(ROOT), "--batch", PROBE, "--round", "probe",
                    "--out-dir", str(path / "ledger")], check=True)

def stratified_winners(results, n):
    strata = defaultdict(list)
    for source, rows in results:
        for r in rows:
            if r["physical_success"]:
                _, pos, face = r["case"][:3]
                strata[source, pos, face].append(r)
    for items in strata.values():
        items.sort(key=lambda r: r["case"][0])
    answer = []
    for rank in range(max(map(len, strata.values()), default=0)):
        for key in sorted(strata):
            if rank < len(strata[key]):
                answer.append(strata[key][rank]["case"])
                if len(answer) == n:
                    return answer
    return answer

def prep_regression(selected):
    seeds_in_bank = {r["case"][0] for r in selected}
    sources = {
        name: [c for c in read_json(REPO / "config/agent_bench/v2" / f"{name}.json") if c[0] not in seeds_in_bank]
        for name in ("bench16", "dev_suite_700001", "loop_01_96")
    }
    def controls(sources, batch_prefix):
        todo = []
        for name, cases in sources.items():
            sn = "reg_pool_" + name
            suite(sn, cases)
            todo += jobs(batch_prefix + "-" + name, sn, alarm=False)
        execute("regression_controls", todo)
        results = []
        for name, cases in sources.items():
            bp = batch_prefix + "-" + name
            by_id = status_for_config(full_rows(bp), bp, [case_id(*c[:3]) for c in cases], GROUP_OFF)
            results.append((name, [{**by_id[case_id(*c[:3])], "case": c} for c in cases]))
        return results
    results = controls(sources, BASE)
    winners = stratified_winners(results, 100)
    if len(winners) < 100:
        extra = {"loop_01_reserve": [
            [seed, label, (seed - 3600001) % 6] for seed in range(3600049, 3600149) for label in (False, True)
        ]}
        results.extend(controls(extra, BASE))
        winners = stratified_winners(results, 100)
    if len(winners) != 100:
        raise RuntimeError(f"regression control winners: {len(winners)} of 100")
    path = suite("drift_regression_selected100", winners)
    selected_ids = {case_id(*c[:3]) for c in winners}
    by_id = {r["case_id"]: r for _, rows in results for r in rows if r["case_id"] in selected_ids}
    if len(by_id) != 100 or not all(r["physical_success"] for r in by_id.values()):
        raise RuntimeError("invalid control selection")
    return path, list(by_id.values())

def verify(bank_path, regression_path, reg_control):
    from verify_fix import main
    off = ROOT / "reports" / BASE
    off_subset = OUT / "verify-inputs" / "reports" / BASE
    off_subset.mkdir(parents=True, exist_ok=True)
    write_jsonl(off_subset / "scores.jsonl", [*read_jsonl(off / "scores.jsonl"), *reg_control])
    fresh = REPO / "config/agent_bench/v2/loop_02_400.json"
    sn = ["bank", "regression", "fresh"]
    pp = [bank_path, regression_path, fresh]
    criteria = OUT / "criteria.json"
    opts = [
        "--base", BASE, "--batch", NEW, "--root", str(ROOT), "--base-root", str(OUT / "verify-inputs"),
        "--group", GROUP_ON, "--base-group", GROUP_OFF, "--out-dir", str(OUT),
        "--criteria", str(criteria), "--engine", "mujoco", "--cluster", "drift-alarm", "--skip-run",
    ]
    for name, path in zip(sn, pp, strict=True):
        opts += ["--" + name, str(path)]
    assert main(opts) == 0
    contents = read_json(OUT / "verify-drift-alarm.json")
    costs = []
    by_role = {}
    b_rows = read_jsonl(off_subset / "scores.jsonl")
    n_rows = read_jsonl(ROOT / "reports" / NEW / "scores.jsonl")
    for role, path in zip(sn, pp, strict=True):
        ids = {case_id(*c[:3]) for c in read_json(path)}
        a = [r for r in b_rows if r["case_id"] in ids]
        b = [r for r in n_rows if r["case_id"] in ids]
        a_map, b_map = checked(a, ids, role + " control"), checked(b, ids, role + " alarm")
        by_role[role] = paired_comparison(a, b, B=2000, seed=SEED)
        for cid in sorted(ids):
            costs.append({"gate": role, "case_id": cid,
                          "frames_delta": b_map[cid]["sim_frames"] - a_map[cid]["sim_frames"],
                          "wall_s_delta": round(b_map[cid]["wall_s"] - a_map[cid]["wall_s"], 3)})
    write_jsonl(OUT / "paired-costs.jsonl", costs)
    write_json(OUT / "paired-gates.json", by_role)
    from statistics import mean, median
    lines = ["# 三道关配对差异及成本", "", "| 关 | 对数 | 候选 − 对照 | 95% CI | n10 | n01 | 帧差均值/中位 | 秒差均值/中位 |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for role, pc in by_role.items():
        subset = [r for r in costs if r["gate"] == role]
        f = [r["frames_delta"] for r in subset]
        t = [r["wall_s_delta"] for r in subset]
        lines.append(f"| {role} | {pc['n_pairs']} | {pc['diff']:+.1%} | "
                     f"[{pc['ci_lo']:+.1%}, {pc['ci_hi']:+.1%}] | {pc['n10']} | {pc['n01']} | "
                     f"{mean(f):+.1f}/{median(f):+.1f} | {mean(t):+.1f}/{median(t):+.1f} |")
    (OUT / "paired-gates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return contents["verdict"]

def publish_ready_gate(role, cases, base_rows=None):
    from verify_fix import evaluate_gate1_bank, evaluate_gate2_regression
    ids = [case_id(*c[:3]) for c in cases]
    if base_rows is None:
        base_rows = full_rows(BASE + "-" + role)
    new_rows = full_rows(NEW + "-" + role)
    checked(base_rows, ids, f"{role} control")
    checked(new_rows, ids, f"{role} candidate")
    gate = (evaluate_gate1_bank if role == "bank" else evaluate_gate2_regression)(
        base_rows, new_rows, expected_ids=ids, allow_group_mismatch=True)
    pairs = paired_comparison(base_rows, new_rows, B=2000, seed=SEED)
    doc = {"role": role, "gate": gate, "paired": pairs}
    write_json(OUT / f"gate-{role}-ready.json", doc)
    print(role, "gate", gate["passed"], pairs["n_pairs"], pairs["diff"], flush=True)

@contextmanager
def fresh_lock():
    import msvcrt
    path = OUT / "fresh.lock"
    if not path.exists():
        path.touch()
    with path.open("r+b") as guard:
        while True:
            guard.seek(0)
            try:
                msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(10)
        try:
            yield
        finally:
            guard.seek(0)
            msvcrt.locking(guard.fileno(), msvcrt.LK_UNLCK, 1)

def run_fresh_pending():
    with fresh_lock():
        fresh_off = jobs(BASE + "-fresh", "loop_02_400", alarm=False)
        fresh_on = jobs(NEW + "-fresh", "loop_02_400")
        fresh_jobs = sorted([*fresh_off, *fresh_on],
                            key=lambda x: (x[1]["experiment"]["layout_id"], x[1]["experiment"]["case_id"]))
        if fresh_jobs:
            execute("fresh_seeds_verification", fresh_jobs)
        write_json(OUT / "fresh-state.json", {"state": "complete", "expected_runs": 800})

def main():
    SUITES.mkdir(parents=True, exist_ok=True)
    prior = read_json(REPO / "config/agent_bench/v2/manifest.json")
    if not (SUITES / "manifest.json").exists():
        for name in ("loop_02_400", "probe_clutter_12"):
            source = REPO / "config/agent_bench/v2" / (name + ".json")
            (SUITES / source.name).write_bytes(source.read_bytes())
        prior["suites"] = {n: prior["suites"][n] for n in ("loop_02_400", "probe_clutter_12")}
        write_json(SUITES / "manifest.json", prior)
    runner.SUITES = SUITES
    criteria = OUT / "criteria.json"
    if not criteria.exists():
        write_json(criteria, {"cluster": "drift-alarm", "bank_min_fix_rate": 0.70,
            "regression_max_net_regression": 1, "regression_min_mcnemar_p": 0.20,
            "fresh_min_ci_lo": -0.02, "bootstrap_b": 2000, "seed": SEED, "metric": "success_rate",
            "bank_records": 8, "regression_pairs": 100, "fresh_layouts": 200, "fresh_pairs": 400,
            "first_fresh_seed": 3601001, "code_commit": runner.code_commit()})
    selected = original_bank()
    bank_path = suite("drift_alarm_bank8", [r["case"] for r in selected])
    if not (ROOT / "reports" / CF / "counterfactual.jsonl").is_file():
        execute("counterfactual", counterfactual_plan(selected))
        publish_cf(selected)
    if not (ROOT / "reports" / PROBE / "failure-rates.md").is_file():
        execute("clutter_probe", [j for cam in ("fixed", "recal", "drift") for j in jobs(PROBE, "probe_clutter_12", cam)])
        probe_report()
    execute("bank_verification", [*jobs(BASE + "-bank", "drift_alarm_bank8", alarm=False),
                                  *jobs(NEW + "-bank", "drift_alarm_bank8")])
    publish_ready_gate("bank", read_json(bank_path))
    reg_path, reg_control = prep_regression(selected)
    execute("regression_verification", jobs(NEW + "-regression", "drift_regression_selected100"))
    publish_ready_gate("regression", read_json(reg_path), base_rows=reg_control)
    write_json(OUT / STATE_NAME, {"state": "waiting_for_fresh_pairs", "regression_pairs": 100})
    run_fresh_pending()
    for inner, batch in ((BASE, (BASE + "-bank", BASE + "-fresh")),
                         (NEW, (NEW + "-bank", NEW + "-regression", NEW + "-fresh"))):
        merged = ROOT / "reports" / inner
        merged.mkdir(parents=True, exist_ok=True)
        write_jsonl(merged / "scores.jsonl", [row for name in batch for row in full_rows(name)])
    final = verify(bank_path, reg_path, reg_control)
    write_json(OUT / STATE_NAME, {"state": "completed", "verdict": final})

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(OUT / STATE_NAME, {"state": "error", "error": repr(exc)})
        raise

