"""Batch report: ``python -m digital_twin_armigo.experiments.report --root D:/Armigo_data_v2 --batch p1-bench16``.

Scans ``<root>/runs/<batch>/<engine>/<group>/<episode>/`` (``episode.json`` or only ``run_status.json``), scores every
episode and writes ``<root>/reports/<batch>/{scores.jsonl,summary.json,SUMMARY.md}``. ``plan.json`` beside a group
(``{"cases": [case_id, ...]}``) gives the planned count; without it planned equals what exists. Several attempts of a
case: the highest attempt (then the latest directory name) is scored, the others are only counted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .precursors import MECHANISMS
from .scoring import score_episode
from .stats import cell_metrics, minimum_detectable_effect, paired_comparison

VIEWER = "http://127.0.0.1:8767/inventory?q="
BOOTSTRAP_B = 2000


def _episode_dirs(batch_dir: Path) -> list[Path]:
	dirs = {p.parent for name in ("episode.json", "run_status.json") for p in batch_dir.rglob(name)}
	return sorted(d for d in dirs if len(d.relative_to(batch_dir).parts) == 3)


def scan_batch(batch_dir: Path) -> tuple[list[dict], dict]:
	"""Scored rows (latest attempt per case) and ``{(engine, group): {"planned": ids|None, "attempts": n}}``."""
	rows: dict[tuple, dict] = {}
	attempts: Counter = Counter()
	for d in _episode_dirs(batch_dir):
		engine, group = d.relative_to(batch_dir).parts[:2]
		row = score_episode(d)
		row["engine"] = row.get("engine") or engine
		row["group"] = row.get("group") or group
		row["engine"], row["group"] = engine, group  # the directory is the truth for grouping
		key = (engine, group, row["case_id"])
		attempts[key] += 1
		rank = (int(row.get("attempt") or 0), d.name)
		if key not in rows or rank > rows[key][0]:
			rows[key] = (rank, row)
	plans = {}
	for plan in batch_dir.glob("*/*/plan.json"):
		try:
			data = json.loads(plan.read_text(encoding="utf-8-sig"))
		except (OSError, ValueError):
			continue
		plans[(plan.parent.parent.name, plan.parent.name)] = list(data.get("cases") or [])
	out = []
	for key, (_, row) in rows.items():
		row["n_attempts"] = attempts[key]
		out.append(row)
	out.sort(key=lambda r: (r["engine"], r["group"], r["case_id"]))
	return out, plans


def coverage(rows: list[dict], plans: dict) -> dict:
	cells = defaultdict(list)
	for r in rows:
		cells[(r["engine"], r["group"])].append(r)
	for key in plans:
		cells.setdefault(key, [])
	out = {}
	for key, cell in sorted(cells.items()):
		seen = {r["case_id"] for r in cell}
		planned_ids = plans.get(key)
		planned = len(planned_ids) if planned_ids is not None else len(cell)
		infra = [r for r in cell if r["infra_failed"] or not r["has_episode"]]
		out[key] = {
			"planned": planned,
			"completed": sum(r["has_episode"] and not r["infra_failed"] for r in cell),
			"infra_failed": len(infra),
			"missing": sorted(set(planned_ids) - seen) if planned_ids is not None else [],
		}
	return out


def _num(x, pct=False):
	if x is None or (isinstance(x, float) and math.isnan(x)):
		return "-"
	return f"{100 * x:.1f}%" if pct else f"{x:.2f}"


def _rate(m):
	return f"{_num(m['value'], True)} [{_num(m['ci_lo'], True)}, {_num(m['ci_hi'], True)}] ({m['k']}/{m['n']})"


def mechanism_table(rows: list[dict], field: str = "mechanism") -> dict[str, dict[str, int]]:
	"""Count each scored symptom by its mechanism or upstream field."""
	out: dict[str, dict[str, int]] = {}
	for r in rows:
		if r.get("root_cause") in (None, "none"):
			continue
		cell = out.setdefault(r["root_cause"], {})
		key = r.get(field) or "(none)"
		cell[key] = cell.get(key, 0) + 1
	return out


def build_summary(rows: list[dict], plans: dict, compare: tuple[str, str] | None = None, B: int = BOOTSTRAP_B) -> dict:
	cells = defaultdict(list)
	for r in rows:
		cells[(r["engine"], r["group"])].append(r)
	cov = coverage(rows, plans)
	summary = {
		"cells": {
			f"{e}/{g}": {
				"coverage": cov.get((e, g)),
				"metrics": cell_metrics(c, B=B),
				"root_causes": dict(Counter(r["root_cause"] for r in c)),
				"physical_outcomes": dict(Counter(r["physical_outcome"] for r in c)),
			}
			for (e, g), c in sorted(cells.items())
		},
		"coverage_only": {f"{e}/{g}": v for (e, g), v in cov.items() if (e, g) not in cells},
		"mechanisms": mechanism_table(rows),
		"upstreams": mechanism_table(rows, "upstream"),
		"paired": {},
	}
	if compare:
		a, b = compare
		for engine in sorted({e for e, _ in cells}):
			if (engine, a) in cells and (engine, b) in cells:
				pc = paired_comparison(cells[(engine, a)], cells[(engine, b)], B=B)
				pc["mde"] = minimum_detectable_effect(
					pc["discordance_rate"] if not math.isnan(pc["discordance_rate"]) else 0.0, pc["n_clusters"]
				)
				summary["paired"][engine] = {"a": a, "b": b, **pc}
	summary["failures"] = [
		{
			"case_id": r["case_id"],
			"engine": r["engine"],
			"group": r["group"],
			"label": r["label"],
			"expected": r["expected_label"],
			"outcome": r["physical_outcome"],
			"root_cause": r["root_cause"],
			"detail": r["first_error"].get("detail"),
			"frame": r["first_error"].get("frame"),
			"link": VIEWER + r["case_id"],
		}
		for r in rows
		if not r["physical_success"] or r["root_cause"] not in ("none",)
	]
	return summary


def render_markdown(batch: str, summary: dict) -> str:
	lines = [f"# 批次报告 {batch}", ""]
	lines += [
		"## 计划 / 完成 / 基础设施失败",
		"",
		"| 引擎/组 | 计划 | 完成 | 基础设施失败 | 缺失 |",
		"|---|---:|---:|---:|---:|",
	]
	for name, cell in {**{k: v for k, v in summary["cells"].items()}, **summary["coverage_only"]}.items():
		cov = cell.get("coverage", cell)
		lines.append(
			f"| {name} | {cov['planned']} | {cov['completed']} | {cov['infra_failed']} | {len(cov['missing'])} |"
		)
	lines += ["", "## 各格指标（布局聚类 95% 区间）", ""]
	lines += [
		"| 引擎/组 | 回合 | 物理分拣成功 | 缺陷漏检 | 良品误报 | unknown | 错碗 | 额外取景 | 决策延迟 s |",
		"|---|---:|---|---|---|---|---|---:|---:|",
	]
	for name, cell in summary["cells"].items():
		m = cell["metrics"]
		lines.append(
			f"| {name} | {m['n_episodes']} | {_rate(m['success_rate'])} | {_rate(m['defect_miss_rate'])} | "
			f"{_rate(m['false_alarm_rate'])} | {_rate(m['unknown_rate'])} | {_rate(m['wrong_bowl_rate'])} | "
			f"{_num(m['extra_views'])} | {_num(m['decision_latency_s'])} |"
		)
	lines += ["", "## 第一处错误分布", ""]
	causes = sorted({c for cell in summary["cells"].values() for c in cell["root_causes"]})
	if causes:
		lines += ["| 引擎/组 | " + " | ".join(causes) + " |", "|---|" + "---:|" * len(causes)]
		for name, cell in summary["cells"].items():
			lines.append(f"| {name} | " + " | ".join(str(cell["root_causes"].get(c, 0)) for c in causes) + " |")
	for key, title, expected in (
		("mechanisms", "症状 × 机制", MECHANISMS),
		("upstreams", "症状 × 上游来源", ("calibration_drift",)),
	):
		table = summary.get(key) or {}
		if not table:
			continue
		present = {label for counts in table.values() for label in counts}
		cols = [label for label in expected if label in present or key == "upstreams"]
		cols += sorted(present - set(cols) - {"(none)"}) + ["(none)"]
		lines += ["", f"## {title}（前兆，未经反事实证实）", ""]
		if key == "upstreams":
			lines += ["上游来源单独计数。同一例可在机制表计入积木位姿误差，在本表计入标定漂移。", ""]
		lines += ["| 症状 | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
		for symptom, counts in sorted(table.items()):
			lines.append(f"| {symptom} | " + " | ".join(str(counts.get(c, 0)) for c in cols) + " |")
	if summary["paired"]:
		lines += ["", "## 配对差异（B - A，按 case_id 配对，按布局聚类）", ""]
		lines += [
			"| 引擎 | A | B | 配对数 | 布局数 | 差异 | 95% 区间 | n10 | n01 | McNemar p | MDE |",
			"|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
		]
		for engine, p in summary["paired"].items():
			lines.append(
				f"| {engine} | {p['a']} | {p['b']} | {p['n_pairs']} | {p['n_clusters']} | {_num(p['diff'], True)} | "
				f"[{_num(p['ci_lo'], True)}, {_num(p['ci_hi'], True)}] | {p['n10']} | {p['n01']} | "
				f"{_num(p['p_mcnemar'])} | {_num(p['mde'], True)} |"
			)
	lines += ["", "## 失败与非 none 根因案例", ""]
	for f in summary["failures"]:
		lines.append(
			f"- [{f['case_id']}]({f['link']}) {f['engine']}/{f['group']}: {f['root_cause']} "
			f"(标签 {f['label']} / 期望 {f['expected']}, 落点 {f['outcome']}) {f['detail'] or ''}"
		)
	return "\n".join(lines) + "\n"


def write_report(root: Path, batch: str, compare: tuple[str, str] | None = None, B: int = BOOTSTRAP_B) -> Path:
	batch_dir = Path(root) / "runs" / batch
	if not batch_dir.is_dir():
		raise FileNotFoundError(batch_dir)
	rows, plans = scan_batch(batch_dir)
	summary = build_summary(rows, plans, compare, B=B)
	out = Path(root) / "reports" / batch
	out.mkdir(parents=True, exist_ok=True)
	with (out / "scores.jsonl").open("w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")
	(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
	(out / "SUMMARY.md").write_text(render_markdown(batch, summary), encoding="utf-8")
	return out


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--root", required=True, help="data root, e.g. D:/Armigo_data_v2")
	parser.add_argument("--batch", required=True)
	parser.add_argument("--compare", nargs=2, metavar=("GROUP_A", "GROUP_B"), help="paired difference B - A")
	parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
	args = parser.parse_args(argv)
	out = write_report(Path(args.root), args.batch, tuple(args.compare) if args.compare else None, args.bootstrap)
	print(out)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
