"""Gold layer: deterministic data-product manifests, data cards, and training isolation invariants."""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..vla import dagger
from .bronze import _commit
from .segments import DEFAULT_SEED_RANGES, is_sealed

__all__ = ["PRODUCTS", "GovernanceError", "Product", "assert_product_invariants", "build_product"]


class GovernanceError(RuntimeError):
	"""Raised when a trainable or gold manifest violates isolation invariants."""


@dataclass(frozen=True)
class Product:
	name: str
	sensing: str | None
	predicate: Callable[[dict, dict], bool]
	trainable: bool


def _base(r: dict) -> bool:
	return not r.get("quarantined") and not (r.get("flags") or {}).get("oracle") and not r.get("superseded")


def _void_streak_of(r: dict, ctx: dict) -> int:
	streaks = ctx["void_streak"]
	vs_key = r.get("void_streak_key")
	if vs_key is not None and vs_key in streaks:
		return int(streaks[vs_key])
	pk_str = r.get("pair_key_str")
	if pk_str is not None and pk_str in streaks:
		return int(streaks[pk_str])
	pk = r.get("pair_key")
	if isinstance(pk, list):
		return int(streaks.get(json.dumps(pk), 0))
	if pk is not None:
		return int(streaks.get(pk, 0))
	return 0


def _is_sealed_row(r: dict) -> bool:
	return r.get("split") == "test" or is_sealed(r.get("seed"), batch=r.get("batch"), options=r.get("case_options"))


def _matches_freeze(r: dict, ctx: dict) -> bool:
	freeze = ctx.get("freeze")
	if not isinstance(freeze, dict) or not freeze.get("code_commit"):
		return False
	if not r.get("teacher_frozen"):
		return False
	ep_commit = r.get("provenance_commit")
	if ep_commit is not None and ep_commit != freeze.get("code_commit"):
		return False
	f_det = freeze.get("detector")
	ep_det = r.get("detector")
	return not (f_det and ep_det and str(ep_det) != str(f_det))


PRODUCTS: dict[str, Product] = {
	"student_train_v3": Product(
		"student_train_v3",
		"realistic",
		lambda r, ctx: (
			_base(r)
			and not _is_sealed_row(r)
			and r.get("split") == "train"
			and bool(r.get("physical_success"))
			and _matches_freeze(r, ctx)
			and (
				r.get("pair_outcome") == "both_pass"
				or (r.get("pair_outcome") == "unpaired" and ctx["realistic_only_rate"].get(r.get("batch"), 1.0) < 0.01)
			)
		),
		trainable=True,
	),
	"privileged_demos_v3": Product(
		"privileged_demos_v3",
		"truth",
		lambda r, ctx: (
			_base(r)
			and not _is_sealed_row(r)
			and r.get("split") == "train"
			and _matches_freeze(r, ctx)
			and r.get("pair_outcome") == "truth_only"
		),
		trainable=True,
	),
	"failure_bank": Product(
		"failure_bank",
		"realistic",
		lambda r, ctx: _base(r) and not _is_sealed_row(r) and r.get("pair_outcome") in ("truth_only", "both_fail"),
		trainable=False,
	),
	"harness_review": Product(
		"harness_review",
		None,
		lambda r, ctx: (
			not _is_sealed_row(r)
			and (
				bool(r.get("quarantined")) or r.get("pair_outcome") == "realistic_only" or _void_streak_of(r, ctx) >= 2
			)
		),
		trainable=False,
	),
}

for _suite in (k for k in DEFAULT_SEED_RANGES if k != "train"):
	PRODUCTS[f"eval_{_suite}"] = Product(
		f"eval_{_suite}",
		"realistic",
		lambda r, ctx, s=_suite: _base(r) and r.get("split") == s and (s != "test" or _matches_freeze(r, ctx)),
		trainable=False,
	)


def assert_product_invariants(rows: list[dict], product: Product, *, freeze: dict | None = None) -> None:
	"""Enforce hard isolation constraints before any gold manifest is written."""
	if product.name != "eval_test":
		bad_sealed = [r.get("path") for r in rows if _is_sealed_row(r)]
		if bad_sealed:
			raise GovernanceError(f"{product.name} cannot contain sealed test episodes: {bad_sealed[:3]}")
	if product.name == "eval_test" and rows:
		ctx = {"freeze": freeze}
		for r in rows:
			if not _matches_freeze(r, ctx):
				raise GovernanceError(f"eval_test requires a matching freeze for {r.get('path')!r}")
	if not product.trainable:
		return
	ctx = {"freeze": freeze}
	case_entries = []
	for r in rows:
		if r.get("split") != "train":
			raise GovernanceError(f"{product.name}: non-train split {r.get('split')!r} in {r.get('path')!r}")
		if r.get("quarantined"):
			raise GovernanceError(f"{product.name}: quarantined episode {r.get('path')!r}")
		if (r.get("flags") or {}).get("oracle"):
			raise GovernanceError(f"{product.name}: oracle episode {r.get('path')!r}")
		if r.get("pair_outcome") == "realistic_only":
			raise GovernanceError(f"{product.name}: realistic_only episode {r.get('path')!r}")
		if freeze is not None and not _matches_freeze(r, ctx):
			raise GovernanceError(f"{product.name}: freeze mismatch in {r.get('path')!r}")
		if r.get("seed") is not None:
			opts = r.get("case_options")
			case_entries.append([int(r["seed"]), False, 0, opts] if isinstance(opts, dict) else [int(r["seed"])])
	if case_entries:
		try:
			dagger.assert_train_layouts_only(cases=case_entries)
		except ValueError as exc:
			raise GovernanceError(str(exc)) from exc


def _predicate_sha(product: Product) -> str:
	try:
		src = inspect.getsource(product.predicate)
	except (OSError, TypeError):
		src = product.name
	return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


def _truth_outcome_priority(outcomes: list[str]) -> str:
	if not outcomes:
		return "unpaired"
	for o in ("realistic_only", "truth_only", "both_pass", "both_fail", "void"):
		if o in outcomes:
			return o
	return "unpaired"


def _prepare_rows_and_ctx(con: sqlite3.Connection, silver_build: str, freeze: dict | None) -> tuple[list[dict], dict]:
	raw_eps = [
		json.loads(r[0])
		for r in con.execute("select row_json from silver_episode where build_id = ? order by path", (silver_build,))
	]
	pairs = con.execute(
		"select batch, pair_key, camera_condition, truth_path, realistic_path, outcome, match_scope "
		"from silver_pair where build_id = ?",
		(silver_build,),
	).fetchall()

	by_real_path: dict[str, tuple[str, str]] = {}
	by_truth_path: dict[str, list[str]] = {}
	batch_valid: Counter = Counter()
	batch_real_only: Counter = Counter()
	ep_by_path = {e["path"]: e for e in raw_eps}

	for batch_col, _pk_json, _cam, t_path, r_path, outcome, scope in pairs:
		if r_path:
			by_real_path[r_path] = (outcome, scope)
		if t_path:
			by_truth_path.setdefault(t_path, []).append(outcome)
		if outcome not in ("void", "unpaired"):
			ref = ep_by_path.get(r_path) or ep_by_path.get(t_path) or {}
			b = batch_col if batch_col is not None else ref.get("batch")
			batch_valid[b] += 1
			if outcome == "realistic_only":
				batch_real_only[b] += 1

	realistic_only_rate: dict[str | None, float] = {}
	for b in {e.get("batch") for e in raw_eps}:
		denom = batch_valid.get(b, 0)
		realistic_only_rate[b] = (batch_real_only.get(b, 0) / denom) if denom > 0 else 1.0

	attempts_by_key: dict[tuple, list[dict]] = {}
	for e in raw_eps:
		pk = e.get("pair_key")
		pk_json = json.dumps(pk) if pk is not None else None
		e["pair_key_str"] = pk_json
		sensing = e.get("sensing")
		eligible = not e.get("superseded") and pk_json is not None and sensing in ("truth", "realistic")
		if sensing == "realistic" and eligible:
			if e["path"] not in by_real_path:
				raise GovernanceError(
					f"missing silver_pair row for latest realistic episode {e['path']!r}; rebuild silver_pair before gold"
				)
			e["pair_outcome"], e["match_scope"] = by_real_path[e["path"]]
		elif sensing == "truth" and eligible:
			if e["path"] not in by_truth_path:
				raise GovernanceError(
					f"missing silver_pair row for latest truth episode {e['path']!r}; rebuild silver_pair before gold"
				)
			e["pair_outcome"] = _truth_outcome_priority(by_truth_path[e["path"]])
			e["match_scope"] = "paired" if any(o != "unpaired" for o in by_truth_path[e["path"]]) else "unpaired"
		else:
			e["pair_outcome"] = "unpaired"
			e["match_scope"] = "excluded"
		if pk_json is not None:
			vkey = (e.get("batch"), pk_json, e.get("camera_condition") or "fixed")
			e["void_streak_key"] = vkey
			attempts_by_key.setdefault(vkey, []).append(e)

	void_streak: dict[tuple, int] = {}
	for vkey, group in attempts_by_key.items():
		group.sort(key=lambda r: (int(r.get("attempt") or 0), r["path"]))
		streak = 0
		for r in reversed(group):
			if r.get("quarantined") or r.get("infra_failed") or r.get("pair_outcome") == "void":
				streak += 1
			else:
				break
		void_streak[vkey] = streak

	ctx = {"realistic_only_rate": realistic_only_rate, "void_streak": void_streak, "freeze": freeze}
	return raw_eps, ctx


def _exclude_reason(r: dict, product: Product, ctx: dict) -> str:
	if product.sensing is not None and r.get("sensing") != product.sensing:
		return f"sensing!={product.sensing}"
	if r.get("superseded"):
		return "superseded"
	if r.get("quarantined"):
		return "quarantined:" + ",".join(r.get("quarantine_reasons") or ["unknown"])
	if (r.get("flags") or {}).get("oracle"):
		return "oracle"
	if product.name == "student_train_v3":
		if r.get("split") != "train":
			return f"split={r.get('split')}"
		if not r.get("physical_success"):
			return "not_physical_success"
		if not r.get("teacher_frozen"):
			return "not_teacher_frozen"
		if r.get("pair_outcome") == "unpaired":
			return "unpaired_batch_high_realistic_only"
		return f"pair_outcome={r.get('pair_outcome')}"
	if product.name == "failure_bank":
		if r.get("split") == "test":
			return "sealed_test"
		return f"pair_outcome={r.get('pair_outcome')}"
	return f"predicate_false(split={r.get('split')},outcome={r.get('pair_outcome')})"


def build_product(
	con: sqlite3.Connection,
	name: str,
	*,
	silver_build: str,
	out_root: Path,
	freeze: dict | None = None,
) -> Path:
	"""Build one deterministic gold product manifest and card under ``<out_root>/gold/<name>/<version>/``."""
	if name not in PRODUCTS:
		raise KeyError(f"unknown gold product {name!r}; expected one of {sorted(PRODUCTS)}")
	product = PRODUCTS[name]
	rows, ctx = _prepare_rows_and_ctx(con, silver_build, freeze)

	selected: list[dict] = []
	excluded_counts: Counter = Counter()
	for r in rows:
		sensing_ok = product.sensing is None or r.get("sensing") == product.sensing
		if sensing_ok and product.predicate(r, ctx):
			selected.append(r)
		else:
			excluded_counts[_exclude_reason(r, product, ctx)] += 1

	assert_product_invariants(selected, product, freeze=freeze)
	selected.sort(key=lambda r: r["path"])

	manifest_items = [
		{
			"path": r["path"],
			"episode_sha256": r.get("sha_episode"),
			"split": r.get("split"),
			"pair_outcome": r.get("pair_outcome"),
			"flags": r.get("flags"),
			"root_cause": r.get("root_cause"),
			"mechanism": r.get("mechanism"),
		}
		for r in selected
	]
	manifest_bytes = "".join(
		json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in manifest_items
	).encode("utf-8")
	version = hashlib.sha256(manifest_bytes).hexdigest()[:12]
	pred_sha = _predicate_sha(product)
	commit = _commit()

	ver_dir = Path(out_root) / "gold" / name / version
	ver_dir.mkdir(parents=True, exist_ok=True)
	(ver_dir / "manifest.jsonl").write_bytes(manifest_bytes)

	by_split = Counter(r.get("split") for r in selected)
	by_cam = Counter(r.get("camera_condition") for r in selected)
	by_outcome = Counter(r.get("pair_outcome") for r in selected)
	card_lines = [
		f"# Gold Product `{name}` (`{version}`)",
		"",
		f"- silver_build: `{silver_build}`",
		f"- commit: `{commit}`",
		f"- predicate_sha: `{pred_sha}`",
		f"- n_items: `{len(selected)}`",
		"",
		"## By Split",
		*[f"- {k}: {v}" for k, v in sorted(by_split.items(), key=lambda kv: str(kv[0]))],
		"",
		"## By Camera Condition",
		*[f"- {k}: {v}" for k, v in sorted(by_cam.items(), key=lambda kv: str(kv[0]))],
		"",
		"## By Pair Outcome",
		*[f"- {k}: {v}" for k, v in sorted(by_outcome.items(), key=lambda kv: str(kv[0]))],
		"",
		"## Excluded Counts",
		*([f"- {k}: {v}" for k, v in sorted(excluded_counts.items())] or ["- (none)"]),
	]
	(ver_dir / "card.md").write_text("\n".join(card_lines) + "\n", encoding="utf-8", newline="\n")

	import datetime as dt

	con.execute(
		"insert or ignore into gold_build (product, version, silver_build, built_at, git_commit, predicate_sha, n_items) "
		"values (?, ?, ?, ?, ?, ?, ?)",
		(name, version, silver_build, dt.datetime.now(dt.UTC).isoformat(), commit, pred_sha, len(selected)),
	)
	con.commit()
	return ver_dir
