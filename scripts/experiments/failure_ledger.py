"""CLI: scan a batch, run automatic triage (R1-R8), join audit verdicts, and write the failure ledger.

Usage:
	python scripts/experiments/failure_ledger.py --root E:/Armigo_data_v3 --batch p2-dev-r2 --round 01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from digital_twin_armigo.experiments.ledger import build_ledger, render_ledger_md, triage_batch
from digital_twin_armigo.inventory import audit

DEFAULT_ROOT = Path("E:/Armigo_data_v3")


def _format_round(value: str) -> str:
	raw = value.strip()
	if raw.isdigit():
		return f"round-{int(raw):02d}"
	return raw if raw.startswith("round-") else f"round-{raw}"


def _load_reviews(audit_db: Path | None, traj_ids: list[str]) -> dict[str, dict]:
	if audit_db is None or not Path(audit_db).is_file():
		return {}
	reviews: dict[str, dict] = {}
	for tid in traj_ids:
		rev = audit.latest(audit_db, tid)
		if rev is not None:
			reviews[tid] = rev
	return reviews


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Data root containing runs/<batch>/")
	parser.add_argument("--batch", required=True, help="Batch name under <root>/runs/")
	parser.add_argument("--round", required=True, help="Loop round number or label (e.g. 01)")
	parser.add_argument("--out-dir", help="Output directory (default: <root>/reports/<batch>/round-<NN>/)")
	parser.add_argument("--audit-db", help="Audit SQLite path (default: <root>/audit/reviews.sqlite if present)")
	args = parser.parse_args(argv)

	root = Path(args.root)
	round_name = _format_round(args.round)
	out_dir = Path(args.out_dir) if args.out_dir else root / "reports" / args.batch / round_name
	audit_db = Path(args.audit_db) if args.audit_db else root / "audit" / "reviews.sqlite"

	rows, factors = triage_batch(root, args.batch)
	reviews = _load_reviews(audit_db, [r.traj_id for r in rows])
	ledger = build_ledger(rows, reviews=reviews, factors=factors)
	ledger_payload = {
		"batch": args.batch,
		"round": round_name,
		"created": datetime.now(UTC).isoformat(timespec="seconds"),
		**ledger,
	}

	out_dir.mkdir(parents=True, exist_ok=True)
	with (out_dir / "triage.jsonl").open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
	(out_dir / "ledger.json").write_text(
		json.dumps(ledger_payload, indent=2, ensure_ascii=False) + "\n",
		encoding="utf-8",
	)
	(out_dir / "ledger.md").write_text(
		render_ledger_md(ledger_payload, batch=args.batch, round_name=round_name),
		encoding="utf-8",
	)
	print(
		f"{out_dir}: {ledger['failure_episodes']}/{ledger['total_episodes']} failures, {len(ledger['clusters'])} clusters"
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
