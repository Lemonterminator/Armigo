"""Diagnosis benchmark: human-confirmed failures, split by a hash of the trajectory id into calib / heldout.

Rules and thresholds may only be tuned on ``calib``; ``heldout`` is evaluated once per round (plan section 2.3)."""

from __future__ import annotations

import hashlib
from collections import Counter


def split_of(traj_id: str) -> str:
	return "calib" if hashlib.sha256(traj_id.encode()).digest()[0] % 2 == 0 else "heldout"


def cohen_kappa(a: list[str], b: list[str]) -> float:
	n = len(a)
	if n == 0:
		return float("nan")
	po = sum(x == y for x, y in zip(a, b, strict=True)) / n
	ca, cb = Counter(a), Counter(b)
	pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
	return 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)


def agreement(bench: list[dict], auto: dict[str, dict], split: str) -> dict:
	items = [b for b in bench if b["split"] == split and b["traj_id"] in auto]
	human = [b["root_cause"] for b in items]
	machine = [auto[b["traj_id"]].get("root_cause") or "other" for b in items]
	mech = [b for b in items if b.get("mechanism")]
	mech_hits = sum(auto[b["traj_id"]].get("mechanism") == b["mechanism"] for b in mech)
	return {
		"n": len(items),
		"root_cause_acc": sum(h == m for h, m in zip(human, machine, strict=True)) / len(items)
		if items
		else float("nan"),
		"root_cause_kappa": cohen_kappa(human, machine),
		"mechanism_acc": mech_hits / len(mech) if mech else float("nan"),
		"mechanism_n": len(mech),
	}
