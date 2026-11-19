"""Offline evaluation metrics for S9: per-segment-kind action error (rad) and End-Step F1 (±2 frames)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..constants import JOINTS
from .contract import ACTION_DIM, IGNORE_INDEX


def _to_numpy(arr: Any, dtype: Any | None = None) -> np.ndarray:
	if type(arr).__module__.startswith("torch"):
		arr = arr.detach().cpu().numpy()
	return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)


def compute_end_step_metrics(
	end_preds: Any,
	end_targets: Any,
	horizon: int,
	*,
	tolerance: int = 2,
	ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
	"""Compute End-Step F1 within ``±tolerance`` frames plus confusion breakdown and MAE.

	Class semantics (``NUM_END_CLASSES = horizon + 2``):
	  - Positive (in-chunk): ``0 <= k <= horizon``
	  - Negative (not within chunk): ``horizon + 1``
	  - Ignored: ``ignore_index`` (``-100``), excluded from all metrics

	A predicted in-chunk step ``k_hat`` (``0 <= k_hat <= horizon``) is a TP iff ground truth is
	in-chunk ``k`` (``0 <= k <= horizon``) AND ``|k_hat - k| <= tolerance``.
	"""
	preds = _to_numpy(end_preds, dtype=np.int64).reshape(-1)
	targets = _to_numpy(end_targets, dtype=np.int64).reshape(-1)
	if preds.shape != targets.shape:
		raise ValueError(f"end_preds shape {preds.shape} != end_targets shape {targets.shape}")

	valid_mask = targets != ignore_index
	preds = preds[valid_mask]
	targets = targets[valid_mask]
	n_valid = len(targets)
	if n_valid == 0:
		return {
			"valid_count": 0,
			"tolerance_frames": int(tolerance),
			"precision": 0.0,
			"recall": 0.0,
			"f1": 0.0,
			"exact_accuracy": 0.0,
			"end_step_mae": None,
			"end_step_mae_in_chunk": None,
			"end_step_mae_on_positives": None,
			"end_step_mae_all": None,
			"confusion": {
				"tp": 0,
				"fp": 0,
				"fn": 0,
				"tn": 0,
				"fp_on_negative": 0,
				"fn_missed": 0,
				"out_of_tolerance": 0,
				"true_positives_total": 0,
				"pred_positives_total": 0,
			},
		}

	true_pos = (targets >= 0) & (targets <= horizon)
	pred_pos = (preds >= 0) & (preds <= horizon)
	within_tol = np.abs(preds - targets) <= tolerance

	tp_mask = true_pos & pred_pos & within_tol
	out_of_tol_mask = true_pos & pred_pos & (~within_tol)
	fp_neg_mask = (~true_pos) & pred_pos
	fn_miss_mask = true_pos & (~pred_pos)
	tn_mask = (~true_pos) & (~pred_pos)

	tp = int(tp_mask.sum())
	out_of_tol = int(out_of_tol_mask.sum())
	fp_on_neg = int(fp_neg_mask.sum())
	fn_missed = int(fn_miss_mask.sum())
	tn = int(tn_mask.sum())

	n_pred_pos = int(pred_pos.sum())
	n_true_pos = int(true_pos.sum())
	fp = n_pred_pos - tp  # fp_on_neg + out_of_tol
	fn = n_true_pos - tp  # fn_missed + out_of_tol

	if n_pred_pos == 0 and n_true_pos == 0:
		precision = recall = f1 = 1.0
	else:
		precision = tp / n_pred_pos if n_pred_pos > 0 else 0.0
		recall = tp / n_true_pos if n_true_pos > 0 else 0.0
		denom = 2 * tp + fp + fn
		f1 = (2.0 * tp / denom) if denom > 0 else 0.0

	exact_acc = float((preds == targets).mean())
	both_in_chunk = true_pos & pred_pos
	abs_diff = np.abs(preds - targets).astype(np.float64)
	mae_in_chunk = float(abs_diff[both_in_chunk].mean()) if both_in_chunk.any() else None
	mae_pos = float(abs_diff[true_pos].mean()) if true_pos.any() else None
	mae_all = float(abs_diff.mean())
	primary_mae = mae_in_chunk if mae_in_chunk is not None else mae_all

	return {
		"valid_count": n_valid,
		"tolerance_frames": int(tolerance),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
		"exact_accuracy": exact_acc,
		"end_step_mae": primary_mae,
		"end_step_mae_in_chunk": mae_in_chunk,
		"end_step_mae_on_positives": mae_pos,
		"end_step_mae_all": mae_all,
		"confusion": {
			"tp": tp,
			"fp": fp,
			"fn": fn,
			"tn": tn,
			"fp_on_negative": fp_on_neg,
			"fn_missed": fn_missed,
			"out_of_tolerance": out_of_tol,
			"true_positives_total": n_true_pos,
			"pred_positives_total": n_pred_pos,
		},
	}


def _action_error_summary(abs_err: np.ndarray, sq_err: np.ndarray) -> dict[str, Any]:
	"""Summarize pre-filtered ``(M, 6)`` valid action errors."""
	if len(abs_err) == 0:
		return {
			"valid_steps": 0,
			"mae_rad": None,
			"rmse_rad": None,
			"per_joint_mae_rad": None,
			"per_joint_rmse_rad": None,
		}
	joint_mae = abs_err.mean(axis=0)
	joint_rmse = np.sqrt(sq_err.mean(axis=0))
	return {
		"valid_steps": len(abs_err),
		"mae_rad": float(abs_err.mean()),
		"rmse_rad": float(np.sqrt(sq_err.mean())),
		"per_joint_mae_rad": {j: float(joint_mae[i]) for i, j in enumerate(JOINTS)},
		"per_joint_rmse_rad": {j: float(joint_rmse[i]) for i, j in enumerate(JOINTS)},
	}


def compute_action_metrics(
	pred_actions: Any,
	true_actions: Any,
	action_mask: Any,
	segment_kinds: Sequence[str],
) -> dict[str, Any]:
	"""Compute overall and per-segment-kind action MAE/RMSE (rad) over unmasked steps only."""
	pred = _to_numpy(pred_actions, dtype=np.float64)
	true = _to_numpy(true_actions, dtype=np.float64)
	mask = _to_numpy(action_mask, dtype=bool)
	if pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != ACTION_DIM:
		raise ValueError(f"expected pred/true actions of shape (N, H, {ACTION_DIM}), got {pred.shape} vs {true.shape}")
	if mask.shape != pred.shape[:2]:
		raise ValueError(f"action_mask shape {mask.shape} != {pred.shape[:2]}")
	if len(segment_kinds) != len(pred):
		raise ValueError(f"segment_kinds len {len(segment_kinds)} != batch len {len(pred)}")

	diff = pred - true
	abs_err_all = np.abs(diff)[mask]
	sq_err_all = (diff**2)[mask]
	overall = _action_error_summary(abs_err_all, sq_err_all)

	kinds_arr = np.asarray(list(segment_kinds), dtype=object)
	per_kind: dict[str, dict[str, Any]] = {}
	for kind in sorted(set(kinds_arr.tolist())):
		idx = kinds_arr == kind
		m_k = mask[idx]
		summary = _action_error_summary(np.abs(diff[idx])[m_k], (diff[idx] ** 2)[m_k])
		per_kind[kind] = {"samples": int(idx.sum()), **summary}

	return {
		"action_mae_rad": overall["mae_rad"],
		"action_rmse_rad": overall["rmse_rad"],
		"valid_action_steps": overall["valid_steps"],
		"per_joint_mae_rad": overall["per_joint_mae_rad"],
		"per_joint_rmse_rad": overall["per_joint_rmse_rad"],
		"per_segment_kind": per_kind,
	}


def compute_vla_metrics(
	pred_actions: Any,
	true_actions: Any,
	action_mask: Any,
	end_preds: Any,
	end_targets: Any,
	segment_kinds: Sequence[str],
	horizon: int,
	*,
	tolerance: int = 2,
) -> dict[str, Any]:
	"""Compute complete S9 evaluation metrics (per-kind action MAE/RMSE + End-Step F1 ±2 frames)."""
	act = compute_action_metrics(pred_actions, true_actions, action_mask, segment_kinds)
	end = compute_end_step_metrics(end_preds, end_targets, horizon=horizon, tolerance=tolerance)

	preds_np = _to_numpy(end_preds, dtype=np.int64).reshape(-1)
	targets_np = _to_numpy(end_targets, dtype=np.int64).reshape(-1)
	kinds_arr = np.asarray(list(segment_kinds), dtype=object)
	for kind, row in act["per_segment_kind"].items():
		idx = kinds_arr == kind
		row["end_step"] = compute_end_step_metrics(preds_np[idx], targets_np[idx], horizon=horizon, tolerance=tolerance)

	return {**act, "end_step": end}


@dataclass
class EvalAccumulator:
	"""Streaming accumulator for validation batches so large datasets need not stay resident in memory."""

	horizon: int
	tolerance: int = 2
	_abs_sum_by_kind: dict[str, np.ndarray] = field(default_factory=dict)
	_sq_sum_by_kind: dict[str, np.ndarray] = field(default_factory=dict)
	_steps_by_kind: dict[str, int] = field(default_factory=dict)
	_samples_by_kind: dict[str, int] = field(default_factory=dict)
	_end_preds: list[np.ndarray] = field(default_factory=list)
	_end_targets: list[np.ndarray] = field(default_factory=list)
	_end_kinds: list[str] = field(default_factory=list)

	def update(
		self,
		pred_actions: Any,
		true_actions: Any,
		action_mask: Any,
		end_preds: Any,
		end_targets: Any,
		segment_kinds: Sequence[str],
	) -> None:
		pred = _to_numpy(pred_actions, dtype=np.float64)
		true = _to_numpy(true_actions, dtype=np.float64)
		mask = _to_numpy(action_mask, dtype=bool)
		ep = _to_numpy(end_preds, dtype=np.int64).reshape(-1)
		et = _to_numpy(end_targets, dtype=np.int64).reshape(-1)
		kinds = [str(k) for k in segment_kinds]

		diff = pred - true
		abs_diff = np.abs(diff)
		sq_diff = diff**2

		for i, kind in enumerate(kinds):
			self._samples_by_kind[kind] = self._samples_by_kind.get(kind, 0) + 1
			m = mask[i]
			n_steps = int(m.sum())
			self._steps_by_kind[kind] = self._steps_by_kind.get(kind, 0) + n_steps
			if kind not in self._abs_sum_by_kind:
				self._abs_sum_by_kind[kind] = np.zeros(ACTION_DIM, dtype=np.float64)
				self._sq_sum_by_kind[kind] = np.zeros(ACTION_DIM, dtype=np.float64)
			if n_steps > 0:
				self._abs_sum_by_kind[kind] += abs_diff[i, m].sum(axis=0)
				self._sq_sum_by_kind[kind] += sq_diff[i, m].sum(axis=0)

		self._end_preds.append(ep)
		self._end_targets.append(et)
		self._end_kinds.extend(kinds)

	def compute(self) -> dict[str, Any]:
		total_steps = sum(self._steps_by_kind.values())
		total_abs = np.zeros(ACTION_DIM, dtype=np.float64)
		total_sq = np.zeros(ACTION_DIM, dtype=np.float64)
		for kind in self._abs_sum_by_kind:
			total_abs += self._abs_sum_by_kind[kind]
			total_sq += self._sq_sum_by_kind[kind]

		if total_steps > 0:
			j_mae = total_abs / total_steps
			j_rmse = np.sqrt(total_sq / total_steps)
			action_mae = float(j_mae.mean())
			action_rmse = float(np.sqrt((total_sq / total_steps).mean()))
			pj_mae = {j: float(j_mae[i]) for i, j in enumerate(JOINTS)}
			pj_rmse = {j: float(j_rmse[i]) for i, j in enumerate(JOINTS)}
		else:
			action_mae = action_rmse = pj_mae = pj_rmse = None

		all_preds = np.concatenate(self._end_preds) if self._end_preds else np.empty(0, dtype=np.int64)
		all_targets = np.concatenate(self._end_targets) if self._end_targets else np.empty(0, dtype=np.int64)
		kinds_arr = np.asarray(self._end_kinds, dtype=object)

		per_kind: dict[str, dict[str, Any]] = {}
		for kind in sorted(self._samples_by_kind):
			n_k = self._steps_by_kind[kind]
			if n_k > 0:
				k_j_mae = self._abs_sum_by_kind[kind] / n_k
				k_j_rmse = np.sqrt(self._sq_sum_by_kind[kind] / n_k)
				k_mae = float(k_j_mae.mean())
				k_rmse = float(np.sqrt((self._sq_sum_by_kind[kind] / n_k).mean()))
				k_pj_mae = {j: float(k_j_mae[i]) for i, j in enumerate(JOINTS)}
				k_pj_rmse = {j: float(k_j_rmse[i]) for i, j in enumerate(JOINTS)}
			else:
				k_mae = k_rmse = k_pj_mae = k_pj_rmse = None
			idx = kinds_arr == kind
			per_kind[kind] = {
				"samples": int(self._samples_by_kind[kind]),
				"valid_steps": int(n_k),
				"mae_rad": k_mae,
				"rmse_rad": k_rmse,
				"per_joint_mae_rad": k_pj_mae,
				"per_joint_rmse_rad": k_pj_rmse,
				"end_step": compute_end_step_metrics(
					all_preds[idx], all_targets[idx], horizon=self.horizon, tolerance=self.tolerance
				),
			}

		end = compute_end_step_metrics(all_preds, all_targets, horizon=self.horizon, tolerance=self.tolerance)
		return {
			"action_mae_rad": action_mae,
			"action_rmse_rad": action_rmse,
			"valid_action_steps": int(total_steps),
			"per_joint_mae_rad": pj_mae,
			"per_joint_rmse_rad": pj_rmse,
			"per_segment_kind": per_kind,
			"end_step": end,
		}
