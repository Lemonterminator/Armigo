"""Train SmolVLA or ACT student policies with masked chunk losses and H+2 End-Step Head (plan S9).

Usage (from repo root in E:\\il\\lerobot-cuda):
	E:\\il\\lerobot-cuda\\Scripts\\python.exe scripts/vla/train.py \\
		--family smolvla --dataset E:/scratch/pilot/lerobot --val-dataset E:/scratch/val/lerobot \\
		--out E:/scratch/runs/smolvla-pilot --init-from lerobot/smolvla_base --steps 20000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.tracking.runs import tracked_run
from digital_twin_armigo.vla.contract import (
	CONTRACT_VERSION,
	DEFAULT_HORIZON,
	STATE_DIM,
	STUDENT_CAMERAS,
)
from digital_twin_armigo.vla.dataset import SegmentChunkDataset, StudentImageAugmentor, vla_collate_fn
from digital_twin_armigo.vla.metrics import EvalAccumulator
from digital_twin_armigo.vla.models import ACTWithEndHead, SmolVLAWithEndHead, load_act_weights_expanding

S9_END_STEP_F1_GATE = 0.9
DEFAULT_END_HEAD_LR = {"act": 1e-3, "smolvla": 1e-4}
SELECTION_RULE = (
	"min action_mae_rad among evals with end_step_f1 >= 0.9; if none qualifies, max end_step_f1 "
	"(diagnostic only, S9 fail)"
)


def _is_better_selection(cand: dict[str, Any], current: dict[str, Any] | None) -> bool:
	"""A3 rule: gate-passing candidates rank by lowest action MAE; until one passes, rank by highest F1."""
	if current is None:
		return True
	cand_pass = cand["end_step_f1"] >= S9_END_STEP_F1_GATE
	cur_pass = current["end_step_f1"] >= S9_END_STEP_F1_GATE
	if cand_pass != cur_pass:
		return cand_pass
	if cand_pass:
		return cand["action_mae_rad"] < current["action_mae_rad"]
	if abs(cand["end_step_f1"] - current["end_step_f1"]) > 1e-6:
		return cand["end_step_f1"] > current["end_step_f1"]
	return cand["action_mae_rad"] < current["action_mae_rad"]


def _argbest(history: list[dict[str, Any]], key: str, *, maximize: bool) -> int | None:
	if not history:
		return None
	pick = max if maximize else min
	return int(pick(history, key=lambda r: r[key])["step"])


def _seed_everything(seed: int) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
	if requested == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(requested)


def _cosine_warmup_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
	if total_steps <= 1:
		return 1.0
	w = min(warmup_steps, max(total_steps // 4, 1)) if warmup_steps > 0 else 0
	if w > 0 and step < w:
		return float(step + 1) / float(w)
	progress = float(step - w) / float(max(total_steps - w, 1))
	progress = min(max(progress, 0.0), 1.0)
	return 0.025 + 0.975 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _cycle_batches(loader: DataLoader):
	while True:
		yield from loader


def evaluate_model(
	model: SmolVLAWithEndHead | ACTWithEndHead,
	val_loader: DataLoader,
	*,
	max_batches: int | None = None,
) -> dict[str, Any]:
	"""Run closed-form chunk + end-step evaluation on ``val_loader`` and check S9 offline gate."""
	acc = EvalAccumulator(horizon=model.horizon, tolerance=2)
	was_training = model.training
	model.eval()
	try:
		for b_idx, batch in enumerate(val_loader):
			if max_batches is not None and b_idx >= max_batches:
				break
			pred_actions_rad, end_probs = model.predict_chunk(batch)
			end_preds = end_probs.argmax(dim=-1)
			acc.update(
				pred_actions=pred_actions_rad,
				true_actions=batch["action"],
				action_mask=batch["action_mask"],
				end_preds=end_preds,
				end_targets=batch["end_step_target"],
				segment_kinds=batch["segment_kind"],
			)
	finally:
		model.train(was_training)

	metrics = acc.compute()
	f1 = float(metrics["end_step"]["f1"])
	metrics["s9_gate_threshold"] = S9_END_STEP_F1_GATE
	metrics["s9_gate_pass"] = bool(f1 >= S9_END_STEP_F1_GATE)
	return metrics


def build_model(
	*,
	family: str,
	horizon: int,
	init_from: str | None,
	lambda_done: float,
	normalizer,
	class_weights: np.ndarray,
	camera_shapes: dict[str, tuple[int, int, int]] | None = None,
	tiny_for_test: bool = False,
	local_files_only: bool = False,
	bowl_token: bool = False,
) -> SmolVLAWithEndHead | ACTWithEndHead:
	if family == "smolvla":
		if bowl_token:
			raise ValueError("--bowl-token is ACT only")
		resolved_init = init_from if init_from is not None else ("scratch" if tiny_for_test else "lerobot/smolvla_base")
		if resolved_init == "scratch":
			resolved_init = None
		return SmolVLAWithEndHead(
			horizon=horizon,
			init_from=resolved_init,
			lambda_done=lambda_done,
			normalizer=normalizer,
			class_weights=class_weights,
			camera_shapes=camera_shapes,
			tiny_for_test=tiny_for_test,
			local_files_only=local_files_only,
		)
	if family == "act":
		model = ACTWithEndHead(
			horizon=horizon,
			lambda_done=lambda_done,
			normalizer=normalizer,
			class_weights=class_weights,
			camera_shapes=camera_shapes,
			tiny_for_test=tiny_for_test,
			bowl_token=bowl_token,
		)
		if init_from not in (None, "none", "scratch"):
			loaded = ACTWithEndHead.load_checkpoint(init_from, device="cpu")
			if bowl_token and not loaded.bowl_token:
				fresh = load_act_weights_expanding(model, loaded.state_dict())
				print(f"[init] expanded {init_from} with a bowl token; fresh tensors: {fresh}", flush=True)
			else:
				model.load_state_dict(loaded.state_dict(), strict=True)
			model.set_normalizer(normalizer)
			model.set_class_weights(class_weights)
		return model
	raise ValueError(f"unsupported --family {family!r}")


def run_training(
	*,
	family: str,
	train_ds: SegmentChunkDataset,
	val_ds: SegmentChunkDataset | None,
	out_dir: Path,
	init_from: str | None = None,
	lambda_done: float = 0.1,
	steps: int = 20_000,
	batch_size: int = 16,
	lr: float | None = None,
	backbone_lr: float | None = None,
	end_head_lr: float | None = None,
	weight_decay: float | None = None,
	warmup_steps: int = 500,
	grad_clip: float = 10.0,
	eval_every: int = 1000,
	save_every: int = 1000,
	log_every: int = 50,
	num_workers: int = 4,
	seed: int = 42,
	device: str = "auto",
	use_amp: bool = True,
	val_max_batches: int | None = None,
	camera_shapes: dict[str, tuple[int, int, int]] | None = None,
	tiny_for_test: bool = False,
	local_files_only: bool = False,
	bowl_token: bool = False,
	gpu_augmentor: StudentImageAugmentor | None = None,
	normalizer_from: Path | None = None,
) -> dict[str, Any]:
	"""Execute the S9 training loop on ``train_ds`` with optional periodic evaluation on ``val_ds``."""
	_seed_everything(seed)
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	dev = _resolve_device(device)
	horizon = train_ds.horizon
	default_lr = 1e-4 if family == "smolvla" else 1e-5
	default_wd = 1e-10 if family == "smolvla" else 1e-4
	lr_val = float(lr if lr is not None else default_lr)
	wd_val = float(weight_decay if weight_decay is not None else default_wd)
	# The End-Step Head starts from scratch: ACT's 1e-5 barely moves it in 20k steps (2026-11-20 probe on frozen
	# step-4000 features: val F1 0.12 at 1e-5 vs 0.32 at 1e-3). SmolVLA already trains at 1e-4.
	head_lr_val = float(end_head_lr if end_head_lr is not None else DEFAULT_END_HEAD_LR[family])

	# Fit stats and end-step class weights EXCLUSIVELY on the training split.
	if normalizer_from is not None:
		# Warm start on a narrower dataset (e.g. grasp-only): keep the initial checkpoint's input / action scaling so
		# its features stay valid, instead of refitting on the new data.
		from digital_twin_armigo.vla.contract import ActionNormalizer

		normalizer = ActionNormalizer.load(Path(normalizer_from) / "normalizer.json")
	else:
		normalizer = train_ds.fit_normalizer()
	class_weights = train_ds.end_step_class_weights()
	normalizer.save_alongside_checkpoint(out_dir)
	(out_dir / "train_summary.json").write_text(
		json.dumps(
			{
				"schema": CONTRACT_VERSION,
				"family": family,
				"horizon": horizon,
				"state_dim": STATE_DIM,
				"student_cameras": list(STUDENT_CAMERAS),
				"takeover_weight": train_ds.takeover_weight,
				"lambda_done": lambda_done,
				"lr": lr_val,
				"end_head_lr": head_lr_val,
				"weight_decay": wd_val,
				"bowl_token": bool(bowl_token),
				"gpu_augment": gpu_augmentor is not None,
				"init_from": init_from,
				"train": train_ds.summary(),
				"val": val_ds.summary() if val_ds is not None else None,
			},
			indent=2,
		)
		+ "\n",
		encoding="utf-8",
	)

	model = build_model(
		family=family,
		horizon=horizon,
		init_from=init_from,
		lambda_done=lambda_done,
		normalizer=normalizer,
		class_weights=class_weights,
		camera_shapes=camera_shapes,
		tiny_for_test=tiny_for_test,
		local_files_only=local_files_only,
		bowl_token=bowl_token,
	)
	model.to(dev)
	# Keep frozen VLM in bfloat16 on CUDA when bf16 AMP is active to reduce VRAM while keeping trainable weights fp32.
	if family == "smolvla" and dev.type == "cuda" and use_amp and model.train_expert_only:
		model.policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)

	if family == "act":
		param_groups = model.get_optim_param_groups(
			lr_val, weight_decay=wd_val, backbone_lr=backbone_lr, end_head_lr=head_lr_val
		)
	else:
		param_groups = model.get_optim_param_groups(lr_val, weight_decay=wd_val, end_head_lr=head_lr_val)
	trainable_params = [p for g in param_groups for p in g["params"]]
	optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95) if family == "smolvla" else (0.9, 0.999))
	scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer, lr_lambda=lambda s: _cosine_warmup_lambda(s, warmup_steps, steps)
	)

	pin_mem = dev.type == "cuda"
	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_mem,
		collate_fn=vla_collate_fn,
		persistent_workers=num_workers > 0,
	)
	val_loader = (
		DataLoader(
			val_ds,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_mem,
			collate_fn=vla_collate_fn,
			persistent_workers=num_workers > 0,
		)
		if val_ds is not None
		else None
	)

	metrics_path = out_dir / "metrics.jsonl"
	metrics_file = metrics_path.open("w", encoding="utf-8")
	best_f1 = -1.0
	best_mae = float("inf")
	val_history: list[dict[str, Any]] = []
	selected: dict[str, Any] | None = None
	last_val_metrics: dict[str, Any] | None = None
	batch_iter = _cycle_batches(train_loader)
	amp_enabled = dev.type == "cuda" and use_amp

	try:
		model.train()
		t0 = time.perf_counter()
		for step in range(1, steps + 1):
			batch = next(batch_iter)
			if gpu_augmentor is not None:  # --gpu-augment: the workers only decode; augment the batch on the device
				cams = {k: batch[k].to(dev, non_blocking=True) for k in STUDENT_CAMERAS}
				batch.update(gpu_augmentor.augment_batch(cams))
			optimizer.zero_grad(set_to_none=True)
			with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
				loss, loss_dict = model(batch)
			loss.backward()
			if grad_clip > 0:
				torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
			optimizer.step()
			scheduler.step()

			if step == 1 or step % log_every == 0 or step == steps:
				elapsed = time.perf_counter() - t0
				cur_lr = optimizer.param_groups[0]["lr"]
				row = {"step": step, "split": "train", "lr": cur_lr, "elapsed_s": round(elapsed, 3), **loss_dict}
				metrics_file.write(json.dumps(row) + "\n")
				metrics_file.flush()
				print(
					f"[step {step:6d}/{steps}] loss={loss_dict['loss']:.4f} "
					f"l_act={loss_dict['l_act']:.4f} l_done={loss_dict['l_done']:.4f} lr={cur_lr:.2e}",
					flush=True,
				)

			should_eval = val_loader is not None and (step % eval_every == 0 or step == steps)
			if should_eval:
				val_metrics = evaluate_model(model, val_loader, max_batches=val_max_batches)
				last_val_metrics = val_metrics
				end_m = val_metrics["end_step"]
				f1 = float(end_m["f1"])
				mae = val_metrics["action_mae_rad"]
				mae_f = float("inf") if mae is None else float(mae)
				metrics_file.write(json.dumps({"step": step, "split": "val", **val_metrics}) + "\n")
				metrics_file.flush()
				print(
					f"[val  {step:6d}] action_mae={mae if mae is not None else -1:.4f} rad "
					f"action_rmse={val_metrics['action_rmse_rad'] if val_metrics['action_rmse_rad'] is not None else -1:.4f} rad "
					f"end_f1={f1:.4f} (P={end_m['precision']:.3f}, R={end_m['recall']:.3f}) "
					f"s9_gate={'PASS' if val_metrics['s9_gate_pass'] else 'FAIL'}",
					flush=True,
				)
				best_f1 = max(best_f1, f1)
				best_mae = min(best_mae, mae_f)
				cand = {"step": step, "end_step_f1": f1, "action_mae_rad": mae_f}
				val_history.append(cand)
				if _is_better_selection(cand, selected):
					selected = {**cand, "s9_gate_pass": f1 >= S9_END_STEP_F1_GATE}
					model.save_checkpoint(
						out_dir / "best",
						extra_meta={"step": step, "val": val_metrics, "selection_rule": SELECTION_RULE},
					)

			if save_every > 0 and step % save_every == 0 and step < steps:
				model.save_checkpoint(out_dir / f"step-{step:06d}", extra_meta={"step": step})
	finally:
		metrics_file.close()

	last_ckpt = model.save_checkpoint(
		out_dir / "last",
		extra_meta={"step": steps, "val": last_val_metrics},
	)
	summary = {
		"schema": CONTRACT_VERSION,
		"family": family,
		"steps": steps,
		"horizon": horizon,
		"last_checkpoint": str(last_ckpt),
		"best_checkpoint": str(out_dir / "best") if (out_dir / "best").exists() else None,
		"selection_rule": SELECTION_RULE,
		"selected": selected,
		"best_end_step_f1": None if best_f1 < 0 else round(best_f1, 6),
		"best_end_step_f1_step": _argbest(val_history, "end_step_f1", maximize=True),
		"best_action_mae_rad": None if not val_history else best_mae,
		"best_action_mae_step": _argbest(val_history, "action_mae_rad", maximize=False),
		"val_history": val_history,
		# Gate of the SELECTED checkpoint (the one written to best/), not of the last evaluation.
		"s9_gate_pass": None if selected is None else bool(selected["s9_gate_pass"]),
		"last_s9_gate_pass": bool(last_val_metrics["s9_gate_pass"]) if last_val_metrics is not None else None,
		"final_val_metrics": last_val_metrics,
	}
	(out_dir / "final_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
	return summary


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--family", choices=("smolvla", "act"), required=True)
	p.add_argument("--dataset", type=Path, required=True, help="Training LeRobot v3 dataset directory")
	p.add_argument("--val-dataset", type=Path, default=None, help="Validation LeRobot v3 dataset directory")
	p.add_argument("--repo-id", default="local/armigo-six-face")
	p.add_argument("--out", type=Path, required=True, help="Output directory for checkpoints & metrics")
	p.add_argument("--init-from", default=None, help="Pretrained HF id / local checkpoint / 'vlm' / 'scratch'")
	p.add_argument("--horizon", type=int, default=None, help="Chunk horizon H (default: 50 smolvla, 100 act)")
	p.add_argument("--takeover-weight", type=float, default=2.0)
	p.add_argument("--lambda-done", type=float, default=0.1)
	p.add_argument("--steps", type=int, default=20_000)
	p.add_argument("--smoke-steps", type=int, default=None, help="Override --steps for a quick smoke training run")
	p.add_argument("--batch-size", type=int, default=16)
	p.add_argument("--lr", type=float, default=None)
	p.add_argument("--backbone-lr", type=float, default=None)
	p.add_argument("--end-head-lr", type=float, default=None, help="End-Step Head LR (default: 1e-3 act, 1e-4 smolvla)")
	p.add_argument("--weight-decay", type=float, default=None)
	p.add_argument("--warmup-steps", type=int, default=500)
	p.add_argument("--grad-clip", type=float, default=10.0)
	p.add_argument("--eval-every", type=int, default=1000)
	p.add_argument("--save-every", type=int, default=1000)
	p.add_argument("--log-every", type=int, default=50)
	p.add_argument("--num-workers", type=int, default=4)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--device", default="auto")
	p.add_argument("--no-amp", action="store_true", help="Disable CUDA bfloat16 AMP")
	p.add_argument("--no-augment", action="store_true", help="Disable training-time image augmentations")
	p.add_argument(
		"--gpu-augment",
		action="store_true",
		help="Augment whole batches on the training device instead of per sample in the DataLoader workers",
	)
	p.add_argument("--val-max-batches", type=int, default=None)
	p.add_argument("--local-files-only", action="store_true")
	p.add_argument(
		"--drop-unsupervised",
		action="store_true",
		help="Leave out training samples with no loss (fully masked actions and no end-step label)",
	)
	p.add_argument(
		"--bowl-token",
		action="store_true",
		help="ACT: target-bowl one-hot as its own encoder token (docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md)",
	)
	p.add_argument(
		"--normalizer-from",
		type=Path,
		default=None,
		help="Checkpoint directory whose normalizer.json to reuse instead of fitting one (warm start on narrower data)",
	)
	p.add_argument("--tiny-for-test", action="store_true", help="Use tiny in-memory backbone for fast smoke tests")
	p.add_argument("--mlflow", action="store_true", help="Record this run (params, metrics, data digests) in MLflow")
	p.add_argument("--mlflow-experiment", default="vla-train")
	p.add_argument("--run-name", default=None, help="MLflow run name (default: output directory name)")
	p.add_argument("--register-model", default=None, metavar="NAME", help="Register the final checkpoint in MLflow")
	args = p.parse_args(argv)

	horizon = int(args.horizon if args.horizon is not None else DEFAULT_HORIZON[args.family])
	total_steps = int(args.smoke_steps if args.smoke_steps is not None else args.steps)

	train_ds = SegmentChunkDataset(
		args.dataset,
		repo_id=args.repo_id,
		horizon=horizon,
		takeover_weight=args.takeover_weight,
		augment=not args.no_augment and not args.gpu_augment,
		drop_unsupervised=args.drop_unsupervised,
	)
	val_ds = (
		SegmentChunkDataset(
			args.val_dataset,
			repo_id=args.repo_id,
			horizon=horizon,
			takeover_weight=1.0,
			augment=False,
		)
		if args.val_dataset is not None
		else None
	)

	datasets = {"train": args.dataset, **({"val": args.val_dataset} if args.val_dataset else {})}
	with tracked_run(
		args.mlflow_experiment,
		args.run_name or args.out.name,
		enabled=args.mlflow,
		params={k: v for k, v in vars(args).items() if not k.startswith("mlflow")} | {"horizon_resolved": horizon},
		tags={"task": "vla_train", "family": args.family},
		datasets=datasets,
	) as run:
		summary = run_training(
			family=args.family,
			train_ds=train_ds,
			val_ds=val_ds,
			out_dir=args.out,
			init_from=args.init_from,
			lambda_done=args.lambda_done,
			steps=total_steps,
			batch_size=args.batch_size,
			lr=args.lr,
			backbone_lr=args.backbone_lr,
			end_head_lr=args.end_head_lr,
			weight_decay=args.weight_decay,
			warmup_steps=args.warmup_steps,
			grad_clip=args.grad_clip,
			eval_every=args.eval_every,
			save_every=args.save_every,
			log_every=args.log_every,
			num_workers=args.num_workers,
			seed=args.seed,
			device=args.device,
			use_amp=not args.no_amp,
			val_max_batches=args.val_max_batches,
			tiny_for_test=args.tiny_for_test,
			local_files_only=args.local_files_only,
			bowl_token=args.bowl_token,
			gpu_augmentor=StudentImageAugmentor(seed=args.seed) if args.gpu_augment and not args.no_augment else None,
			normalizer_from=args.normalizer_from,
		)
		run.log_report(summary["final_val_metrics"] or {}, "val")
		run.log_metrics({"best_end_step_f1": summary["best_end_step_f1"]})
		run.set_tags({"s9_gate_pass": summary["s9_gate_pass"]})
		for f in (*args.out.glob("*summary.json"), *args.out.glob("*.jsonl")):
			run.log_artifact(f)
		if args.register_model:
			run.log_artifact(Path(summary["best_checkpoint"] or summary["last_checkpoint"]), "model")
			run.register_model(args.register_model, "model", "candidate")
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
