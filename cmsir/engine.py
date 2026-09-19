import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from .data import discover, make_loader, training_loaders
from .alignment import aligned_mix_batch
from .losses import crossover_mask, mix_batch, prototype_loss, segmentation_loss, source_loss
from .metrics import SegmentationEvaluator, SegmentationMeter
from .model import SourceDiscriminator, UNet
from .utils import InfiniteLoader, Timer, ema_update, ensure_result_tree, format_duration, write_json

def learning_rate(epoch, cfg):
    peak, warmup = cfg["learning_rate"], cfg["warmup_epochs"]
    if epoch <= warmup:
        return peak * epoch / max(1, warmup)
    minimum = peak * cfg["minimum_lr_ratio"]
    progress = (epoch - warmup) / max(1, cfg["epochs"] - warmup)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2

@torch.inference_mode()
def reporting_labels(labels, cfg):
    mapping = cfg.get("reporting", {}).get("label_map")
    if mapping is None:
        return labels
    result = labels.clone() if isinstance(labels, torch.Tensor) else labels.copy()
    for source, destination in enumerate(mapping):
        result[labels == source] = destination
    return result

@torch.inference_mode()
def evaluate(model, loader, cfg, device, surface=False, reporting=False):
    model.eval()
    count = (len(cfg["reporting"]["class_names"]) if reporting and cfg.get("reporting")
             else cfg["model"]["num_classes"])
    meter = (SegmentationEvaluator if surface else SegmentationMeter)(count)
    for batch in loader:
        with autocast("cuda", enabled=cfg["training"]["amp"]):
            logits = model(batch["image"].to(device))["logits"]
        prediction, target = logits.argmax(dim=1), batch["mask"]
        if reporting:
            prediction, target = reporting_labels(prediction, cfg), reporting_labels(target, cfg)
        meter.update(prediction, target)
    return meter.compute()

def source_hashes():
    root = Path(__file__).resolve().parent.parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}

def train(cfg, device):
    paths = ensure_result_tree(cfg["output"]["dir"])
    if (paths["model"] / "last.pth").exists():
        raise FileExistsError("Experiment already contains a checkpoint; use a new experiment directory")
    loaders = training_loaders(cfg, paths["root"])
    student = UNet(cfg["model"]).to(device)
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    discriminator = SourceDiscriminator(
        cfg["model"]["channels"][cfg["model"]["feature_level"]],
        cfg["cmsir"]["discriminator_hidden"], cfg["cmsir"]["grl_scale"],
    ).to(device)
    params = list(student.parameters()) + list(discriminator.parameters())
    t = cfg["training"]
    optimizer = AdamW(params, lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scaler = GradScaler("cuda", enabled=t["amp"])
    labeled, unlabeled = InfiniteLoader(loaders["labeled"]), InfiniteLoader(loaders["unlabeled"])
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    total_timer = Timer()
    best_dice, best_epoch, stale, skipped = -1.0, 0, 0, 0
    history = []
    hashes = source_hashes()
    write_json(paths["root"] / "provenance.json", {
        "source_sha256": hashes, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(),
    })
    for epoch in range(1, t["epochs"] + 1):
        epoch_timer = Timer()
        student.train()
        discriminator.train()
        lr = learning_rate(epoch, t)
        for group in optimizer.param_groups:
            group["lr"] = lr
        progress = min(1.0, epoch / max(1, t["rampup_epochs"]))
        ramp = math.exp(-t["rampup_coefficient"] * (1 - progress) ** 2)
        sums = dict.fromkeys(["total", "sup", "mix", "source", "proto", "source_accuracy",
                              "prototype_layers", "pseudo_ratio", "alignment_abs_shift",
                              "alignment_applied_ratio", "alignment_score_gain"], 0.0)
        meter = SegmentationMeter(cfg["model"]["num_classes"])
        for _ in range(t["steps_per_epoch"]):
            lb, ub = labeled.next(), unlabeled.next()
            xl, yl = lb["image"].to(device), lb["mask"].to(device)
            xu, xw = ub["strong"].to(device), ub["weak"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=t["amp"]):
                with torch.no_grad():
                    confidence, pseudo = teacher(xw)["logits"].float().softmax(1).max(1)
                    if "valid" in ub:
                        valid = ub["valid"].to(device)
                        confidence = confidence.masked_fill(~valid, 0)
                        pseudo = pseudo.masked_fill(~valid, cfg["data"]["ignore_index"])
                mask = crossover_mask(xl.shape[0], xl.shape[-1], cfg["crossover"]["width_ratio"],
                                      device, generator)
                alignment_cfg = cfg.get('axial_alignment', {})
                mixed_valid = None
                alignment_stats = dict.fromkeys(['alignment_abs_shift', 'alignment_applied_ratio', 'alignment_score_gain'], 0.0)
                if alignment_cfg.get('enabled', False):
                    mixed, targets, weights, sources, mixed_valid, alignment_stats = aligned_mix_batch(
                        xl, xu, yl, pseudo, confidence, mask, t['pseudo_threshold'],
                        lb['alignment_image'].to(device), ub['alignment_image'].to(device),
                        lb['geometry_valid'].to(device), ub['geometry_valid'].to(device),
                        alignment_cfg, cfg['data']['ignore_index'])
                else:
                    mixed, targets, weights, sources = mix_batch(
                        xl, xu, yl, pseudo, confidence, mask, t["pseudo_threshold"])
                original = student(xl)
                outputs = student(mixed)
                sup = segmentation_loss(original["logits"], yl, ignore_index=cfg["data"]["ignore_index"])
                b = xl.shape[0]
                mix = (segmentation_loss(outputs["logits"][:b], targets[:b], weights[:b], cfg["data"]["ignore_index"])
                       + segmentation_loss(outputs["logits"][b:], targets[b:], weights[b:], cfg["data"]["ignore_index"])) / 2
                src, accuracy = source_loss(discriminator, outputs["feature"], sources, mixed_valid)
                proto, layers = prototype_loss(outputs["feature"], targets, weights, sources,
                                                cfg["cmsir"]["layer_ids"], cfg["data"]["ignore_index"])
                w = cfg["loss_weights"]
                total = sup + ramp * (w["mix"] * mix + w["source"] * src + w["prototype"] * proto)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, t["gradient_clip_norm"])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                ema_update(student, teacher, t["ema_decay"])
            else:
                skipped += 1
            meter.update(original["logits"].argmax(1), yl)
            values = {"total": total, "sup": sup, "mix": mix, "source": src, "proto": proto,
                      "source_accuracy": accuracy, "prototype_layers": layers,
                      "pseudo_ratio": (confidence >= t["pseudo_threshold"]).float().mean(), **alignment_stats}
            for key, value in values.items():
                sums[key] += float(value.detach()) if isinstance(value, torch.Tensor) else value
        averages = {key: value / t["steps_per_epoch"] for key, value in sums.items()}
        validation = evaluate(teacher, loaders["eval"], cfg, device)
        if not math.isfinite(validation["dice"]):
            raise FloatingPointError("Nonfinite validation Dice")
        improved = validation["dice"] > best_dice
        if improved:
            best_dice, best_epoch, stale = validation["dice"], epoch, 0
        else:
            stale += 1
        record = {"epoch": epoch, "learning_rate": lr, "ramp": ramp, **averages,
                  "train_dice": meter.compute()["dice"], "val_dice": validation["dice"],
                  "epoch_seconds": epoch_timer.elapsed(), "amp_skipped_steps": skipped}
        history.append(record)
        line = (
            f"Epoch: {epoch:03d} | Epoch Time: {format_duration(record['epoch_seconds'])} | LR: {lr:.4f}\n"
            f"\tTrain Loss: {averages['total']:.4f} - Dice: {record['train_dice']:.4f}\n"
            f"\tVal. Dice: {validation['dice']:.4f} - mIoU: {validation['miou']:.4f}\n"
            f"\tSup: {averages['sup']:.4f} - Mix: {averages['mix']:.4f} - Source: {averages['source']:.4f}"
            f" - Proto: {averages['proto']:.4f} - Ramp: {ramp:.4f}\n"
            f"\tSource Accuracy: {averages['source_accuracy']:.4f} - Prototype Layers: {averages['prototype_layers']:.4f}"
            f" - Pseudo Ratio: {averages['pseudo_ratio']:.4f} - AMP Skipped: {skipped}\n"
            f"\tAxial Shift (px): {averages['alignment_abs_shift']:.4f} - Applied: {averages['alignment_applied_ratio']:.4f}"
            f" - Correlation Gain: {averages['alignment_score_gain']:.4f}\n")
        print(line, end="", flush=True)
        with (paths["train"] / "train_log.txt").open("a", encoding="utf-8") as stream:
            stream.write(line)
        checkpoint = {"epoch": epoch, "student": student.state_dict(), "teacher": teacher.state_dict(),
                      "discriminator": discriminator.state_dict(), "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(), "config": cfg, "best_val_dice": best_dice,
                      "best_epoch": best_epoch, "source_sha256": hashes}
        torch.save(checkpoint, paths["model"] / "last.pth")
        if improved:
            torch.save(checkpoint, paths["model"] / "best.pth")
        write_json(paths["train"] / "history.json", history)
        if stale >= t["early_stopping_patience"]:
            break
    result = {"best_val_dice": best_dice, "best_checkpoint_epoch": best_epoch,
              "completed_epochs": epoch, "total_training_seconds": total_timer.elapsed(),
              "amp_skipped_steps": skipped, "early_stopped": epoch < t["epochs"]}
    write_json(paths["train"] / "metrics.json", result)
    (paths["train"] / "result.txt").write_text(
        f"Best Val Dice: {best_dice:.4f}\nBest Checkpoint Epoch: {best_epoch}\n"
        f"Completed Epochs: {epoch}\nTotal Training Time: {format_duration(result['total_training_seconds'])}\n"
        f"AMP Skipped Steps: {skipped}\n", encoding="utf-8")
    return result

@torch.inference_mode()
def test(cfg, device, checkpoint_path=None):
    cfg = copy.deepcopy(cfg)
    reporting_path = Path(cfg["output"]["dir"]) / "reporting.yaml"
    if reporting_path.exists():
        cfg["reporting"] = yaml.safe_load(reporting_path.read_text(encoding="utf-8"))
        mapping = cfg["reporting"]["label_map"]
        if len(mapping) != cfg["model"]["num_classes"] or set(mapping) != set(range(len(cfg["reporting"]["class_names"]))):
            raise ValueError("Invalid reporting label mapping")
    paths = ensure_result_tree(cfg["output"]["dir"])
    checkpoint_path = Path(checkpoint_path or paths["model"] / "best.pth").resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for section in ("model", "data"):
        if checkpoint["config"][section] != cfg[section]:
            raise ValueError(f"Test configuration differs from checkpoint: {section}")
    model = UNet(cfg["model"]).to(device).eval()
    model.load_state_dict(checkpoint["teacher"])
    files = discover(cfg["data"], cfg["data"]["test_split"])
    loader = make_loader(files, cfg, "test")
    timer = Timer()
    metrics = evaluate(model, loader, cfg, device, surface=True, reporting=True)
    metrics.update({"checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
                    "test_time_seconds": timer.elapsed(), "class_names": cfg.get("reporting", cfg["data"])["class_names"],
                    "evaluation_resolution": cfg["data"]["image_size"], "test_count": len(files),
                    "dice_aggregation": "dataset confusion matrix, macro foreground classes",
                    "surface_aggregation": "pool symmetric distances per class over images, then macro foreground",
                    "empty_surface_policy": "both absent: skip; one absent: image diagonal penalty"})
    if cfg.get("reporting"):
        metrics["reporting_protocol"] = cfg["reporting"]
        metrics["checkpoint_selection"] = "Original training validation macro Dice over classes 1-5; no reselection"
        validation_files = discover(cfg["data"], cfg["data"]["eval_split"])
        validation = evaluate(model, make_loader(validation_files, cfg, "eval"), cfg, device, reporting=True)
        validation.update(checkpoint=str(checkpoint_path), checkpoint_epoch=checkpoint["epoch"],
                          count=len(validation_files), reporting_protocol=cfg["reporting"],
                          checkpoint_selection=metrics["checkpoint_selection"])
        metrics["validation_dice_recomputed"] = validation["dice"]
        write_json(paths["train"] / "validation_recomputed.json", validation)
    write_json(paths["test"] / "metrics.json", metrics)
    lines = [f"Checkpoint: {checkpoint_path}", f"Checkpoint Epoch: {checkpoint['epoch']}",
             f"Test Time: {format_duration(metrics['test_time_seconds'])}"]
    for key in ["dice", "miou", "mpa", "hd95", "mASSD"]:
        lines.append(f"{key}: {metrics[key]:.4f}")
    for name, dice in zip(metrics["class_names"][1:], metrics["class_dice"][1:]):
        lines.append(f"{name} Dice: {dice:.4f}" if dice is not None else f"{name} Dice: NA")
    (paths["test"] / "test_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    palette = np.asarray(cfg["evaluation"]["palette"], dtype=np.uint8)
    if cfg.get("reporting"):
        palette = palette[:len(cfg["reporting"]["class_names"])]
        palette[0] = [0, 0, 0]
    saved = 0
    for batch in loader:
        with autocast("cuda", enabled=cfg["training"]["amp"]):
            predictions = reporting_labels(model(batch["image"].to(device))["logits"].argmax(1).cpu().numpy(), cfg)
        for i, name in enumerate(batch["name"]):
            gray = (batch["image"][i, 0].numpy() * 255).round().astype(np.uint8)
            target = reporting_labels(batch["mask"][i].numpy(), cfg)
            valid = target != cfg["data"]["ignore_index"]
            gt = palette[np.where(valid, target, 0)]
            gt[~valid] = cfg["evaluation"]["ignore_color"]
            panel = np.concatenate([np.repeat(gray[..., None], 3, axis=2), gt, palette[predictions[i]]], axis=1)
            Image.fromarray(panel).save(paths["visualization"] / (Path(name).stem + "_image_gt_pred.bmp"))
            saved += 1
            if saved >= cfg["evaluation"]["visualization_count"]:
                return metrics
    return metrics
