from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

class SegmentationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().cpu().long().reshape(-1)
        target = target.detach().cpu().long().reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        indices = self.num_classes * target[valid] + prediction[valid]
        self.confusion += torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        true_positive = self.confusion.diag()
        target_count = self.confusion.sum(dim=1)
        prediction_count = self.confusion.sum(dim=0)
        dice_denominator = target_count + prediction_count
        union = target_count + prediction_count - true_positive
        class_dice = torch.where(
            dice_denominator > 0,
            2.0 * true_positive / dice_denominator,
            torch.full_like(dice_denominator, torch.nan),
        )
        class_iou = torch.where(
            union > 0,
            true_positive / union,
            torch.full_like(union, torch.nan),
        )
        class_accuracy = torch.where(
            target_count > 0,
            true_positive / target_count,
            torch.full_like(target_count, torch.nan),
        )
        foreground = slice(1, None)
        return {
            "dice": float(torch.nanmean(class_dice[foreground])),
            "miou": float(torch.nanmean(class_iou[foreground])),
            "mpa": float(torch.nanmean(class_accuracy[foreground])),
            "class_dice": _finite_list(class_dice),
            "class_iou": _finite_list(class_iou),
        }

def _finite_list(values: torch.Tensor) -> list[float | None]:
    array = values.cpu().numpy()
    return [float(value) if np.isfinite(value) else None for value in array]

def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)

def _symmetric_surface_distances(
    prediction: np.ndarray, target: np.ndarray
) -> np.ndarray:
    prediction_surface = _surface(prediction)
    target_surface = _surface(target)
    if not prediction_surface.any() and not target_surface.any():
        return np.empty((0,), dtype=np.float64)
    if not prediction_surface.any() or not target_surface.any():
        return np.asarray([math.hypot(*prediction.shape)], dtype=np.float64)
    distance_to_target = distance_transform_edt(~target_surface)
    distance_to_prediction = distance_transform_edt(~prediction_surface)
    return np.concatenate(
        (distance_to_target[prediction_surface], distance_to_prediction[target_surface])
    ).astype(np.float64, copy=False)

class SegmentationEvaluator(SegmentationMeter):

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)
        self.surface_distances: list[list[np.ndarray]] = [
            [] for _ in range(num_classes)
        ]

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        super().update(prediction, target)
        prediction_array = prediction.detach().cpu().numpy()
        target_array = target.detach().cpu().numpy()
        for sample_prediction, sample_target in zip(prediction_array, target_array):
            valid = (sample_target >= 0) & (sample_target < self.num_classes)
            for class_index in range(1, self.num_classes):
                class_prediction = (sample_prediction == class_index) & valid
                class_target = (sample_target == class_index) & valid
                if not class_prediction.any() and not class_target.any():
                    continue
                distances = _symmetric_surface_distances(class_prediction, class_target)
                if distances.size:
                    self.surface_distances[class_index].append(distances)

    def compute(self) -> dict[str, Any]:
        metrics = super().compute()
        class_hd95: list[float | None] = [None] * self.num_classes
        class_assd: list[float | None] = [None] * self.num_classes
        for class_index in range(1, self.num_classes):
            values = self.surface_distances[class_index]
            if not values:
                continue
            distances = np.concatenate(values)
            class_hd95[class_index] = float(np.percentile(distances, 95))
            class_assd[class_index] = float(distances.mean())
        foreground_hd95 = [value for value in class_hd95[1:] if value is not None]
        foreground_assd = [value for value in class_assd[1:] if value is not None]
        metrics.update(
            {
                "hd95": float(np.mean(foreground_hd95)) if foreground_hd95 else None,
                "mASSD": float(np.mean(foreground_assd)) if foreground_assd else None,
                "class_hd95": class_hd95,
                "class_assd": class_assd,
                "surface_distance_unit": "pixel_at_evaluation_resolution",
            }
        )
        return metrics
