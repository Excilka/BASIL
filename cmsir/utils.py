from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)

def require_cuda(device_index: int, minimum_free_memory_mib: int) -> torch.device:
    report = ""
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            report = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            break
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(2.0)
    if not report:
        raise RuntimeError("nvidia-smi failed three consecutive times before training") from last_error
    print("GPU status before execution:\n" + report, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a GPU, but PyTorch reports no CUDA device")
    if not 0 <= device_index < torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_index} does not exist; "
            f"{torch.cuda.device_count()} device(s) visible"
        )
    free_bytes, _ = torch.cuda.mem_get_info(device_index)
    free_mib = free_bytes / (1024**2)
    if free_mib < minimum_free_memory_mib:
        raise RuntimeError(
            f"GPU {device_index} has {free_mib:.0f} MiB free, below the configured "
            f"minimum of {minimum_free_memory_mib} MiB"
        )
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")

def configure_threads(thread_count: int) -> None:
    value = str(thread_count)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = value
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(1)

def ensure_result_tree(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).resolve()
    paths = {
        "root": root,
        "train": root / "train",
        "model": root / "train" / "model",
        "test": root / "test",
        "visualization": root / "test" / "Visualization",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"

def sigmoid_rampup(epoch: int, start_epoch: int, rampup_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if rampup_epochs <= 0:
        return 1.0
    progress = min(1.0, (epoch - start_epoch + 1) / rampup_epochs)
    return math.exp(-5.0 * (1.0 - progress) ** 2)

def ema_update(
    student: torch.nn.Module, teacher: torch.nn.Module, decay: float
) -> None:
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters()
        ):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)

class InfiniteLoader:
    def __init__(self, loader: torch.utils.data.DataLoader[Any]) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def next(self) -> Any:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
