import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

def discover(cfg, split):
    root = Path(cfg["root"]) / split
    files = sorted((root / cfg["image_dir"]).glob("*" + cfg["extension"]))
    if not files:
        raise ValueError(f"No images in {root}")
    for path in files:
        if not (root / cfg["mask_dir"] / path.name).is_file():
            raise FileNotFoundError(path.name)
    return files

def group_name(path):
    return path.stem.removesuffix("_flip")

def split_training(files, count, seed):
    groups = sorted({group_name(p) for p in files})
    if not 0 < count < len(groups):
        raise ValueError("Invalid labeled group count")
    random.Random(seed).shuffle(groups)
    selected = set(groups[:count])
    return ([p for p in files if group_name(p) in selected],
            [p for p in files if group_name(p) not in selected])

def letterbox(image, size, mask=False, padding=0):
    width, height = image.size
    scale = min(size / width, size / height)
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(shape, Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR)
    canvas = Image.new("L", (size, size), padding)
    canvas.paste(resized, ((size - shape[0]) // 2, (size - shape[1]) // 2))
    return canvas

def photometric(image, cfg, strong):
    if not strong:
        return (image + torch.randn_like(image) * cfg["weak_noise_std"]).clamp(0, 1)
    gamma = torch.empty(()).uniform_(*cfg["gamma_range"]).item()
    contrast = torch.empty(()).uniform_(*cfg["contrast_range"]).item()
    noise = torch.empty(()).uniform_(*cfg["strong_noise_range"]).item()
    mean = image.mean()
    image = (image.clamp_min(1e-6).pow(gamma) - mean) * contrast + mean
    image = image + torch.randn_like(image) * noise
    if torch.rand(()).item() < cfg["blur_probability"]:
        kernel = cfg["blur_kernel"]
        image = F.avg_pool2d(image[None], kernel, stride=1, padding=kernel // 2)[0]
    return image.clamp(0, 1)

class OCTDataset(Dataset):
    def __init__(self, files, cfg, augmentation, mode):
        self.files, self.cfg, self.augmentation, self.mode = files, cfg, augmentation, mode
        self.lookup = np.full(256, -9999, dtype=np.int64)
        for index, value in enumerate(cfg["mask_values"]):
            self.lookup[value] = index
        if cfg["ignore_value"] is not None:
            self.lookup[cfg["ignore_value"]] = cfg["ignore_index"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as raw:
            original_size = raw.size
            image = np.array(letterbox(raw.convert("L"), self.cfg["image_size"]), dtype=np.float32)
        image = torch.from_numpy(image / 255)[None]
        alignment_sample = {}
        if self.mode in {'labeled', 'unlabeled'}:
            valid = letterbox(Image.new('L', original_size, 1), self.cfg['image_size'], True, 0)
            alignment_sample = {'alignment_image': image.clone(),
                                'geometry_valid': torch.from_numpy(np.array(valid, dtype=bool))}
        if self.mode == "unlabeled":

            sample = {"weak": photometric(image, self.augmentation, False),
                      "strong": photometric(image, self.augmentation, True), "name": path.name}
            sample.update(alignment_sample)
            if self.cfg["ignore_value"] is not None:

                valid = letterbox(Image.new("L", original_size, 1), self.cfg["image_size"], True, 0)
                sample["valid"] = torch.from_numpy(np.array(valid, dtype=bool))
            return sample
        with Image.open(path.parent.parent / self.cfg["mask_dir"] / path.name) as raw:
            mask = np.array(letterbox(raw.convert("L"), self.cfg["image_size"], True,
                                      self.cfg["mask_padding_value"]))
        mask = self.lookup[mask]
        if (mask == -9999).any():
            raise ValueError(f"Unknown mask intensity: {path.name}")
        if self.mode == "labeled":
            image = photometric(image, self.augmentation, True)
        return {"image": image, "mask": torch.from_numpy(mask), "name": path.name, **alignment_sample}

def make_loader(files, cfg, mode):
    training = mode in {"labeled", "unlabeled"}
    generator = torch.Generator().manual_seed(cfg["seed"] + int(mode == "unlabeled"))
    return DataLoader(
        OCTDataset(files, cfg["data"], cfg["augmentation"], mode),
        batch_size=cfg["training" if training else "evaluation"]["batch_size"],
        shuffle=training, num_workers=cfg["data"]["num_workers"],
        pin_memory=True, drop_last=training, generator=generator,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )

def select_by_name(files, names):
    index = {path.name: path for path in files}
    missing = [name for name in names if name not in index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} file(s) from the frozen split are missing, "
                                f"first: {missing[0]}")
    return [index[name] for name in names]

def training_loaders(cfg, output):
    data = cfg["data"]
    files = discover(data, data["train_split"])
    reference = data.get("split_manifest")
    external = data.get("external_unlabeled_dir")
    if reference:
        source = Path(reference)
        if not source.is_absolute():
            source = Path(cfg["_config_path"]).parent / source
        frozen = json.loads(source.read_text(encoding="utf-8"))
        labeled = select_by_name(files, frozen["labeled"])
        unlabeled = (sorted(Path(external).glob("*" + data["extension"])) if external
                     else select_by_name(files, frozen["unlabeled"]))
        remainder = [p for p in files if p not in set(labeled)]
    else:
        if "labeled_group_count" not in data or "split_seed" not in data:
            raise ValueError("Set data.split_manifest, or data.labeled_group_count with data.split_seed")
        labeled, remainder = split_training(files, data["labeled_group_count"], data["split_seed"])
        unlabeled = (sorted(Path(external).glob("*" + data["extension"])) if external
                     else remainder)
    if not unlabeled:
        raise ValueError("The unlabeled pool is empty")
    val = discover(data, data["eval_split"])
    test_files = discover(data, data["test_split"])
    if (len(labeled), len(unlabeled), len(val), len(test_files)) != tuple(data["expected_counts"]):
        raise ValueError("Dataset counts do not match configuration")
    sets = [{group_name(p) for p in part} for part in (files, val, test_files)]
    collisions = {f"{a}/{b}": sorted(sets[i] & sets[j])
                  for i, a in enumerate(("train", "eval", "test"))
                  for j, b in enumerate(("train", "eval", "test")) if i < j}

    fingerprints = {}
    for split, part in zip(("train", "eval", "test"), (files, val, test_files)):
        for path in part:
            with Image.open(path) as raw:
                pixels = np.array(raw.convert("L"))
            digest = min(hashlib.sha256(pixels.tobytes()).hexdigest(),
                         hashlib.sha256(pixels[:, ::-1].tobytes()).hexdigest())
            key = (pixels.shape, digest)
            if key in fingerprints and fingerprints[key][0] != split:
                raise ValueError(f"Duplicate image across splits: {fingerprints[key]}, {split}/{path.name}")
            fingerprints[key] = (split, path.name)
    manifest = {
        "seed": cfg["seed"],
        "requested_labeled_fraction": data["labeled_fraction"],
        "effective_labeled_fraction": len(labeled) / len(files),
        "split_source": reference or "seed",
        "external_unlabeled_dir": external,
        "labeled": [p.name for p in labeled], "unlabeled": [p.name for p in unlabeled],
        "eval": [p.name for p in val], "test": [p.name for p in test_files],
        "cross_split_name_collisions": collisions,
        "cross_split_identical_or_horizontally_flipped_images": 0,
        "subject_independence": "not established by the supplied fixed split metadata",
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"labeled": make_loader(labeled, cfg, "labeled"),
            "unlabeled": make_loader(unlabeled, cfg, "unlabeled"),
            "eval": make_loader(val, cfg, "eval")}
