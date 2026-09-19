from pathlib import Path

import yaml

def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    output = cfg.setdefault("output", {}).get("dir")
    if output is None:
        cfg["output"]["dir"] = str(path.parent)
    elif Path(output).resolve() != path.parent:
        raise ValueError("Experiment YAML must be located in its output directory")
    data, model, training = cfg["data"], cfg["model"], cfg["training"]
    if len(data["mask_values"]) != model["num_classes"]:
        raise ValueError("Mask mapping and class count differ")
    if data["ignore_value"] in data["mask_values"]:
        raise ValueError("Ignore value overlaps a semantic class")
    if not 0 < data["labeled_fraction"] < 1:
        raise ValueError("Labeled fraction must lie in (0,1)")
    if not 0 <= model["feature_level"] < len(model["channels"]):
        raise ValueError("Invalid encoder feature level")
    if any(c % model["norm_groups"] for c in model["channels"]):
        raise ValueError("GroupNorm groups must divide every encoder width")
    if not all(0 < k < model["num_classes"] for k in cfg["cmsir"]["layer_ids"]):
        raise ValueError("Prototype classes must be foreground classes")
    if training["epochs"] < 1 or training["steps_per_epoch"] < 1:
        raise ValueError("Training duration must be positive")
    if not 0 < cfg["crossover"]["width_ratio"] <= 1:
        raise ValueError("Invalid crossover width ratio")
    if training["batch_size"] > min(data["expected_counts"][:2]):
        raise ValueError("Batch size exceeds training subset size")
    cfg["_config_path"] = str(path)
    alignment = cfg.get('axial_alignment', {})
    if alignment.get('enabled', False):
        if not (0 < alignment['max_shift_fraction'] < 0.5 and alignment['boundary_half_width'] > 0
                and 0 < alignment['smoothing_sigma'] < alignment['background_sigma']
                and 0 <= alignment['minimum_correlation'] <= 1
                and alignment['minimum_improvement'] >= 0 and alignment['minimum_energy'] > 0):
            raise ValueError('Invalid axial alignment parameters')
    return cfg
