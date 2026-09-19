import torch
from torch.nn import functional as F

def segmentation_loss(logits, target, weight=None, ignore_index=-100):

    logits = logits.float()
    valid = target != ignore_index
    weight = valid.float() if weight is None else weight.float() * valid
    ce = F.cross_entropy(logits, target, reduction="none", ignore_index=ignore_index)
    ce = (ce * weight).sum() / weight.sum().clamp_min(1e-7)
    probability = logits.softmax(dim=1)[:, 1:]
    safe = target.masked_fill(~valid, 0)
    expected = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:].float()
    weights = weight.unsqueeze(1)
    intersection = (probability * expected * weights).sum(dim=(0, 2, 3))
    denominator = ((probability + expected) * weights).sum(dim=(0, 2, 3))
    dice = 1 - ((2 * intersection + 1e-7) / (denominator + 1e-7)).mean()
    return ce + dice

def crossover_mask(batch, width, ratio, device, generator=None):
    window = max(1, min(width, round(width * ratio)))
    starts = torch.randint(width - window + 1, (batch, 1, 1, 1),
                           device=device, generator=generator)
    columns = torch.arange(width, device=device).view(1, 1, 1, width)
    return (columns >= starts) & (columns < starts + window)

def mix_batch(image_l, image_u, target_l, pseudo, confidence, mask, threshold):
    image_lu = torch.where(mask, image_u, image_l)
    image_ul = torch.where(mask, image_l, image_u)
    source_lu = mask[:, 0].expand_as(target_l)
    source_ul = ~source_lu
    pseudo_weight = confidence * (confidence >= threshold)
    targets = torch.cat([torch.where(source_lu, pseudo, target_l),
                         torch.where(source_ul, pseudo, target_l)])
    weights = torch.cat([torch.where(source_lu, pseudo_weight, 1.0),
                         torch.where(source_ul, pseudo_weight, 1.0)])
    sources = torch.cat([source_lu, source_ul])
    return torch.cat([image_lu, image_ul]), targets, weights, sources

def feature_targets(targets, weights, sources, size):
    def resize(tensor):
        return F.interpolate(tensor[:, None].float(), size=size, mode="nearest")[:, 0]
    return resize(targets).long(), resize(weights), resize(sources).bool()

def source_loss(discriminator, features, sources, valid=None):
    spatial_valid = (F.interpolate(valid[:, None].float(), size=features.shape[-2:], mode='nearest')
                     if valid is not None else None)
    logits = discriminator(features, spatial_valid).float()
    labels = F.interpolate(sources[:, :1].float(), size=features.shape[-1],
                           mode="nearest")[:, 0].long()
    column_valid = spatial_valid.sum(2)[:, 0] > 0 if valid is not None else torch.ones_like(labels, dtype=torch.bool)
    loss = (F.cross_entropy(logits, labels, reduction='none') * column_valid).sum() / column_valid.sum().clamp_min(1)
    accuracy = ((logits.argmax(dim=1) == labels) * column_valid).float().sum() / column_valid.sum().clamp_min(1)
    return loss, accuracy.detach()

def prototype_loss(features, targets, weights, sources, layer_ids, ignore_index=-100):
    targets, weights, sources = feature_targets(targets, weights, sources, features.shape[-2:])
    features = features.float().permute(0, 2, 3, 1)
    valid = (targets != ignore_index) & (weights > 0)
    losses = []
    for layer in layer_ids:
        selected = valid & (targets == layer)
        labeled = selected & ~sources
        unlabeled = selected & sources
        if labeled.any() and unlabeled.any():
            pl = features[labeled].mean(dim=0)
            pu = features[unlabeled].mean(dim=0)
            losses.append(1 - F.cosine_similarity(pl[None], pu[None], dim=1)[0])
    loss = torch.stack(losses).mean() if losses else features.sum() * 0
    return loss, len(losses)
