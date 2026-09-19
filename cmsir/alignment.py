import torch
from torch.nn import functional as F

def translate_y(value, shifts, fill=0):
    squeeze = value.ndim == 3
    x = value[:, None] if squeeze else value
    b, c, h, w = x.shape
    rows = torch.arange(h, device=x.device)[None, :] - shifts[:, None]
    valid = (rows >= 0) & (rows < h)
    index = rows.clamp(0, h-1)[:, None, :, None].expand(b, c, h, w)
    moved = x.gather(2, index).masked_fill(~valid[:, None, :, None], fill)
    return moved[:, 0] if squeeze else moved

def _smooth(profiles, sigma):
    radius = max(1, round(3 * sigma))
    x = torch.arange(-radius, radius+1, device=profiles.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma).square())
    kernel = (kernel / kernel.sum())[None, None]
    shape = profiles.shape
    values = profiles.reshape(-1, 1, shape[-1])
    return F.conv1d(F.pad(values, (radius, radius), mode='replicate'), kernel).reshape(shape)

@torch.no_grad()
def estimate_axial_shift(image_l, image_u, mask, cfg):
    with torch.autocast(device_type=image_l.device.type, enabled=False):
        image_l, image_u = image_l.float(), image_u.float()
        b, _, h, w = image_l.shape
        columns = torch.arange(w, device=image_l.device)
        selected = mask[:, 0, 0]
        starts = torch.where(selected, columns, w).amin(-1)
        ends = torch.where(selected, columns+1, 0).amax(-1)
        boundaries = torch.stack([starts, ends], dim=1)
        existing = (boundaries > 0) & (boundaries < w)
        bands = ((columns[None, None] >= boundaries[..., None]-cfg['boundary_half_width'])
                 & (columns[None, None] < boundaries[..., None]+cfg['boundary_half_width']))
        bands = bands.float() / bands.sum(-1, keepdim=True).clamp_min(1)
        def profiles(image):
            p = torch.einsum('bhw,bkw->bkh', image[:, 0], bands)
            return (_smooth(p, cfg['smoothing_sigma']) - _smooth(p, cfg['background_sigma'])).clamp_min(0)
        reference, donor = profiles(image_l), profiles(image_u)
        limit = min(round(h * cfg['max_shift_fraction']), h-1)

        deltas = torch.tensor([0] + [d for n in range(1, limit+1) for d in (-n, n)], device=image_l.device)
        rows = torch.arange(h, device=image_l.device)[None] - deltas[:, None]
        overlap = (rows >= 0) & (rows < h)
        shifted = donor[:, :, rows.clamp(0, h-1)] * overlap[None, None]
        r = reference[:, :, None] * overlap[None, None]
        energy = r.square().sum(-1).sqrt() * shifted.square().sum(-1).sqrt()
        correlation = (r * shifted).sum(-1) / energy.clamp_min(1e-12)
        usable = existing & (reference.square().sum(-1) > cfg['minimum_energy']) & (donor.square().sum(-1) > cfg['minimum_energy'])
        scores = (correlation * usable[..., None]).sum(1) / usable.sum(1)[:, None].clamp_min(1)
        best_score, best = scores.max(-1)
        gain = best_score - scores[:, 0]
        accepted = usable.any(-1) & (best_score >= cfg['minimum_correlation']) & (gain >= cfg['minimum_improvement'])
        shifts = torch.where(accepted, deltas[best], 0)
        return shifts, torch.where(accepted, gain, 0)

def aligned_mix_batch(image_l, image_u, target_l, pseudo, confidence, mask, threshold,
                      raw_l, raw_u, valid_l, valid_u, cfg, ignore_index=-100):
    shifts, gain = estimate_axial_shift(raw_l, raw_u, mask, cfg)
    valid_l = valid_l & (target_l != ignore_index)
    valid_u = valid_u & (pseudo != ignore_index)
    weight_l = valid_l.float()
    weight_u = confidence * (confidence >= threshold) * valid_u
    strips = mask[:, 0].expand_as(target_l)
    images, targets, weights, validity = [], [], [], []
    for recipient, donor, yr, yd, wr, wd, vr, vd, delta in [
        (image_l, image_u, target_l, pseudo, weight_l, weight_u, valid_l, valid_u, shifts),
        (image_u, image_l, pseudo, target_l, weight_u, weight_l, valid_u, valid_l, -shifts),
    ]:
        valid = torch.where(strips, translate_y(vd, delta, False), vr)
        target = torch.where(strips, translate_y(yd, delta, ignore_index), yr).masked_fill(~valid, ignore_index)
        weight = torch.where(strips, translate_y(wd, delta, 0), wr) * valid
        images.append(torch.where(mask, translate_y(donor, delta, cfg['image_fill']), recipient))
        targets.append(target)
        weights.append(weight)
        validity.append(valid)
    diagnostics = {'alignment_abs_shift': shifts.float().abs().mean(),
                   'alignment_applied_ratio': (shifts != 0).float().mean(),
                   'alignment_score_gain': gain.mean()}
    return (torch.cat(images), torch.cat(targets), torch.cat(weights),
            torch.cat([strips, ~strips]), torch.cat(validity), diagnostics)
