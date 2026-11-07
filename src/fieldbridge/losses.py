import torch
import torch.distributed as dist
from torch.nn import functional as F


def masked_mean(value, mask):
    mask = mask.to(value.dtype).expand_as(value)
    denominator = mask.sum().detach()
    world = 1
    if dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(denominator)
    # DDP averages gradients: compensate for globally unequal missing-label counts.
    return (value * mask).sum() * world / denominator.clamp_min(1)


def balanced_contact_bce(logits, target, mask):
    """Equal positive/negative region weight across ranks; absent classes omitted."""
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    positive = mask.to(loss.dtype) * (target > .5)
    negative = mask.to(loss.dtype) * (target <= .5)
    counts = torch.stack((positive.sum(), negative.sum())).detach()
    world = 1
    if dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(counts)
    class_means = torch.stack(((loss * positive).sum(), (loss * negative).sum())) * world / counts.clamp_min(1)
    present = counts > 0
    return (class_means * present).sum() / present.sum().clamp_min(1)


def contact_bce_dice(logits, target, mask):
    """Native-prevalence BCE plus per-observation Dice; padded masks excluded.

    Per-frame Dice is globally averaged with the same DDP count correction as
    other losses. Empty targets use BCE only, avoiding unstable empty Dice.
    """
    logits, target, mask = logits.float(), target.float(), mask.float()
    bce = masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction='none'), mask)
    prediction = logits.sigmoid()
    intersection = (prediction * target * mask).sum((-3, -2, -1))
    support = (target * mask).sum((-3, -2, -1))
    denominator = ((prediction + target) * mask).sum((-3, -2, -1))
    dice = 1 - (2*intersection + 1e-6) / (denominator + 1e-6)
    return bce + masked_mean(dice, support > 0)


def compute_loss(output, batch, cfg, field_only=False):
    mu, lv = output["mean"].float(), output["logvar"].float()
    y, m = batch["target"], batch["target_mask"]
    err = mu - y
    parameter = masked_mean(F.smooth_l1_loss(mu, y, reduction="none"), m)
    nll = masked_mean(.5 * (err.square() * (-lv).exp() + lv), m)
    law_mask = batch["law"] >= 0
    law = masked_mean(F.cross_entropy(output["law_logits"].float(), batch["law"].clamp_min(0), reduction="none"), law_mask)
    fields, fm = batch["fields"], batch["field_mask"]
    contact_target, contact_mask = fields[..., :1, :, :], fm[..., :1, :, :]
    if cfg.train.contact_loss == "balanced_bce":
        contact = balanced_contact_bce(output["contact_logits"], contact_target, contact_mask)
    elif cfg.train.contact_loss == "bce_dice":
        contact = contact_bce_dice(output["contact_logits"], contact_target, contact_mask)
    else:
        contact = masked_mean(F.binary_cross_entropy_with_logits(output["contact_logits"].float(), contact_target, reduction="none"), contact_mask)
    flow_c = 3 if cfg.model.field_mode == "legacy_visual" else 2
    flow = masked_mean(F.smooth_l1_loss(output["fields"][..., 1:1+flow_c, :, :].float(), fields[..., 1:1+flow_c, :, :], reduction="none"), fm[..., 1:1+flow_c, :, :])
    stress = masked_mean(F.smooth_l1_loss(output["fields"][..., 1+flow_c:, :, :].float(), fields[..., 1+flow_c:, :, :], reduction="none"), fm[..., 1+flow_c:, :, :])
    # No empirical stress-speed/contact correlation is treated as a physical law.
    supervised = parameter + cfg.train.nll_weight * nll + cfg.train.law_weight * law
    total = (0. if field_only else cfg.train.parameter_weight) * supervised + cfg.train.field_weight * (contact + flow + stress)
    return total, {"parameter": parameter.detach(), "nll": nll.detach(), "law": law.detach(),
                   "contact": contact.detach(), "flow": flow.detach(), "stress": stress.detach()}
