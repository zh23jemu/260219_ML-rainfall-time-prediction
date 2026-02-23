import torch
import torch.nn.functional as F

def _reduce_loss(loss, sample_weight=None):
    if sample_weight is None:
        return torch.mean(loss)
    w = sample_weight.to(loss.dtype)
    return torch.sum(loss * w) / (torch.sum(w) + 1e-8)

def pinball_loss(y_true, y_pred, tau: float, sample_weight=None):
    # y_true,y_pred: [..., H]
    u = y_true - y_pred
    loss = torch.maximum(tau * u, (tau - 1) * u)
    return _reduce_loss(loss, sample_weight=sample_weight)

def huber_pinball_loss(y_true, y_pred, tau: float, delta: float = 1.0, sample_weight=None):
    # Huberized pinball (smooth)
    u = y_true - y_pred
    abs_u = torch.abs(u)
    huber = torch.where(abs_u <= delta, 0.5 * (u**2) / delta, abs_u - 0.5 * delta)
    # pinball weighting
    weight = torch.where(u >= 0, tau, 1 - tau)
    loss = weight * huber
    return _reduce_loss(loss, sample_weight=sample_weight)

def quantile_loss(y_true, y_pred_q, quantiles, use_huber=False, delta=1.0, sample_weight=None):
    # y_pred_q: [B, H, Q]
    loss = 0.0
    for qi, tau in enumerate(quantiles):
        pred = y_pred_q[..., qi]
        if use_huber:
            loss = loss + huber_pinball_loss(y_true, pred, float(tau), delta, sample_weight=sample_weight)
        else:
            loss = loss + pinball_loss(y_true, pred, float(tau), sample_weight=sample_weight)
    return loss / len(quantiles)

def domain_ce_loss(domain_logits, domain_labels):
    # logits: [B, Kd], labels: [B]
    return F.cross_entropy(domain_logits, domain_labels)

def mse_loss(y_true, y_pred, sample_weight=None):
    loss = (y_true - y_pred) ** 2
    return _reduce_loss(loss, sample_weight=sample_weight)
