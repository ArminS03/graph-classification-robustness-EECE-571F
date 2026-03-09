import torch.nn.functional as F


def jensen_shannon_divergence_loss(logits_orig, logits_mixed):
    """Compute the Jensen-Shannon Divergence between the original and mixed predictions.

    This follows the AugMix consistency loss: JSD(p_orig || p_mixed), computed as
    the average KL divergence of each distribution from their midpoint M.

    Args:
        logits_orig: Logits from the clean (original) forward pass.
        logits_mixed: Logits from the mixed augmented forward pass.

    Returns:
        Scalar JSD loss.
    """
    p_orig = F.softmax(logits_orig, dim=1)
    p_mixed = F.softmax(logits_mixed, dim=1)

    m = 0.5 * (p_orig + p_mixed)

    loss = 0.5 * (
        F.kl_div(F.log_softmax(logits_orig, dim=1), m, reduction="batchmean") +
        F.kl_div(F.log_softmax(logits_mixed, dim=1), m, reduction="batchmean")
    )

    return loss