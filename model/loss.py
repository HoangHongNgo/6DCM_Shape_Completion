import torch
import torch.nn.functional as F
from extensions.chamfer_distance.chamfer_distance import ChamferDistance


def CD_Loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = ChamferDistance(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    dist1 = torch.mean(dist1)
    dist2 = torch.mean(dist2)
    res = (dist1 + dist2) / 2.0

    return res


def smooth_l1_loss(vertex_pred, vertex_targets, mask, sigma=10., normalize=True, reduction=True):
    '''
    :param reduction:
    :param vertex_pred:     [b,k,h,w]
    :param vertex_targets:  [b,k,h,w]
    :param mask:  [b,1,h,w]
    :param sigma:
    :param normalize:
    :param reduce:
    :return:
    '''
    # sigma = 1.0
    # sigma = 5.0
    b, ver_dim, _, _ = vertex_pred.shape
    # sigma_2 = sigma ** 2

    abs_diff = abs(mask * (vertex_pred - vertex_targets))

    smoothL1_sign = (abs_diff < 1. / sigma).detach().float()
    in_loss = abs_diff ** 2 * (sigma / 2.) * smoothL1_sign + \
        (abs_diff - (0.5 / sigma)) * (1. - smoothL1_sign)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / \
            (2. * torch.sum(mask.view(b, -1), 1) + 1e-9)

    if reduction:
        in_loss = torch.mean(in_loss)

    return in_loss


def focal_loss(input, target, alpha=None, gamma=2, reduction='mean'):
    """
    Focal Loss for binary or multi-class classification tasks.

    Args:
    - input (torch.Tensor): Model predictions, shape [batch_size, num_classes]
    - target (torch.Tensor): Ground truth labels, shape [batch_size]
    - alpha (torch.Tensor or None): Class balancing weights, shape [num_classes] or None. Default is None (no class weighting)
    - gamma (float): Focusing parameter for down-weighting easy examples. Default is 2.
    - reduction (str): Specifies the reduction to apply to the output. Default is 'mean'.
                        Options: 'none', 'mean', 'sum'.

    Returns:
    - loss (torch.Tensor): Computed focal loss.
    """
    # Convert target to long tensor for compatibility with cross_entropy
    target = target.long()

    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(input, target, weight=alpha, reduction='none')

    # Compute focal loss
    focal_loss = ((1 - torch.exp(-ce_loss)) ** gamma) * ce_loss

    # Apply reduction if specified
    if reduction == 'mean':
        return torch.mean(focal_loss)
    elif reduction == 'sum':
        return torch.sum(focal_loss)
    else:
        return focal_loss
