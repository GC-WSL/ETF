import torch
import torch.nn.functional as F
import torch.nn as nn


def sigmoid_focal_loss(inputs, targets, num_masks, alpha=0.25, gamma=2):
    """
    Focal loss used for binary segmentation.
    
    Args:
        inputs: Tensor of shape [B*C, H*W], raw logits from the model.
        targets: Binary tensor of shape [B* C, H* W], ground truth masks (0 or 1).
        num_masks: Number of masks in the batch (used for normalization).
        alpha: Weighting factor to balance positive/negative classes.
        gamma: Modulating factor to focus on hard examples.

    Returns:
        Scalar focal loss.
    """
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(dim=1).sum() / num_masks


def dice_loss(inputs, targets, num_masks):
    """
    Dice loss for binary segmentation.

    Args:
        inputs: Tensor of shape [B*C, H*W], raw logits.
        targets: Binary tensor of shape [B*C, H*W].
        num_masks: Number of masks in the batch.

    Returns:
        Scalar dice loss.
    """
    inputs = torch.sigmoid(inputs).flatten(1)
    targets = targets.float().flatten(1)

    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)

    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def mask_losses(pred_masks, target_masks,num_mask=None, alpha=0.25, gamma=2):
    """
    Compute both focal loss and dice loss for binary segmentation.

    Args:
        pred_masks: Tensor of shape [B, C, H, W], predicted logits.
        target_masks: Tensor of shape [B, C, H, W], binary ground truth masks.
        num_masks: Total number of masks across the batch.
        alpha, gamma: Parameters for focal loss.

    Returns:
        Dict containing 'loss_mask' (focal loss) and 'loss_dice'.
    """
    # Ensure shapes match
    if pred_masks.shape != target_masks.shape:
        raise ValueError("pred_masks and target_masks must have the same shape")

    # Flatten batch and channel dimensions
    B, C, H, W = pred_masks.shape
    if num_mask is None:
        num_mask = (target_masks.sum(dim=(2,3))>0).sum()
    # import ipdb;ipdb.set_trace()
    pred_masks = pred_masks.reshape((B * C, H, W))
    target_masks = target_masks.reshape((B * C, H, W))

    # Resize predictions to match targets (if needed)
    if pred_masks.shape[-2:] != target_masks.shape[-2:]:
        pred_masks = F.interpolate(
            pred_masks.unsqueeze(1),
            size=target_masks.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    # Flatten spatial dimensions
    pred_masks = pred_masks.flatten(1)
    target_masks = target_masks.flatten(1)

    # Compute losses
    loss_mask = sigmoid_focal_loss(pred_masks, target_masks,num_mask, alpha=alpha, gamma=gamma)
    # loss_dice = dice_loss(pred_masks, target_masks,num_mask)
    
    return {
        "loss_mask": 20*loss_mask,
        # "loss_dice": 0.5*loss_dice
    }

def get_masked_ptc_loss(inputs, mask):
    """
    inputs:feature of [B,D,H,W]
    mask: pseudo label of [B,H,W]
    """
    b, c, h, w = inputs.shape
    
    inputs = inputs.reshape(b, c, h*w)

    def label_to_aff_mask(cam_label, ignore_index=255):
        if len(cam_label.shape) == 4:
            cam_label = cam_label.squeeze()
        b, h, w = cam_label.shape

        _cam_label = cam_label.reshape(b, 1, -1)
        _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
        _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1)
        aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)

        for i in range(b):
            aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
            aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index
        aff_label[:, range(h * w), range(h * w)] = ignore_index
        return aff_label
    mask = label_to_aff_mask(mask)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1,2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5*(1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum()+1)) + 0.5 * torch.sum(neg_mask * inputs_cos) / (neg_mask.sum()+1)
    return loss

if __name__ =="__main__":
    
    pred_masks = torch.randn(2, 3, 64, 64)   # [B, C, H, W]
    target_masks = torch.randint(0, 2, (2, 3, 64, 64))  # binary masks

    # 计算损失
    losses = mask_losses(pred_masks, target_masks)

    print(losses)
    # 输出类似：{'loss_mask': tensor(...), 'loss_dice': tensor(...)}