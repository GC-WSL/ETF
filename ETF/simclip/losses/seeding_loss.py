import torch
import torch.nn.functional as F

def calculate_seeding_loss(pred, label, bg=True, bg_idxs=None,bg_fg_weight=(1,1),
                      noisy_weights=None, label_smooth=False):
    device = pred.device
    B, C, H, W = pred.shape  

    if label_smooth:
        eps = 0.01  
        label = (1 - eps) * label.float() + eps / C  

    if noisy_weights is None:
        noisy_weights = torch.ones((B, 1, H, W), device=device)
    else:
        noisy_weights = torch.nn.functional.interpolate(
            noisy_weights.float(), size=(H, W), mode='nearest'
        )

    eps_count = 1e-4  # Minimum denominator value

    if not bg:
        # All as foreground
        fg_mask = label
        pred_fg = pred
        fg_term = -(noisy_weights * fg_mask * torch.log(pred_fg.clamp(min=1e-8)))
        fg_count = fg_mask.sum(dim=(1, 2, 3), keepdim=True)
        total_loss = (fg_term.sum(dim=(1, 2, 3), keepdim=True) / fg_count.clamp(min=eps_count)).mean()
        return total_loss
    else:
        # Handle background classes
        if bg_idxs is None:
            bg_idxs = [0]
        elif isinstance(bg_idxs, int):
            bg_idxs = [bg_idxs]
        # Validate bg_idxs
        for idx in bg_idxs:
            if idx < 0 or idx >= C:
                raise ValueError(f"bg_idx must be in [0, {C-1}], but got {idx}")
        
        total_loss = 0.0
        for bg_idx in bg_idxs:
            # Split background and foreground
            bg_mask = label[:, bg_idx:bg_idx+1]  # [B, 1, H, W]
            pred_bg = pred[:, bg_idx:bg_idx+1]   # [B, 1, H, W]
            
            # Foreground is all channels except bg_idx
            fg_mask = torch.cat([label[:, :bg_idx], label[:, bg_idx+1:]], dim=1)
            pred_fg = torch.cat([pred[:, :bg_idx], pred[:, bg_idx+1:]], dim=1)
            
            # Background loss
            bg_term = -noisy_weights *( bg_mask * torch.log(pred_bg.clamp(min=eps_count)))
            # bg_count = bg_mask.sum(dim=(1, 2, 3), keepdim=True)
            bg_loss = bg_term.mean()
            
            # Foreground loss
            fg_term = -(noisy_weights * fg_mask * torch.log(pred_fg.clamp(min=1e-8)))
            fg_count = fg_mask.sum(dim=(1, 2, 3), keepdim=True)
            fg_loss = (fg_term.sum(dim=(1, 2, 3), keepdim=True) / fg_count.clamp(min=eps_count)).mean()
            # fg_loss = fg_term.mean(dim=(2,3)).sum()
            
            total_loss += (bg_fg_weight[0]*bg_loss + bg_fg_weight[1]*fg_loss)
            # bg_fg_loss = torch.cat([bg_fg_weight[0]*bg_term,bg_fg_weight[1]*fg_term.sum(dim=1,keepdim=True)],dim=1)
            # bg_fg_var = adaptive_weighted_pool(bg_fg_loss,decay_parameter=0.999,descending=False)
            
            # total_loss+= bg_fg_loss.mean()
        
        # Average loss over all specified bg_idxs
        total_loss /= len(bg_idxs)
        return total_loss
    
def get_seg_loss(pred, label, ignore_index=255):
    
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5

def adaptive_weighted_pool(x_patch_cls,decay_parameter=0.995,descending=True):
    """
    x_patch_cls:[B,C,H,W]
    """
    x_patch_flattened = x_patch_cls.view(x_patch_cls.shape[0], x_patch_cls.shape[1], -1).permute(0, 2, 1)

    sorted_patch_token, indices = torch.sort(x_patch_flattened, dim=-2, descending=descending)

    # 构建指数衰减权重
    weights = torch.logspace(
        start=0,
        end=x_patch_flattened.size(-2) - 1,
        steps=x_patch_flattened.size(-2),
        base=decay_parameter,
    ).to(x_patch_cls.device)

    # 加权平均得到最终 patch_logits
    x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()

    return x_patch_logits