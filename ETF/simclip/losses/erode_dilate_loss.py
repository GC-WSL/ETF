import torch
import cv2
import numpy as np
import torch.nn.functional as F
from simclip.losses.seeding_loss import calculate_seeding_loss

def morphological_operation_multi_class(mask, kernel_size=5, op="erode"):

    device = mask.device
    B, K, H, W = mask.shape
    
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=mask.dtype)

    mask_reshaped = mask.view(B * K, 1, H, W)
    
    if op == "erode":

        conv_result = F.conv2d(mask_reshaped, kernel, padding=kernel_size//2)
        output = (conv_result == kernel.sum()).float()
        
    elif op == "dilate":
        conv_result = F.conv2d(mask_reshaped, kernel, padding=kernel_size//2)
        output = (conv_result > 0).float()
    
    return output.view(B, K, H, W)

def erode_dilate_loss(pred_logits, pseudo_mask,bg=False,seed=False,kernel_size=3, ignore_index=255,cross_mask=None,**kwargs):
    """
    pred_logits: [B,K,H,W], 模型输出的未归一化 logit
    pseudo_mask: [B,K,H,W], 弱监督伪掩码(二值化,每个通道为0或1)
    kernel_size: 形态学操作的核大小
    ignore_index: 忽略的标签索引（用于损失计算）
    cross_mask: [B,1,H,W], 融合伪标签的mask
    return: 损失值
    """
    device = pred_logits.device
    batch_size, num_classes, H, W = pseudo_mask.shape
    pred_probs = torch.softmax(pred_logits,dim=1)
    
    # 腐蚀和膨胀操作
    core_mask = morphological_operation_multi_class(pseudo_mask, kernel_size=kernel_size, op="erode")  # 核心区域
    dilated_mask = morphological_operation_multi_class(pseudo_mask, kernel_size=kernel_size, op="dilate")  # 扩展区域

    eroded_mask = core_mask#binary_mask - core_mask   
    expanded_mask = dilated_mask  

    # pseudo_mask[:,1:] = ignore_inde
    # import ipdb
    # ipdb.set_trace()

    if cross_mask is None:
        corr_mask = 0.5*(eroded_mask+dilated_mask)
    else :
        cross_mask = torch.nn.functional.interpolate(cross_mask,size=(H,W),mode='bilinear')
        corr_mask = 0.5*cross_mask*(eroded_mask+dilated_mask)

    corr_mask = corr_mask/(torch.sum(corr_mask,dim=1,keepdim=True)+1e-7)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    # factor = 0.5
    pseudo_mask = pseudo_mask*(1-pred_probs)+pred_probs*corr_mask

    cond1 = (pseudo_mask.sum(dim=1)==0)
    target1 = torch.argmax(pseudo_mask,dim=1)
    target1[cond1] = ignore_index

    if bg:
        loss_o = calculate_seeding_loss(pred_probs,pseudo_mask)
    elif seed:
        loss_o = calculate_seeding_loss(pred_probs,pseudo_mask,bg=False)
    else:
        loss_o = loss_fn(pred_logits,target1)

    return loss_o

if __name__ == '__main__':
    pred_logits = torch.tensor([[[3,4,5,1,1,1],
                                [3,3,3,6,1,0],
                                [2,2,1,0,0,0],
                                [4,0,0,4,5,0],
                                [0,0,0,0,0,0],
                                [0,0,0,0,0,0]],
                                [
                                [-3,-4,-5,-1,-1,-1],
                                [-3,-3,-3,-6,-1,1.0],
                                [-2,-2,-1,1.0,1.0,1.0],
                                [-4,1.0,1.0,-4,-5,1.0],
                                [1.0,1.0,1.0,1.0,1.0,1.0],
                                [1.0,1.0,1.0,1.0,1.0,1.0]
                                ]]).unsqueeze(0)
    
    pseudo_mask = (pred_logits>0).float()
    print('ori_mask',pseudo_mask)
    loss_ed = erode_dilate_loss(pred_logits,pseudo_mask,3)
    print(loss_ed)
