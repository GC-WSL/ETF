import torch
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import torch.nn.functional as F
from simclip.tools import dgcn_get_cues_from_seg_gt_tensor
from .losses.mask_loss import get_masked_ptc_loss 
from .losses.erode_dilate_loss import erode_dilate_loss


@HEADS.register_module()
class IdentityHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self,decay_parameter=0.996,if_gradclip=False,non_reduce=True,
                 bg=False,bg_fg_weight=(1,1),focaldice=False,seeding=False,
                 erodil=False,boundary=False,kernel_size = 3,
                 **kwargs):
        super(IdentityHead, self).__init__(
            input_transform=None, **kwargs)
        # self.conv_seg = None
        self.decay_parameter=decay_parameter
        self.non_reduce = non_reduce
        self.focaldice=focaldice
        self.seeding = seeding
        self.bg = bg
        self.bgfgweight = bg_fg_weight
        self.erodil = erodil
        self.boundary = boundary
        self.kernel_size=kernel_size

    def forward(self, inputs):
        return inputs

    def forward_train(self, inputs, data_samples,imgs=None,cls_logits = None,
                      pat_logits=None,cls_labels = None,img_metas=None,cross_mask=None,
                      vis_patch_tokens=None):
        gt_semantic_seg = [data_sample.gt_sem_seg.data for data_sample in data_samples]
        gt_semantic_seg = torch.stack(gt_semantic_seg)
        seg_logits = self.forward(inputs)
        losses = dict()
        if cls_labels is None:
            cls_labels = get_image_level_labels(gt_semantic_seg,seg_logits.shape[1])
        if cls_logits is not None:
            weights = torch.ones(cls_labels.shape[-1]+1,device=cls_labels.device)
            weights[-1] = 0.1
            loss_cls = F.cross_entropy(cls_logits,cls_labels,ignore_index=self.ignore_index,weight=weights)
            losses.update({'loss_cls':loss_cls})
        if pat_logits is not None:
            targets = torch.zeros_like(cls_labels,device=cls_labels.device)
            targets[cls_labels!=cls_labels.shape[-1]]=1
            loss_pat = F.binary_cross_entropy_with_logits(pat_logits[:,1:],target=targets[:,1:].float())
            losses.update({'loss_pat':loss_pat})

        if vis_patch_tokens is not None:

            seg_target = F.interpolate(gt_semantic_seg.float(),size=vis_patch_tokens.shape[-2:],mode='nearest')
            loss_pwc = get_masked_ptc_loss(vis_patch_tokens,seg_target)
            losses.update({'loss_pwc':loss_pwc})
        
        if self.erodil:
            seg_logits = F.interpolate(seg_logits,gt_semantic_seg.shape[-2:],mode='bilinear')
            seg_target = dgcn_get_cues_from_seg_gt_tensor(gt_semantic_seg.float(),seg_logits.shape)
            loss_ed = erode_dilate_loss(seg_logits,seg_target,bg=self.bg,seed=self.seeding,
                                        kernel_size=self.kernel_size,cross_mask=cross_mask)

            losses.update({'loss_erodil':loss_ed})
        else :
            loss_ce = self.loss(seg_logits, data_samples,dict())
            losses.update(loss_ce)   
        return losses

import torch

def get_image_level_labels(mask: torch.Tensor, num_classes: int, ignore_label=255,reduce_zero=False):
    """
    Args:
        mask (torch.Tensor): shape [B, 1, H, W], 每个像素是类别ID
        num_classes (int): 总类别数 C
        ignore_label (int): 要忽略的类别ID

    Returns:
        image_labels (torch.Tensor): shape [B, C], 每行表示图像中存在的类别ID,其他为0
    """
    B, _, H, W = mask.shape
    device = mask.device

    # 初始化输出
    image_labels = torch.zeros((B, num_classes), device=device, dtype=torch.long)

    flat_mask = mask.view(B, -1)  # shape [B, H*W]

    for b in range(B):
        valid_pixels = flat_mask[b][flat_mask[b] != ignore_label]  # 忽略255
        unique_classes = torch.unique(valid_pixels)
        for cls in unique_classes:
            if cls < num_classes and cls >= 0:
                image_labels[b, cls] = cls  # 在对应列填入类别编号

    return image_labels

    
