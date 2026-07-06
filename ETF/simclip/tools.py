import os
import cv2
from matplotlib import cm
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
import torch.nn as nn

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

class GradientClipping(nn.Module):
    def __init__(self, start_value, patch_size):
        super().__init__()
        self.start_value = start_value
        self.patch_size = patch_size

    def forward(self, ori_loss):
        detach_loss = ori_loss.detach().clone()

        mean_loss = detach_loss.mean()

        # set start loss clamp threshold
        if mean_loss > self.start_value:
            return ori_loss, ori_loss

        b, h, w = detach_loss.shape

        # all batch average
        detach_loss = detach_loss.mean(dim=0).unsqueeze(0)
        local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=self.patch_size,
                                  stride=self.patch_size, padding=h % self.patch_size,
                                  count_include_pad=False).squeeze(1)
        local_mean = torch.maximum(local_mean, mean_loss)
        local_mean = torch.repeat_interleave(local_mean, b, dim=0)
        local_mean = torch.repeat_interleave(local_mean, self.patch_size, dim=1)
        local_mean = torch.repeat_interleave(local_mean, self.patch_size, dim=2)

        clamp_loss = ori_loss - local_mean
        clamp_loss = torch.clamp(clamp_loss, None, 0)
        loss = clamp_loss + local_mean

        return ori_loss, loss

def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result

def apply_heatmap(img, cam, alpha=0.5,normal=False):
    
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) if normal else cam
    
    heatmap = cm.jet(cam)[:, :, :3]  
    heatmap = (heatmap * 255).astype(np.uint8)  
    
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  
    result = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    
    return result

def colorful(img)->Image.Image:
    color_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    palette=[]
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21]=np.array([   [128, 0, 0],
                                [0, 128, 0],
                                [128, 128, 0],
                                [0, 0, 128],
                                [128, 0, 128],
                                [0, 128, 128],
                                [128, 128, 128],
                                [64, 0, 0],
                                [192, 0, 0],
                                [64, 128, 0],
                                [192, 128, 0],
                                [64, 0, 128],
                                [192, 0, 128],
                                [64, 128, 128],
                                [192, 128, 128],
                                [0, 64, 0],
                                [128, 64, 0],
                                [0, 192, 0],
                                [128, 192, 0],
                                [0, 64, 128]
                             ], dtype='uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color_img[i,j]=palette[img[i,j]]
    return Image.fromarray(color_img)


from scipy.ndimage import zoom

def crf_inference(img, probs, t=10, scale_factor=1):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    class_num, h, w = probs.shape
    n_labels = class_num

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80 / scale_factor, srgb=3, rgbim=np.copy(img), compat=5)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def dgcn_crf_operation(images, probs, img_metas):
    img_mean_4_dataset = img_metas['mean']
    img_std_4_dataset = img_metas['std']

    batchsize, _, h, w = probs.shape
    probs[probs < 0.0001] = 0.0001
    # unary = np.transpose(probs, [0, 2, 3, 1])

    im = images
    im = zoom(im, (1.0, 1.0, float(h) / im.shape[2], float(w) / im.shape[3]), order=1)
    im = np.transpose(im, [0, 2, 3, 1])
    im = im * img_std_4_dataset
    im = im + img_mean_4_dataset
    im = np.ascontiguousarray(im, dtype=np.uint8)
    result = np.zeros(probs.shape)
    for i in range(batchsize):
        result[i] = crf_inference(im[i], probs[i])

    result[result < 0.0001] = 0.0001
    result = result / np.sum(result, axis=1, keepdims=True)
    
    result = np.log(result)

    return result


def clip(x, min, max):
    x_min = x < min
    x_max = x > max
    y = torch.mul(torch.mul(x, (~x_min).float()), (~x_max).float()) + ((x_min.float()) * min) + (x_max * max).float()
    return y


def dgcn_get_cues_from_seg_gt_tensor(gt_semantic_seg, cues_shape):
    B, K, H, W = cues_shape
    cues = torch.zeros(cues_shape, dtype=torch.float32, device=gt_semantic_seg.device)
    gt_semantic_seg = F.interpolate(gt_semantic_seg,size=(H,W),mode='nearest')
    for c in range(K):
        pos = torch.where(gt_semantic_seg == c)
        cues[pos[0], pos[1] + c, pos[2], pos[3]] = 1

    return cues


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
            bg_term = -(noisy_weights * bg_mask * torch.log(pred_bg.clamp(min=1e-8)))
            bg_count = bg_mask.sum(dim=(1, 2, 3), keepdim=True)
            bg_loss = (bg_term.sum(dim=(1, 2, 3), keepdim=True) / bg_count.clamp(min=eps_count)).mean()
            
            # Foreground loss
            fg_term = -(noisy_weights * fg_mask * torch.log(pred_fg.clamp(min=1e-8)))
            fg_count = fg_mask.sum(dim=(1, 2, 3), keepdim=True)
            fg_loss = (fg_term.sum(dim=(1, 2, 3), keepdim=True) / fg_count.clamp(min=eps_count)).mean()
            
            total_loss += (bg_fg_weight[0]*bg_loss + bg_fg_weight[1]*fg_loss)
        
        # Average loss over all specified bg_idxs
        total_loss /= len(bg_idxs)
        return total_loss
    
def dice_loss(inputs, targets, num_masks):
    # inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def sigmoid_focal_loss(inputs, targets, num_masks, alpha=0.25, gamma=2):
    # prob = inputs.sigmoid()
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
        
    return loss.mean(1).sum() / num_masks


def dgcn_softmax(preds, min_prob):
    preds_max = torch.max(preds, dim=1, keepdim=True)
    preds_exp = torch.exp(preds - preds_max[0])
    probs = preds_exp / torch.sum(preds_exp, dim=1, keepdim=True)
    min_prob = torch.ones((probs.shape), device=min_prob.device) * min_prob
    probs = probs + min_prob
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs

def orthogonal_cam_loss(cams, targets):
    """
    Penalizes overlap between CAMs of co-occurring positive classes.
    Args:
        cams: (B, C, H, W)  # CAMs for all classes
        targets: (B, C)     # Multi-hot binary targets (1 for positive, 0 for negative)
    Returns:
        loss: scalar        # Orthogonal CAM loss (penalizes pairwise positive-class CAM overlap)
    """
    B, C, H, W = cams.shape
    
    # Normalize CAMs to [0, 1] using sigmoid
    cams = torch.sigmoid(cams)  # (B, C, H, W)
    
    # Get mask of positive classes for each sample
    pos_mask = targets.float()  # (B, C)
    
    # For each sample, compute pairwise dot products between all positive-class CAMs
    loss = 0.0
    for b in range(B):
        # Get indices of positive classes for this sample
        pos_indices = torch.where(pos_mask[b] > 0)[0]  # (num_pos,)
        num_pos = len(pos_indices)
        
        if num_pos < 2:
            continue  # No overlap to penalize if fewer than 2 positive classes
        
        # Extract CAMs of positive classes
        pos_cams = cams[b, pos_indices]  # (num_pos, H, W)
        
        # Compute all pairwise dot products (upper triangular to avoid duplicates)
        pairwise_dots = 0.0
        for i in range(num_pos):
            for j in range(i + 1, num_pos):
                pairwise_dots += (pos_cams[i] * pos_cams[j]).sum()
        
        # Normalize by number of pairs (C(num_pos, 2) = num_pos*(num_pos-1)/2)
        normalized_dots = pairwise_dots / (H*W*(num_pos * (num_pos - 1) / 2 + 1e-6))
        loss += normalized_dots
    
    # Average over batch
    loss = loss / (B + 1e-6)
    return loss


from scipy.optimize import curve_fit

def should_update_pseudo_labels(
    current_iters: int, 
    ious: list, 
    eval_interval: int, 
    threshold: float = 0.9,
    min_points:int = 10,
    max_iters:int = 3000,
) -> bool:
    current_iters = current_iters // eval_interval

    if len(ious) < min_points:
        return False
    # return True
    try:
        xdata = np.linspace(
            0, 
            len(ious) * eval_interval, 
            len(ious),
        )
        
        popt, _ = curve_fit(
            lambda x, a, b, c: a*(1 - np.exp(-x**b/c)),
            xdata, 
            ious,
            p0=(1, 1, 1),
            bounds=([0,0,0], [1,1,np.inf]),
            method='trf',
            sigma=np.geomspace(1, 0.1, len(ious)),
            absolute_sigma=True,
        )
        a, b, c = popt
        
        iters = np.arange(1, max_iters//eval_interval)
        def deriv(x):
            x = x + 1e-6
            return a*b/c * np.exp(-x**b/c) * x**(b-1)
        
        base_deriv = deriv(1)
        relative_change = np.abs(deriv(iters) - base_deriv) / np.abs(base_deriv)
        relative_change[relative_change > 1] = 0
        update_iters = np.sum(relative_change <= threshold) + 1

        return current_iters >= update_iters

    except (RuntimeError, ValueError):
        return False
    

from operator import itemgetter

def getlabel(label_dict: dict, batch_img_metas,n_class=5,non_reduce=False):

    keys = [os.path.basename(meta['img_path']).replace('.png','') for meta in batch_img_metas]

    
    getter = itemgetter(*keys)
    values = getter(label_dict)

    values = values if isinstance(values, tuple) else (values,)

    num_samples = len(keys)
    multi_hot = np.zeros((num_samples, n_class), dtype=np.uint8)

    if non_reduce:
        rows_cols = [
            (i, int(label_str))
            for i, labels in enumerate(values)
            for label_str in labels
        ]
    else:
        rows_cols = [
            (i, int(label_str)-1)
            for i, labels in enumerate(values)
            for label_str in labels
            if label_str != '0'  # 过滤背景类
        ]
    
    if rows_cols:
        rows, cols = zip(*rows_cols)
        multi_hot[rows, cols] = 1  # 向量化赋值

    return multi_hot

def get_multihot(values:list[str],n_class=5,non_reduce=False):

    num_samples = len(values)
    multi_hot = np.zeros((num_samples, n_class), dtype=np.uint8)

    if non_reduce:
        rows_cols = [
            (i, int(label_str))
            for i, labels in enumerate(values)
            for label_str in labels
        ]
    else:
        rows_cols = [
            (i, int(label_str)-1)
            for i, labels in enumerate(values)
            for label_str in labels
            if label_str != '0'  # 过滤背景类
        ]
    
    if rows_cols:
        rows, cols = zip(*rows_cols)
        multi_hot[rows, cols] = 1

    return multi_hot


def adaptive_weighted_pool(x_patch_cls,decay_parameter=0.995):
    """
    x_patch_cls:[B,C,H,W]
    return patch logits:[B,C]
    """

    x_patch_flattened = x_patch_cls.view(x_patch_cls.shape[0], x_patch_cls.shape[1], -1).permute(0, 2, 1)

    sorted_patch_token, indices = torch.sort(x_patch_flattened, dim=-2, descending=True)


    weights = torch.logspace(
        start=0,
        end=x_patch_flattened.size(-2) - 1,
        steps=x_patch_flattened.size(-2),
        base=decay_parameter,
    ).to(x_patch_cls.device)


    x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()

    return x_patch_logits


def visual_backup(img,datasample,cams:torch.Tensor,batch_img_metas=None,train_cfg=None,tag='auxhead'):
        
        def find_latest_log_dir(base_dir):
            subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not subdirs:
                return None
            subdirs.sort(key=lambda x: os.path.getctime(os.path.join(base_dir, x)), reverse=True)
            return os.path.join(base_dir, subdirs[0])
        if train_cfg is not None:
            confid_dirs = find_latest_log_dir(train_cfg.work_dir)
            heatmap_dir = os.path.join(confid_dirs,'confid_weight')
        else :
            heatmap_dir = f"./work_dirs/{tag}/cams/"
        os.makedirs(heatmap_dir,exist_ok='True')
        if (datasample is None)&(batch_img_metas is not None):
            img_name = batch_img_metas[0]['img_path'].split('/')[-1].split('.')[0]
        else :
            img_name = datasample[0].metainfo['img_path'].split('/')[-1].split('.')[0]
        img_mean_4_dataset = [123.675, 116.28, 103.53]
        img_std_4_dataset = [58.395, 57.12, 57.375]

        im = img
        im = im * img_std_4_dataset
        im = im + img_mean_4_dataset
        im = np.ascontiguousarray(im, dtype=np.uint8)
        for idx in range(cams.shape[0]):
            img_heatmap = apply_heatmap(im,cams[idx],alpha=0.4)
            cv2.imwrite(os.path.join(heatmap_dir,img_name+f'_c{idx}'+'.png'),cv2.cvtColor(img_heatmap, cv2.COLOR_RGB2BGR).astype(np.uint8))


import threading
import queue

_visual_queue = queue.Queue(maxsize=100)  
_visual_worker = None
_visual_stop_event = threading.Event()

def _start_visual_worker():
    """启动可视化工作者线程"""
    global _visual_worker
    if _visual_worker is None or not _visual_worker.is_alive():
        _visual_stop_event.clear()
        _visual_worker = threading.Thread(
            target=_visual_worker_loop,
            daemon=True  # 设置为守护线程，主程序退出时自动结束
        )
        _visual_worker.start()

def _stop_visual_worker():
    """停止可视化工作者线程"""
    _visual_stop_event.set()
    _visual_queue.put(None)  # 发送停止信号

def _visual_worker_loop():
    """可视化工作线程的主循环"""
    while not _visual_stop_event.is_set():
        try:
            task = _visual_queue.get(timeout=1.0)
            if task is None:  # 收到停止信号
                break
            
            img, cams, heatmap_dir, img_name = task
            img_mean_4_dataset = [123.675, 116.28, 103.53]
            img_std_4_dataset = [58.395, 57.12, 57.375]

            im = img.copy()
            im = im * img_std_4_dataset
            im = im + img_mean_4_dataset
            im = np.ascontiguousarray(im, dtype=np.uint8)
            
            for idx in range(cams.shape[0]):
                if cams[idx].sum() <= 1:
                    continue
                img_heatmap = apply_heatmap(im, cams[idx], alpha=0.4)
                cv2.imwrite(
                    os.path.join(heatmap_dir, f"{img_name}_c{idx}.png"),
                    cv2.cvtColor(img_heatmap, cv2.COLOR_RGB2BGR)
                )
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Visualization worker error: {e}")

def visual(img, datasample, cams, batch_img_metas=None, train_cfg=None, tag='auxhead'):
    """非阻塞式可视化函数，将任务放入队列"""
    # 确保工作者线程已启动
    _start_visual_worker()
    def find_latest_log_dir(base_dir):
            subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not subdirs:
                return None
            subdirs.sort(key=lambda x: os.path.getctime(os.path.join(base_dir, x)), reverse=True)
            return os.path.join(base_dir, subdirs[0])
    # 准备目录
    if train_cfg is not None:
        confid_dirs = find_latest_log_dir(train_cfg.work_dir)
        heatmap_dir = os.path.join(confid_dirs, 'confid_weight')
    else:
        heatmap_dir = f"./work_dirs/{tag}/cams/"
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # 获取图像名称
    if datasample is None and batch_img_metas is not None:
        img_name = batch_img_metas[0]['img_path'].split('/')[-1].split('.')[0]
    else:
        img_name = datasample[0].metainfo['img_path'].split('/')[-1].split('.')[0]
    
    # 转换数据格式并放入队列
    img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
    cams_np = cams.cpu().numpy() if isinstance(cams, torch.Tensor) else cams
    
    try:
        _visual_queue.put_nowait((img_np, cams_np, heatmap_dir, img_name))
    except queue.Full:
        print("Visualization queue full, skipping this batch")