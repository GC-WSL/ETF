from dataclasses import dataclass
import os
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import BatchNorm2d, GroupNorm, LayerNorm
import numpy as np
from torch import Tensor
from mmseg.utils import (
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.utils import resize
from mmengine.logging import print_log
from mmseg.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
import logging
from .utils import tokenize
from operator import itemgetter
from .tools import adaptive_weighted_pool
from peft import OFTConfig,get_peft_model

@MODELS.register_module()
class SimCLIP(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 class_names,
                 context_length,
                 data_preprocessor,
                 class_names_ms=None,
                 text_encoder=None,
                 context_decoder=None,
                 neck=None,
                 tau=0.07,
                 auxiliary_head=None,
                 identity_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,

                 multi_class_context = False,
                 multi_scale_crossattn = False,
                 feature_stride = None,
                 cross_mask = False,

                 superpixel = False,
                 noisy_loss = False,

                 align_corners=False,
                 label_txt='train.txt',
                 reduce_zero = False,

                 norm_eval=True,
                 layer_norm_eval = True,
                 tau_learn=False,
                 frozen = True,
                 **args):
        super().__init__(data_preprocessor,init_cfg)

        self.img_metas = dict(mean=data_preprocessor.mean,
                              std=data_preprocessor.std)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained


            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'

            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained \
                    and 'ViT-L' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        self.text_encoder = MODELS.build(text_encoder)
        # self.context_decoder = MODELS.build(context_decoder)
        self.context_length = context_length

        # if tau_learn:
        #     self.tau = nn.Parameter(torch.ones(1)*0.07) # useless
        # else:
        self.tau = tau

        self.multi_scale_crossattn = multi_scale_crossattn

        self.multi_class_context = multi_class_context
        self.superpixel = superpixel
        self.noisy_loss = noisy_loss
        
        # if self.superpixel :
            # self.superpixelCluster = SuperPixelCluster(K=10,iterations=10)

        self.norm_eval = norm_eval
        self.layer_norm_eval = layer_norm_eval
        self.decay_parameter=0.99

        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.with_identity_head = False

        self._init_identity_head(identity_head)
        self.out_indices = backbone.out_indices
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # if feature_stride is not None:
        #     self.adapter = pyramid_adapter(self.backbone.output_dim,feature_stride)
        # else :
        #     self.adapter = nn.Identity()
        # added template

        self.texts = torch.cat([tokenize(f'an aerial view:{c}', context_length=self.context_length) for c in class_names]) # shape: torch.Size([21, 5])
        self.num_classes = len(self.texts)

        context_length = self.text_encoder.context_length - self.context_length # 13 - 5

        # token embed dim is same as the width of text encoder
        token_embed_dim = self.text_encoder.embed_dim  # 512 for base; 768 for large
        if context_length >0 :
            self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim)) # shape: torch.Size([1, 8, 512])
            nn.init.trunc_normal_(self.contexts)
        else :
            self.contexts = None

        if class_names_ms is not None:
            self.texts_ms = torch.cat([tokenize(f'an aerial view:{c}', context_length=self.context_length) for c in class_names_ms])
            self.contexts_ms = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
            nn.init.trunc_normal_(self.contexts_ms)
        else :
            self.texts_ms = None

        # text dim is equal to the width of text encoder embed
        text_dim = self.text_encoder.embed_dim  # 512 for base; 768 for large

        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-1)
        self.beta = nn.Parameter(torch.ones(text_dim) * 1e-1)
        # self.cls_scale = nn.Parameter(torch.ones(1)*self.num_classes,requires_grad=True)
        
        if multi_class_context:
            self.cls_tokens = nn.Embedding(self.num_classes+1 , self.text_encoder.width)

        self.interactor = nn.Sequential()
        self.zoom_in = nn.Sequential()
        self.zoom_out = nn.Sequential()
        for _ in self.out_indices:
            self.zoom_in.append(nn.Linear(self.backbone.width,self.text_encoder.width))
            self.interactor.append(MODELS.build(context_decoder))
            self.zoom_out.append(nn.Linear(self.text_encoder.width,self.backbone.width))
        # assert self.with_decode_head

        self.cross_mask = cross_mask
        if cross_mask:
            self.adjustor = FeedForward_Sigmoid(self.backbone.output_dim+self.num_classes)

        self.iters = 1
        self.max_iters = 10000

        self.label_txt = label_txt
        self.label_list = self.load_label_txt()
        self.reduce_zero = reduce_zero

        self.align_corners = align_corners
        self.frozen = frozen
        self.train()
        self._init_finetune()

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        if decode_head is not None:
            self.decode_head = MODELS.build(decode_head)
            # self.align_corners = self.decode_head.align_corners
            self.num_classes = self.decode_head.num_classes
        else :
            self.decode_head = None

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)
    
    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = MODELS.build(identity_head)

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg,
                                    **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode,pseudo_gt = self.decode_head.forward_train(x, img_metas,
                                                         gt_semantic_seg,
                                                         self.train_cfg,
                                                         **kwargs)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses,pseudo_gt


    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits


    def _identity_head_forward_train(self, x, gt_semantic_seg, imgs=None,**kwarg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if imgs == None:
            loss_aux = self.identity_head.forward_train(
                x, gt_semantic_seg,**kwarg)
        else:
            loss_aux = self.identity_head.forward_train(
                x, gt_semantic_seg,imgs,**kwarg)
        losses.update(add_prefix(loss_aux, 'identity'))
        return losses

    def _auxiliary_head_forward_train(self, x, gt_semantic_seg,img=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, 
                                                  gt_semantic_seg,
                                                  )
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux,_ = self.auxiliary_head.forward_train(
                x, gt_semantic_seg,img)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses,_

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self._forward(img, None)

        return seg_logit
    
    def extract_feat(self, img):
        """Extract features from images."""

        vis_pre = self.backbone.forward_pre(img) # LND
        txt_pre,eos_indx,t_size = self.text_encoder.forward_pre(self.texts.to(vis_pre.device), self.contexts) #text_pre: LND
        num_layers = self.backbone.transformer.layers
        if (self.texts_ms is not None) & self.multi_class_context:
            txt_pre_ms,eos_indx_ms,t_size_ms = self.text_encoder.forward_pre(self.texts_ms.to(vis_pre.device), self.contexts_ms) #text_pre: LND 
        else :
            txt_pre_ms = txt_pre.clone()
        vis_cls_logits = []
        pat_cls_logits = []
        # if self.superpixel:
        #     superpixels = []
        j = 0
        features = []
        for i in range(num_layers):
            vis_pre = self.backbone.transformer.resblocks[i](vis_pre)
            txt_pre = self.text_encoder.transformer.resblocks[i](txt_pre)
            if (self.texts_ms is not None) & self.multi_class_context:
                txt_pre_ms = self.text_encoder.transformer.resblocks[i](txt_pre_ms)
            if i not in self.out_indices:
                continue
            else :
                v = vis_pre.clone() # [H*W+1,B,768]
                v = self.zoom_in[j](v)
                v = v[1:, :, :].permute(1, 0, 2) # [B,H*W,512]

                B,N,D = v.shape
                # if self.superpixel:
                #     superpixels.append(self.superpixelCluster(v))

                L,K,_ = txt_pre.shape
                t = txt_pre.clone().reshape(-1,L*K,D).expand(B,-1,-1) # [B,L*K,512]
                qlen = 0
                if self.multi_class_context:
                    qlen = K+1
                    global_feat = self.cls_tokens.weight.unsqueeze(0).repeat(B, 1, 1) # [B, K, D]
                    v = torch.cat([global_feat,v],dim=1)
                    # t_ms = txt_pre_ms.clone() # [L,K,512]

                t_diff,v_diff,_v_self_attn = self.interactor[j](t, v)

                txt_h = (t + self.gamma*t_diff).permute(1,0,2)
                txt_h = torch.mean(txt_h,dim=1).reshape(L,K,D)

                if qlen>0:
                    vis_cls_tokens = v_diff[:,:qlen,:] # [B,K,512]
                    cls_logits = torch.einsum('bkd,lcd->bklc',
                                              F.normalize(vis_cls_tokens,p=2,dim=-1),
                                              F.normalize(txt_pre,p=2,dim=-1))
                    vis_h = v[:,qlen:,:]+self.beta*v_diff[:,qlen:,]
                    vis_pat_tokens = vis_h.permute(0,2,1).reshape((B,D,int(N**0.5),int(N**0.5)))

                    txt_ms = (txt_pre_ms+txt_h)/2
                    pat_logits = torch.einsum('bdhw,lcd->blchw',
                                              F.normalize(vis_pat_tokens,p=2,dim=-1),
                                              F.normalize(txt_ms,p=2,dim=-1)).sum(dim=1)
                    pat_cls_logits.append(pat_logits)
                    vis_cls_logits.append(cls_logits.sum(dim=2))

                v_diff = self.zoom_out[j](self.beta*v_diff[:,qlen:,])
                vis_pre[1:,:,:] = vis_pre[1:,:,:]+(v_diff).permute(1,0,2) # [H*W+1,B,768]
                txt_pre = txt_h.clone()

                j+=1
 
                features.append(vis_pre[1:,:,:].permute(1,2,0).reshape(B,-1,int(N**0.5),int(N**0.5)))
        x = {
            'vis':self.backbone.forward_post(vis_pre),
            'txt':self.text_encoder.forward_post(txt_pre,eos_indx,t_size),
            'features':features,
        }
        if len(vis_cls_logits)>0:
            x['cls_logits'] = torch.stack(vis_cls_logits,dim=0).mean(dim=0)
        if len(pat_cls_logits)>0:
            x['pat_logits'] = torch.stack(pat_cls_logits,dim=0).mean(dim=0)
        return x

    def after_extract_feat(self,x):
        vis_embedding = x['vis']
        txt_embedding = x['txt']
        features = x['features']
        visual_embedding = vis_embedding[:,1:,:]
        B,N,C = visual_embedding.shape
        visual_embedding = visual_embedding.permute(0,2,1).reshape(B,C,int(N**0.5),int(N**0.5))
        text_embedding = txt_embedding.expand(B,-1,-1)
        B,K,C = text_embedding.shape

        score_map = torch.einsum('bchw,bkc->bkhw', 
                                    F.normalize(visual_embedding, dim=1, p=2), 
                                    F.normalize(text_embedding, dim=2, p=2))
        
        visual_cls_logits = x.get('cls_logits',None)
        pat_logits = x.get('pat_logits',None)
        if pat_logits is not None:
            pat_logits = adaptive_weighted_pool(pat_logits,decay_parameter=0.995)

        cam_f = x.get('pat_logits',None)
            
        return AfterExtractFeatResult(
            text_embeddings=text_embedding,
            features=features,
            score_map=score_map,
            visual_embeddings=visual_embedding,
            visual_cls_logits=visual_cls_logits,
            pat_logits=pat_logits,
            cam_f=cam_f
        )

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
            gt_semantic_seg = [data_sample.gt_sem_seg.data for data_sample in data_samples]
            gt_semantic_seg = torch.stack(gt_semantic_seg)
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        img_metas = self.img_metas

        keys = [os.path.basename(meta['img_path']).replace('.png','') for meta in batch_img_metas]
        getter = itemgetter(*keys)
        cls_labels = getter(self.label_list)
        image_labels = torch.zeros((inputs.shape[0], self.num_classes), device=inputs.device, dtype=torch.long)+self.num_classes
        for b in range(image_labels.shape[0]):
            for cls in cls_labels[b]:
                cls = int(cls) -1 if self.reduce_zero else int(cls)
                if cls < self.num_classes and cls >= 0:
                    image_labels[b, cls] = cls

        # for name, params in self.named_parameters():
        #     if params.requires_grad == True:
        #         print(name, params.requires_grad)

        # import ipdb;ipdb.set_trace()

        x = self.extract_feat(inputs)

        result = self.after_extract_feat(x)

        text_embeddings = result.text_embeddings
        vis_features = result.features
        score_map = result.score_map
        visual_embeddings = result.visual_embeddings
        visual_cls_logits = result.visual_cls_logits
        vis_pat_logits = result.pat_logits
        cam_f = result.cam_f

        losses = dict()

        if cam_f is not None:
            
        #     cam_prob = torch.softmax(cam_f/self.tau,dim=1)
        #     prob = torch.softmax(score_map/self.tau,dim=1)
        #     B,K,H,W = prob.shape
        #     entropy = -prob * torch.log(prob + 1e-8) - (1 - prob) * torch.log(1 - prob + 1e-8)
        #     weight = (1 - entropy)/ torch.log(torch.tensor(K+1))  # high confidence → high weight
        # # #     # L1 loss with adaptive weighting
        #     # cam_prob = torch.where(cam_prob>0.5,1,0)

        #     loss_dist = torch.abs(prob - cam_prob.detach())  # [B,K,H,W]
        #     loss_dist = (loss_dist * weight).sum(dim=1).mean() 
        # # #     loss_dist = (loss_dist).sum(dim=1).mean() 
        #     losses.update({'loss_dist':loss_dist})
            
            score_map = score_map*cam_f

        if self.cross_mask:
            # import ipdb;ipdb.set_trace()
            cross_mask = self.adjustor(torch.cat([result.score_map,visual_embeddings],dim=1))

            softmax_probs = torch.softmax((score_map/self.tau).detach(), dim=1)
            entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1, keepdim=True)
            max_entropy = torch.log(torch.tensor(self.num_classes).float().to(score_map.device))
            normalized_entropy =  1- entropy / max_entropy
            if self.iters<=1000:
                loss_cross = F.mse_loss(cross_mask,normalized_entropy,reduction='mean')
            else :
                loss_cross = 1e-3*F.mse_loss(cross_mask,normalized_entropy,reduction='mean')
            losses.update(add_prefix({'loss_mse':loss_cross},'crossMask'))
        else:
            cross_mask = None

        if self.decode_head is not None:
            loss_decode, pseudo_gt = self._decode_head_forward_train(vis_features, img_metas,gt_semantic_seg, 
                                                            batch_img_metas=batch_img_metas,
                                                            label_list=self.label_list)
            losses.update(loss_decode)

        # if cam_f is not None:
        #     score_map = score_map*cam_f
        if self.with_identity_head:
            loss_identity = self._identity_head_forward_train(
                score_map/self.tau, data_samples,cls_logits = visual_cls_logits,pat_logits = vis_pat_logits,
                cls_labels = image_labels,img_metas=img_metas,imgs=inputs,cross_mask = cross_mask,
                # vis_patch_tokens=visual_embeddings
                )
            losses.update(loss_identity)
            
        self.iters+=1
        
        return losses

    def encode_decode(self, img,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # x = self.extract_feat(img)

        x = self.extract_feat(img)
        # _x_orig = [x[i] for i in range(len(self.out_indices))]
        result = self.after_extract_feat(x)

        text_embeddings = result.text_embeddings
        vis_features = result.features
        score_map = result.score_map
        visual_embeddings = result.visual_embeddings
        visual_cls_logits = result.visual_cls_logits
        cam_f = result.cam_f


        if cam_f is not None:
            score_map = score_map*cam_f
        #     score_map=score_map*cam_f+score_map

            # score_map[:,1:] = F.relu(cam_f[:,1:])*score_map[:,1:]
            # pass
        #     score_map[:,1:] =  F.relu(cam_f)[:,1:]
            

        if self.test_cfg.get('save_context',False):
            save_dir = self.test_cfg.get('outdir', os.path.join(self.train_cfg.work_dir,'context_npy'))
            
            ori_filename = os.path.basename(batch_img_metas[0]['img_path'])
            save_path = os.path.join(save_dir, ori_filename.replace('.png', '.npy')) 

            os.makedirs(save_dir, exist_ok=True)

            np.save(save_path, {
                'text_embed': text_embeddings.detach().cpu().numpy(),
                'score_map': score_map.detach().cpu().numpy(),
                'vis_embed': visual_embeddings.detach().cpu().numpy(),
            }, allow_pickle=True)

        # if self.with_neck:
        #     x_with_score = list(self.neck(x_with_score))

        # if self.text_head:
        #     x = [text_embeddings,] + x_orig
        # else:
        # x = x_with_score
        # print('text_embedding=', text_embeddings[0])
        # out = self._decode_head_forward_test(x,batch_img_metas)
        # print('cls_map=', out[0,:,40, 40])
        out = score_map/self.tau

        if self.decode_head is not None:
            out = self._decode_head_forward_test(vis_features,batch_img_metas)

        
        out = resize(
            input=out,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        result = self.after_extract_feat(x)

        text_embeddings = result.text_embeddings
        vis_features = result.features
        if self.with_neck:
            x = list(self.neck(vis_features))
        if self.with_decode_head:
            x = self.decode_head.forward(x)
        return x

    # TODO refactor
    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        if self.test_cfg.get('save_infer',False):
            save_dir = self.test_cfg.get('outdir', os.path.join(self.train_cfg.work_dir,'prob_npy'))
            
            ori_filename = os.path.basename(batch_img_metas[0]['img_path'])
            save_path = os.path.join(save_dir, ori_filename.replace('.png', '.npy')) 

            os.makedirs(save_dir, exist_ok=True)
            seg_probs = F.softmax(seg_logit,dim=1)
            seg_probs_np = seg_probs.squeeze(0).cpu().numpy()  # [C, H, W]
            seg_pred_np = seg_pred.cpu().numpy().squeeze(0)    # [H, W]
            unique_keys = np.unique(seg_pred_np)

            seg_probs_np = seg_probs_np[unique_keys]

            np.save(save_path, {
                'prob': seg_probs_np,
                'keys': unique_keys,
                'pred': seg_pred_np
            }, allow_pickle=True)

        return seg_logit

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(SimCLIP, self).train(mode)
        if self.frozen :
            if hasattr(self, 'backbone'):  
                self.backbone = getattr(self, 'backbone').eval()
                for name, params in self.backbone.named_parameters():
                        params.requires_grad = False
            if hasattr(self, 'text_encoder'):  
                self.text_encoder = getattr(self, 'text_encoder').eval()
                for name, params in self.text_encoder.named_parameters():
                        params.requires_grad = False
        else :
            backbone = getattr(self, 'backbone')
            if mode and self.norm_eval:
                for m in backbone.modules():
                        # trick: eval have effect on BatchNorm only
                    if isinstance(m, BatchNorm2d):
                        for param in m.parameters():
                            param.requires_grad = False
                        m.eval()
                    if self.layer_norm_eval:
                        if isinstance(m, LayerNorm):
                            for param in m.parameters():
                                param.requires_grad = False
                            m.eval()
        """
        if hasattr(self, 'backbone'):  
            backbone = getattr(self, 'backbone')
            if mode and self.norm_eval:
                for m in backbone.modules():
                        # trick: eval have effect on BatchNorm only
                    if isinstance(m, BatchNorm2d):
                        for param in m.parameters():
                            param.requires_grad = False
                        m.eval()
                    if self.layer_norm_eval:
                        if isinstance(m, LayerNorm):
                            for param in m.parameters():
                                param.requires_grad = False
                            m.eval()
            else:
                backbone.eval()
                for name, params in backbone.named_parameters():
                    params.requires_grad = False
        #             # print(name, params.requires_grad)
        if hasattr(self, 'text_encoder'):  
            text_encoder = getattr(self, 'text_encoder')
            if mode and self.norm_eval:
                for m in text_encoder.modules():
                        # trick: eval have effect on BatchNorm only
                    if isinstance(m, BatchNorm2d):
                        for param in m.parameters():
                            param.requires_grad = False
                        m.eval()
                    if self.layer_norm_eval:
                        if isinstance(m, LayerNorm):
                            for param in m.parameters():
                                param.requires_grad = False
                            m.eval()
            else:
                text_encoder.eval()
                for name, params in text_encoder.named_parameters():
                    params.requires_grad = False
            """
            # todo: configure what it mean
            # self._freeze_stages()
            
        # if mode and self.fix_clip_things:
        #     pass
    def _init_finetune(self):
        if self.train_cfg.get('clip_finetune',None) is not None:
            modelt = self.train_cfg.clip_finetune.get('targetm','backbone')
            model = getattr(self,modelt)
            # layert = self.train_cfg.clip_finetune.get('targetl','')
            for name, params in model.named_parameters():
                if "transformer" in name:
                    if self.train_cfg.clip_finetune.get('method','none') == "oft":
                        if "attn" in name or "position" in name:
                            params.requires_grad = True
                        else:
                            params.requires_grad = False
                else:
                    params.requires_grad = False
        for name, params in self.named_parameters():
                if params.requires_grad == True:
                    print(name, params.requires_grad)

    def load_label_txt(self):
        with open(self.label_txt,'r') as f:
            label_list = {line.strip().split()[0]:line.strip().split()[1:] for line in f}
        return label_list


@dataclass
class AfterExtractFeatResult:
    text_embeddings: torch.Tensor
    features: List[torch.Tensor]
    score_map: torch.Tensor
    visual_embeddings: torch.Tensor
    visual_cls_logits: Optional[torch.Tensor] = None
    cam_f: Optional[torch.Tensor] = None
    pat_logits:Optional[torch.Tensor] = None


class pyramid_adapter(nn.Module):
    def __init__(self, in_channels,stride = 4):
        super(pyramid_adapter, self).__init__()
        embed_dim = in_channels
        self.stride = stride
        self.fpn1 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                
            )

        self.fpn2 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                
            )

        self.fpn3 = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            )

        self.fpn4 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        
    def forward(self, x, ):
        if self.stride == 4:
            x = self.fpn1(x)
        elif self.stride == 8:
            x = self.fpn2(x)
        elif self.stride == 16:
            x = self.fpn3(x)
        else :
            x = self.fpn4(x)
        return x
    
class FeedForward_Sigmoid(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)