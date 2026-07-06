# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
out_indices=[3, 5, 7, 11]
model = dict(
    type='SimCLIP',
    data_preprocessor=data_preprocessor,
    pretrained='pretrained/ViT-B-16.pt',
    context_length=5,
    label_txt='../datasets/Potsdam_pd/train.txt',
    class_names = (),
    class_names_ms = None,
    multi_class_context = False,
    multi_scale_crossattn = False,

    backbone=dict(
        type='CLIPVisionTransformer',
        patch_size=16,
        width=768,
        output_dim=512,
        out_indices=[3, 5, 7, 11],
        get_embeddings=True,
        get_embed_proj=False,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=512,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        context_length=16,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        if_decouple=True,
        # outdim=512,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[768+21, 768+21, 768+21, 768+21],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
            type='ATMHead',
            img_size=512,
            in_channels=256,
            channels=256,
            embed_dims=(256) // 2,
            num_heads=4,
            num_classes=16,
            use_stages=len(out_indices),
            loss_decode=dict(
                type='ATMLoss', num_classes=16, dec_layers=len(out_indices), loss_weight=1.0),
        ),
    identity_head=dict(
        type='IdentityHead',
        in_channels=1,
        channels=1,
        num_classes=1,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        clip_finetune = None,
    ),
    test_cfg=dict(mode='slide',
                  save_context = False,
                  save_infer = False,
                  crop_size=(512, 512), stride=(341, 341)),
)