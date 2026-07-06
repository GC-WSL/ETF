_base_ = [
    '_base_/models/simclip_vitb.py',
    '_base_/datasets/vaihingen_wsss.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_10k.py'
]
_NAMES=('road surface and pavement', 'building and housetop',
        'low vegetation and grassground', 
        'tree and trunk','car and vehicle in road')
_NAMES_MS=('Impervious Surfaces: Hard surfaces and Non-permeable surfaces and Solid ground and Pavement and Asphalt and Concrete surfaces',
            'Building: Structure and Construction and Edifice and Residential building and Commercial building and House and House-like structure and Skyscraper',
            'Low Vegetation: Grassland and Shrubs and Ground cover and Sparse vegetation and Low-lying plants and Herbaceous vegetation',
            'Tree: Large plant and Tall vegetation and Woodland tree and Deciduous tree and Evergreen tree and Forest tree',
            'Car: Vehicle and Automobile and Sedan and Passenger car and Motor vehicle and Road vehicle')
randomness=dict(seed=0,diff_rank_seed=False)
# norm_cfg = dict(type='SyncBN', requires_grad=True)

custom_imports = dict(imports=['simclip'], allow_failed_imports=False)
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
        class_names =_NAMES,
        class_names_ms =_NAMES_MS,
        data_preprocessor=data_preprocessor,
        label_txt='../datasets/Vaihingen_256/train.txt',
        context_length=63,
        backbone=dict(
            input_resolution=256,
            output_dim=512,
            out_indices=[3, 5, 7, 11],
        ),
        multi_scale_crossattn =True,
        multi_class_context=True, 
        feature_stride = 4,
        cross_mask = True,
        
        text_encoder=dict(
            context_length=70,
        ),
        context_decoder =dict(
            type='EffiContextDecoder',
            context_length=77,
            transformer_layers=3,
            visual_dim=512),
        neck=None,
        identity_head=dict(erodil = True,seeding=True,bg=True,kernel_size=11),
        decode_head = None,
        train_cfg=dict(),                 
        test_cfg=dict(mode='slide', 
                    #   save_context=True,
                      crop_size=(256, 256), stride=(171, 171))
        )
find_unused_parameters = True

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01), 
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        })
                                        )

train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=500)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500,
                    published_keys=['meta', 'state_dict'],save_best='mIoU',rule='greater',max_keep_ckpts=2),)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='crf_pseudo_ome'),))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     dataset=dict(
#         data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),))
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])