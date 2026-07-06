_base_ = [
    '_base_/models/simclip_vitb.py',
    '_base_/datasets/iSAID1x_wsss.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_10k.py'
]
iSAID_NAMES=('background', 'ship', 'store tank', 'baseball diamond',
                 'tennis court', 'basketball court', 'Ground Track Field',
                 'Bridge', 'Large Vehicle', 'Small Vehicle', 'Helicopter',
                 'Swimming pool', 'Roundabout', 'Soccer ball field', 'plane',
                 'Harbor')
iSAID_NAMES_MS=('background,river,parking lot,parking apron', 
'Ship: vessel and boat and cargo ship and tanker and freighter',
'Store Tank: storage tank and fuel tank and water tank and industrial tank and silo',
'Baseball Diamond: baseball field and baseball pitch and ballpark and sports diamond',
'Tennis Court: tennis field and tennis surface and tennis area and tennis grounds',
'Basketball Court: basketball field and basketball arena and hoop court and basketball playground',
'Ground Track Field: athletic field and running track and sports field and track and field',
'Bridge: overpass and span and viaduct and causeway and overcrossing',
'Large Vehicle: truck and lorry and bus and heavy vehicle and transport vehicle',
'Small Vehicle: car and sedan and hatchback and compact vehicle and passenger vehicle',
'Helicopter: chopper and rotorcraft and copter and helicopter aircraft',
'Swimming Pool: pool and swimming area and water pool and aquatic facility',
'Roundabout: traffic circle and rotary and traffic ring and intersection',
'Soccer Ball Field: football field and soccer pitch and football pitch and soccer ground',
'Plane: airplane and aircraft and jet and airliner and flying machine',
'Harbor: port and dock and marina and seaport and quay')

randomness=dict(seed=0,diff_rank_seed=False)
# norm_cfg = dict(type='SyncBN', requires_grad=True)
custom_imports = dict(imports=['simclip'], allow_failed_imports=False)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
        class_names =iSAID_NAMES,
        class_names_ms =iSAID_NAMES_MS,
        data_preprocessor=data_preprocessor,
        label_txt='../datasets/iSAID_512_sampled_2/train.txt',
        context_length=63,
        backbone=dict(
            input_resolution=512,
            output_dim=512,
            out_indices=[3, 5, 7, 10],
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
        identity_head=dict(erodil = True,seeding=True,kernel_size=7),
        decode_head = None,
        train_cfg=dict(),                 
        test_cfg=dict(mode='slide', 
                    #   save_context=True,
                      crop_size=(512, 512), stride=(341, 341))
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

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000,
                    published_keys=['meta', 'state_dict'],save_best='mIoU',rule='greater',max_keep_ckpts=3),)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='crf_pseudo_ctfa'),))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     dataset=dict(
#         data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_prefix=dict(img_path=f'img_dir/test', seg_map_path=f'ann_dir/test'),))
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])