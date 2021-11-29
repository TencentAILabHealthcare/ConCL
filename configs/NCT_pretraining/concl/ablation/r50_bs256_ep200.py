_base_ = '../r18_bs256_ep200.py'
# model settings
model = dict(
    queue_len=16384,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[2,3,4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=False)
    )