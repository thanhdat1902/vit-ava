from .ViTPose_common import *

# Channel configuration
channel_cfg = dict(
    num_output_channels=65,
    dataset_joints=65,
    dataset_channel=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64], ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64])

# Set models channels
data_cfg['num_output_channels'] = channel_cfg['num_output_channels']
data_cfg['num_joints']= channel_cfg['dataset_joints']
data_cfg['dataset_channel']= channel_cfg['dataset_channel']
data_cfg['inference_channel']= channel_cfg['inference_channel']

names = ['small', 'base', 'large', 'huge']
for name in names:
    globals()[f'model_{name}']['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
