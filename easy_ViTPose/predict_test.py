import os
import cv2
import json
import random
import numpy as np
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

# detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.engine import DefaultPredictor


# VitPose package
from vit_utils.inference import draw_bboxes, pad_image
from configs.ViTPose_common import data_cfg
from sort import Sort
from vit_models.model import ViTPose
from vit_utils.inference import draw_bboxes, pad_image
from vit_utils.top_down_eval import keypoints_from_heatmaps
from vit_utils.util import dyn_model_import, infer_dataset_by_path
from vit_utils.visualization import draw_points_and_skeleton, joints_dict
import torch

# Ignore warnings
import warnings
warnings.filterwarnings('ignore') 

# Setup logger
setup_logger()

Data_Resister_test="test_xworld_kps";
from detectron2.data.datasets import register_coco_instances
from configs.ViTPose_base_coco_256x192 import model as model_cfg

# register_coco_instances(Data_Resister_test,{},'/home/ubuntu/test/dataset/test_xworld.json', Path("/home/ubuntu/test/dataset/test_xworld"))

# dataset_test = DatasetCatalog.get(Data_Resister_test)

# keypoint_names = ["crl_hips__C",
#     "crl_spine__C",
#     "crl_spine01__C",
#     "crl_shoulder__L",
#     "crl_arm__L",
#     "crl_foreArm__L",
#     "crl_hand__L",
#     "crl_handThumb__L",
#     "crl_handThumb01__L",
#     "crl_handThumb02__L",
#     "crl_handThumbEnd__L",
#     "crl_handIndex__L",
#     "crl_handIndex01__L",
#     "crl_handIndex02__L",
#     "crl_handIndexEnd__L",
#     "crl_handMiddle__L",
#     "crl_handMiddle01__L",
#     "crl_handMiddle02__L",
#     "crl_handMiddleEnd__L",
#     "crl_handRing__L",
#     "crl_handRing01__L",
#     "crl_handRing02__L",
#     "crl_handRingEnd__L",
#     "crl_handPinky__L",
#     "crl_handPinky01__L",
#     "crl_handPinky02__L",
#     "crl_handPinkyEnd__L",
#     "crl_neck__C",
#     "crl_Head__C",
#     "crl_eye__L",
#     "crl_eye__R",
#     "crl_shoulder__R",
#     "crl_arm__R",
#     "crl_foreArm__R",
#     "crl_hand__R",
#     "crl_handThumb__R",
#     "crl_handThumb01__R",
#     "crl_handThumb02__R",
#     "crl_handThumbEnd__R",
#     "crl_handIndex__R",
#     "crl_handIndex01__R",
#     "crl_handIndex02__R",
#     "crl_handIndexEnd__R",
#     "crl_handMiddle__R",
#     "crl_handMiddle01__R",
#     "crl_handMiddle02__R",
#     "crl_handMiddleEnd__R",
#     "crl_handRing__R",
#     "crl_handRing01__R",
#     "crl_handRing02__R",
#     "crl_handRingEnd__R",
#     "crl_handPinky__R",
#     "crl_handPinky01__R",
#     "crl_handPinky02__R",
#     "crl_handPinkyEnd__R",
#     "crl_thigh__R",
#     "crl_leg__R",
#     "crl_foot__R",
#     "crl_toe__R",
#     "crl_toeEnd__R",
#     "crl_thigh__L",
#     "crl_leg__L",
#     "crl_foot__L",
#     "crl_toe__L",
#     "crl_toeEnd__L"
#                  ]

# keypoint_flip_map = [("crl_shoulder__L", "crl_shoulder__R"),
#     ("crl_arm__L", "crl_arm__R"),
#     ("crl_foreArm__L", "crl_foreArm__R"),
#     ("crl_hand__L", "crl_hand__R"),
#     ("crl_handThumb__L", "crl_handThumb__R"),
#     ("crl_handThumb01__L", "crl_handThumb01__R"),
#     ("crl_handThumb02__L", "crl_handThumb02__R"),
#     ("crl_handThumbEnd__L", "crl_handThumbEnd__R"),
#     ("crl_handIndex__L", "crl_handIndex__R"),
#     ("crl_handIndex01__L", "crl_handIndex01__R"),
#     ("crl_handIndex02__L", "crl_handIndex02__R"),
#     ("crl_handIndexEnd__L", "crl_handIndexEnd__R"),
#     ("crl_handMiddle__L", "crl_handMiddle__R"),
#     ("crl_handMiddle01__L", "crl_handMiddle01__R"),
#     ("crl_handMiddle02__L", "crl_handMiddle02__R"),
#     ("crl_handMiddleEnd__L", "crl_handMiddleEnd__R"),
#     ("crl_handRing__L", "crl_handRing__R"),
#     ("crl_handRing01__L", "crl_handRing01__R"),
#     ("crl_handRing02__L", "crl_handRing02__R"),
#     ("crl_handRingEnd__L", "crl_handRingEnd__R"),
#     ("crl_handPinky__L", "crl_handPinky__R"),
#     ("crl_handPinky01__L", "crl_handPinky01__R"),
#     ("crl_handPinky02__L", "crl_handPinky02__R"),
#     ("crl_handPinkyEnd__L", "crl_handPinkyEnd__R"),
#     ("crl_eye__L", "crl_eye__R"),
#     ("crl_thigh__L", "crl_thigh__R"),
#     ("crl_leg__L", "crl_leg__R"),
#     ("crl_foot__L", "crl_foot__R"),
#     ("crl_toe__L", "crl_toe__R"),
#     ("crl_toeEnd__L", "crl_toeEnd__R")
#                     ]

# keypoint_connection_rules = [
#     ("crl_eye__L", "crl_Head__C", (0, 255, 128)),
#     ("crl_eye__R", "crl_Head__C", (0, 255, 128)),
#     ("crl_Head__C", "crl_neck__C", (0, 255, 128)),
#     ("crl_neck__C", "crl_spine01__C", (0, 255, 128)),
#     ("crl_spine01__C", "crl_spine__C", (0, 255, 128)),
#     ("crl_spine__C", "crl_hips__C", (0, 255, 128)),

#     ("crl_hips__C", "crl_thigh__R", (0, 128, 255)),
#     ("crl_hips__C", "crl_thigh__L", (255, 128, 0)),
#     ("crl_thigh__R", "crl_leg__R", (0, 128, 255)),
#     ("crl_thigh__L", "crl_leg__L", (255, 128, 0)),
#     ("crl_leg__R", "crl_foot__R", (0, 128, 255)),
#     ("crl_leg__L", "crl_foot__L", (255, 128, 0)),
#     ("crl_foot__R", "crl_toe__R", (0, 128, 255)),
#     ("crl_foot__L", "crl_toe__L", (255, 128, 0)),
#     ("crl_toe__R", "crl_toeEnd__R", (0, 128, 255)),
#     ("crl_toe__L", "crl_toeEnd__L", (255, 128, 0)),


#     ("crl_spine01__C", "crl_shoulder__R", (0, 128, 255)),
#     ("crl_shoulder__R", "crl_arm__R", (0, 128, 255)),
#     ("crl_arm__R", "crl_foreArm__R", (0, 128, 255)),
#     ("crl_foreArm__R", "crl_hand__R", (0, 128, 255)),

#     ("crl_hand__R", "crl_handThumb__R", (0, 128, 255)),
#     ("crl_handThumb__R", "crl_handThumb01__R", (0, 128, 255)),
#     ("crl_handThumb01__R", "crl_handThumb02__R", (0, 128, 255)),
#     ("crl_handThumb02__R", "crl_handThumbEnd__R", (0, 128, 255)),

#     ("crl_hand__R", "crl_handIndex__R", (0, 128, 255)),
#     ("crl_handIndex__R", "crl_handIndex01__R", (0, 128, 255)),
#     ("crl_handIndex01__R", "crl_handIndex02__R", (0, 128, 255)),
#     ("crl_handIndex02__R", "crl_handIndexEnd__R", (0, 128, 255)),

#     ("crl_hand__R", "crl_handMiddle__R", (0, 128, 255)),
#     ("crl_handMiddle__R", "crl_handMiddle01__R", (0, 128, 255)),
#     ("crl_handMiddle01__R", "crl_handMiddle02__R", (0, 128, 255)),
#     ("crl_handMiddle02__R", "crl_handMiddleEnd__R", (0, 128, 255)),

#     ("crl_hand__R", "crl_handRing__R", (0, 128, 255)),
#     ("crl_handRing__R", "crl_handRing01__R", (0, 128, 255)),
#     ("crl_handRing01__R", "crl_handRing02__R", (0, 128, 255)),
#     ("crl_handRing02__R", "crl_handRingEnd__R", (0, 128, 255)),

#     ("crl_hand__R", "crl_handPinky__R", (0, 128, 255)),
#     ("crl_handPinky__R", "crl_handPinky01__R", (0, 128, 255)),
#     ("crl_handPinky01__R", "crl_handPinky02__R", (0, 128, 255)),
#     ("crl_handPinky02__R", "crl_handPinkyEnd__R", (0, 128, 255)),


#     ("crl_spine01__C", "crl_shoulder__L", (255, 128, 0)),
#     ("crl_shoulder__L", "crl_arm__L", (255, 128, 0)),
#     ("crl_arm__L", "crl_foreArm__L", (255, 128, 0)),
#     ("crl_foreArm__L", "crl_hand__L", (255, 128, 0)),
    
#     ("crl_hand__L", "crl_handThumb__L", (255, 128, 0)),
#     ("crl_handThumb__L", "crl_handThumb01__L", (255, 128, 0)),
#     ("crl_handThumb01__L", "crl_handThumb02__L", (255, 128, 0)),
#     ("crl_handThumb02__L", "crl_handThumbEnd__L", (255, 128, 0)),

#     ("crl_hand__L", "crl_handIndex__L", (255, 128, 0)),
#     ("crl_handIndex__L", "crl_handIndex01__L", (255, 128, 0)),
#     ("crl_handIndex01__L", "crl_handIndex02__L", (255, 128, 0)),
#     ("crl_handIndex02__L", "crl_handIndexEnd__L", (255, 128, 0)),

#     ("crl_hand__L", "crl_handMiddle__L", (255, 128, 0)),
#     ("crl_handMiddle__L", "crl_handMiddle01__L", (255, 128, 0)),
#     ("crl_handMiddle01__L", "crl_handMiddle02__L", (255, 128, 0)),
#     ("crl_handMiddle02__L", "crl_handMiddleEnd__L", (255, 128, 0)),

#     ("crl_hand__L", "crl_handRing__L", (255, 128, 0)),
#     ("crl_handRing__L", "crl_handRing01__L", (255, 128, 0)),
#     ("crl_handRing01__L", "crl_handRing02__L", (255, 128, 0)),
#     ("crl_handRing02__L", "crl_handRingEnd__L", (255, 128, 0)),

#     ("crl_hand__L", "crl_handPinky__L", (255, 128, 0)),
#     ("crl_handPinky__L", "crl_handPinky01__L", (255, 128, 0)),
#     ("crl_handPinky01__L", "crl_handPinky02__L", (255, 128, 0)),
#     ("crl_handPinky02__L", "crl_handPinkyEnd__L", (255, 128, 0)),
# ]

# KEYPOINT_OKS_SIGMAS = [
#     0.197, 0.212, 0.298, 0.155, 0.206, 0.364, 0.103, 0.103, 0.114, 0.166, 0.435, 0.101, 0.129,
#     0.085, 0.131, 0.067, 0.077, 0.077, 0.135, 0.080, 0.451, 0.127, 0.137, 0.084, 0.089, 0.058, 
#     0.108, 0.124, 0.153, 0.112, 0.112, 0.156, 0.211, 0.462, 0.108, 0.072, 0.075, 1.00, 0.062, 
#     0.058, 0.139, 0.794, 0.082, 0.056, 0.055, 0.059, 0.053, 0.147, 0.051, 0.066, 0.051, 0.073,
#     0.063, 0.048, 0.050, 0.210, 0.483, 0.548, 0.580, 0.893, 0.205, 0.546, 0.340, 0.175, 0.278
#     ]


# cfg = get_cfg()
# config_name = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml" 
# cfg.merge_from_file(model_zoo.get_config_file(config_name))


# # cfg.MODEL.WEIGHTS ="detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
# cfg.MODEL.WEIGHTS ="/home/ubuntu/test/keypoint/model_final.pth"
# # cfg.MODEL.WEIGHTS ="./YOLOV8/best.pt"

# cfg.MODEL.DEVICE = "cuda:0"

# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 65
# cfg.TEST.KEYPOINT_OKS_SIGMAS = KEYPOINT_OKS_SIGMAS

# cfg.DATALOADER.NUM_WORKERS = 8

# cfg.SOLVER.IMS_PER_BATCH = 8 
# cfg.SOLVER.BASE_LR = 0.01 
# cfg.SOLVER.WARMUP_ITERS = 0
# cfg.SOLVER.MAX_ITER = 5
# cfg.SOLVER.STEPS = (500, 1000) 
# cfg.SOLVER.CHECKPOINT_PERIOD=1

# cfg.TEST.EVAL_PERIOD = 1

# cfg.OUTPUT_DIR = "./keypoint-test"

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# predictor = DefaultPredictor(cfg)
# dataset_dicts = DatasetCatalog.get("test_xworld_kps")

import json

predictions = []

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def pre_img(img):
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, data_cfg['image_size'], interpolation=cv2.INTER_LINEAR) / 255
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return img_input, org_h, org_w

model = "/home/ubuntu/test/easy_ViTPose/easy_ViTPose/runs/train/016/epoch004.pth"
dataset_path = "/home/ubuntu/test/dataset/test_xworld/"


# Load model
vit_pose = ViTPose(model_cfg)
vit_pose.eval()

ckpt = torch.load(model)
if 'state_dict' in ckpt:
    vit_pose.load_state_dict(ckpt['state_dict'])
else:
    vit_pose.load_state_dict(ckpt)
vit_pose.to(torch.device("cuda"))
with open("bbox_test_xworld.json") as jsonfile:
    dataset_dicts = json.load(jsonfile)
    
    for d in dataset_dicts:
        # Load image
        img_path = dataset_path + "0"+ str(d["image_id"]) + ".jpg"
        img = cv2.imread(img_path)
        
        # Make predictions
        # outputs = predictor(img)
        # instances = outputs["instances"].to("cpu")
        
        # Loop through each instance
        # for i in range(len(instances)):
        #     instance = instances[i]
            
        score = d["score"]
        if score < 0.8:
            continue
        bbox = d["bbox"]

        bbox_return = bbox.copy()
        # convert bbox x1y1wh to x1y1x2y2
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]

        ############# Using VitPose to predict keypoints:

        bbox = np.array(bbox).round().astype(int)
        pad_bbox = 10
        bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 0, img.shape[1])
        bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 0, img.shape[0])

        # Crop image and pad to 3/4 aspect ratio
        img_inf = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)
        
        img_input, org_h, org_w = pre_img(img)
        img_input = torch.from_numpy(img_input).to(torch.device("cuda"))

        
        
        heatmaps = vit_pose(img_input).detach().cpu().numpy()
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                                center=np.array([[org_w // 2,
                                                                    org_h // 2]]),
                                                scale=np.array([[org_w, org_h]]),
                                                unbiased=True, use_udp=True)
        keypoints = np.concatenate([points[:, :, ::-1], prob], axis=2)[0]
        
        
        # Transform keypoints to original image
        keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]

        ############# End using VitPose to predict keypoints:



        # Prepare the prediction dictionary
        prediction = {
            "image_id": d["image_id"],
            "bbox": bbox_return,
            "category_id": 1,  # assuming 1 for person in keypoint detection task
            "keypoints": np.array(keypoints).flatten().tolist(),
            "score": score
        }
        # Add the prediction to the list
        predictions.append(prediction)
    
        print(len(predictions))

with open("predictions-model-test.json", "w") as f:
    json.dump(predictions, f)
