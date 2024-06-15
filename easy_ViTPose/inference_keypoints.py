import cv2
from inference import VitInference

# Image to run inference RGB format
img = cv2.imread('./020000002323.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# set is_video=True to enable tracking in video inference
# be sure to use VitInference.reset() function to reset the tracker after each video
# There are a few flags that allows to customize VitInference, be sure to check the class definition
model_path = './runs/train/007/epoch004.pth'
yolo_path = './yolov8s.pt'

# If you want to use MPS (on new macbooks) use the torch checkpoints for both ViTPose and Yolo
# If device is None will try to use cuda -> mps -> cpu (otherwise specify 'cpu', 'mps' or 'cuda')
# dataset and det_class parameters can be inferred from the ckpt name, but you can specify them.
model = VitInference(model_path, yolo_path, model_name='b', yolo_size=320, is_video=False, device=None, dataset="coco_65")

# Infer keypoints, output is a dict where keys are person ids and values are keypoints (np.ndarray (25, 3): (y, x, score))
# If is_video=True the IDs will be consistent among the ordered video frames.
keypoints, bboxes = model.inference(img)

# call model.reset() after each video

# img = model.draw(show_yolo=True)  # Returns RGB image with drawings
# cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); cv2.waitKey(0)

# print(keypoints)

# print(bboxes)