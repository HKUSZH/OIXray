import cv2
import pickle

import mmdet.apis
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
# %matplotlib inline
import torch
import time

import os
import bestModels_front
from tqdm import tqdm

import mmseg.apis
from mmseg.apis import init_model, inference_model
import mmcv
##############
from mmengine.registry import init_default_scope
from mmpose.evaluation.functional import nms
from mmdet.apis import inference_detector, init_detector

from mmpose.structures import merge_data_samples
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
##############
pointMODELS = [ "hourglass", "Hrnet", "mobilenet", "rtmpose", "shufflenet"]

def mmpose_pt_pred(thisfold, img_path2, kptFile, device):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    # Now import mmpose modules
    detectorFile = f"model_detector_files_{thisfold}.pkl"
    start = time.time()
    if(os.path.exists(detectorFile)):
        print(rf"file [{detectorFile}] exists, loading detector file")
        with open(detectorFile, 'rb') as f:  # 'rb' means read binary
            detector = pickle.load(f)
    else:        # mmdet.apis.init_detector()
        detector = mmdet.apis.init_detector(
            bestModels_front.mmpose_configs[thisfold]["faster_r_cnn"],
            bestModels_front.mmpose_models[thisfold]["faster_r_cnn"],
            device=device
        )
        with open(detectorFile, 'wb') as f:
            pickle.dump(detector, f)
    CONF_THRES = 0.3
    end = time.time()
    print(f"Process took {end - start:.4f} seconds")
    # Perform detection
    init_default_scope(detector.cfg.get('default_scope', 'mmdet'))
    detect_result = inference_detector(detector, img_path2)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > CONF_THRES)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4].astype('int')
    print(bboxes)
    #
    modShapes = np.zeros([5, 3])
    listkeypoints=dict()
    for i, ptModel in enumerate(pointMODELS):
        pose_estimator = init_pose_estimator(
            bestModels_front.mmpose_configs[thisfold][ptModel],
            bestModels_front.mmpose_models[thisfold][ptModel],
            device=device,
            cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
        )

        # Perform pose estimation
        pose_results = inference_topdown(pose_estimator, img_path2, bboxes)
        data_samples = merge_data_samples(pose_results)

        keypoints = data_samples.pred_instances.keypoints.astype('int')
        listkeypoints[ptModel] = keypoints
        modShapes[i, :] = np.array(keypoints.shape)
        print(f"          * * *          ")
        print(f"handling model:{ptModel}, with predicted key points:")
        print(keypoints)

        # listkeypoints
        # break
    # Save to binary file
    with open(kptFile, 'wb') as f:  # 'wb' means write binary
        pickle.dump((listkeypoints, modShapes), f)
    print(modShapes)
    # return modShapes

import argparse  # Add this at the top with other imports

def parse_args():
    parser = argparse.ArgumentParser(description='Process three input strings')
    parser.add_argument('string1', type=str, help='First input string')
    parser.add_argument('string2', type=str, help='Second input string')
    parser.add_argument('string3', type=str, help='Third input string')
    parser.add_argument('string4', type=str, help='Third input string')
    return parser.parse_args()

def main(input_string=None):
    if input_string is None:
        args = parse_args()
        fold = args.string1
        xray_abs_path = args.string2
        keypoints_path = args.string3
        DEVICE = args.string4
    else:
        print(f"Received input_string: {input_string}")  # For debugging
        fold, xray_abs_path, keypoints_path, DEVICE = input_string.split(" ")

    # Use the directly provided string when debugging
    print(f"Received string 1: {fold}")  # For debugging
    print(f"Received string 2: {xray_abs_path}")  # For debugging
    print(f"Received string 3: {keypoints_path}")  # For debugging
    print(f"Received string 4: {DEVICE}")  # For debugging
    # Use the strings as needed
    modShapes = mmpose_pt_pred(fold, xray_abs_path, keypoints_path, DEVICE)


    return fold

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     FOLDS = ["fold1", "fold2", "fold3", "fold4"]
#     DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     # label_dir = r"E:\mmseg-pkchen\mmsegmentation\inputs\Val1_train234\ann_dir\val"
#     dictFOLDER = dict(fold1="Front_val1_train234", fold2="Front_val2_train134", fold3="Front_val3_train124",
#                       fold4="Front_val4_train123")
#     for fold in FOLDS:
#         image_folder = os.path.join("data", dictFOLDER[fold], "img_dir", "val")
#         truthFolder = os.path.join("data", dictFOLDER[fold], "ann_dir", "val")
#
#         listShapes=[]
#         for eachJPG in tqdm(os.listdir(image_folder)):
#             xray_abs_path = os.path.join(image_folder, eachJPG)
#             modShape = mmposeDraw(fold, xray_abs_path, DEVICE)
#             listShapes.append(modShape)
#
#         print(listShapes)
