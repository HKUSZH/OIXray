import subprocess

from ultralytics import YOLO
import torch

import pickle
import re
#############
import mmcv
# from mmseg.apis import inference_segmentor, init_segmentor
import mmseg.apis
from mmseg.apis import init_model, inference_model
#############

from PIL import Image

import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
# import mmpose_point_pred
import logging
from datetime import datetime

#############
import bestModels_front
from iouHelpers import *
from alignmentHelpers import *
# from subsamplingbestKeyPointModels import *
# from jointsDrawHelper import *

from logisticStacking2024dec8 import *
#############

from skimage import morphology

import time
#############
# from mmengine.registry import init_default_scope
# from mmpose.apis import inference_topdown
# from mmpose.apis import init_model as init_pose_estimator
# from mmpose.evaluation.functional import nms
# from mmpose.structures import merge_data_samples
# from mmdet.apis import inference_detector, init_detector
#############

#############
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_name = f"outputs\\log_skeleton\\skeleton_log_{current_time}.log"

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file_name,
    filemode="w"  # Use "w" to overwrite the file or "a" to append
)

# Log an info message
logging.info("This info message is written to a file.")

segMODELS = ["PSPNet", "UNet", "KNet", "FastSCNN", "DeepLabV3plus", "Segformer"]
#"faster_r_cnn",
pointMODELS = [ "hourglass", "Hrnet", "mobilenet", "rtmpose", "shufflenet"]

# MODELS = ["PSPNet", "UNet", "FastSCNN", "DeepLabV3plus", "Segformer"]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

FOLDS=["fold1", "fold2", "fold3", "fold4"]

checkModelAndConfigs(logging)

def keyPt_Seg_and_Alignment_pred(thisfold, img_folder, gtruthFolder, output_dir, save=False):
    foldNum=thisfold[4]

    theseSegConfigs = bestModels_front.allConfigs[thisfold]
    theseSegModels = bestModels_front.allBestModels[thisfold]

    # Clean up both mmpose and mmseg registries before starting
    import sys
    for module in ['mmpose.registry', 'mmpose.apis', 'mmseg.registry']:
        if module in sys.modules:
            del sys.modules[module]
    
    # Initialize segmentation models
    learnedModels_seg = {}
    for i, modelName_sg in enumerate(segMODELS):
        logging.info(f"handling seg model {i}:{modelName_sg}")
        config_seg = theseSegConfigs[modelName_sg]
        checkpoint_seg = theseSegModels[modelName_sg]

        logging.info(fr"fold {thisfold}, model {i} ({modelName_sg}): \t({config_seg}, {checkpoint_seg})")
        try:
            from mmengine.registry import init_default_scope
            init_default_scope('mmseg')
        except Exception as e:
            print(f"Scope warning (can be ignored): {str(e)}")
            
        model_seg = init_model(config_seg, checkpoint_seg, device='cuda:0')
        learnedModels_seg[modelName_sg] = model_seg

    modelYolo = YOLO(bestModels_front.YOLO_Front_Models[thisfold])
    modelYolo.to(DEVICE)
#######################

    jpeg_files = glob.glob(os.path.join(img_folder, '*.jpg')) + glob.glob(os.path.join(img_folder, '*.jpeg'))
    total_iterations=len(jpeg_files)
    start_time = time.time()
    listGT_Meta=[]
    # listGT_Meta_test = []
    listPred = [[], [], [],[], [], []]
    # listValidate = [[], [], [], [], [], []]
    listShapes = []


    # listIOUs = []
    cntFile = 0
    for eachJPG in tqdm(os.listdir(img_folder)):
        opacity = 0.6

        cntFile+=1
        if eachJPG!="Oi_SURUIKE__3.JPG":
            continue
        # xray_abs_path = os.path.join(img_folder, eachJPG)


        if cntFile<62:
            continue
        # if cntFile > 3:
        #      break
        # if not eachJPG.lower().startswith("Oi".lower()):
        #     print(f"[{eachJPG}] does not start with [Oi] (case-insensitive)")
        #     continue

        if not eachJPG.endswith("JPG"):
            continue
        kptTruthJsonFolderOI = r"D:\Jupyter\Rawdata\Labelme_rawdata\Kpt_Data\Front\OI_patient_n216"
        kptTruthJsonFolderCtrl = r"D:\Jupyter\Rawdata\Labelme_rawdata\Kpt_Data\Front\Control_n216"

        kptTruthFile = re.sub("^(Control_|Oi_)", "", eachJPG.split('.')[0])+ ".json"
        if os.path.exists(os.path.join(kptTruthJsonFolderOI, kptTruthFile)):
            kptTruthFileFullJson=os.path.join(kptTruthJsonFolderOI, kptTruthFile)
        elif os.path.exists(os.path.join(kptTruthJsonFolderCtrl, kptTruthFile)):
            kptTruthFileFullJson = os.path.join(kptTruthJsonFolderCtrl, kptTruthFile)
        else:
            kptTruthFileFullJson = None
            continue

        print(rf"kptTruthFile is: [{kptTruthFile}]")
        print(rf"kptTruthFileFull is: [{kptTruthFileFullJson}]")
        landmarks = read_landmarks_from_json(kptTruthFileFullJson)


        #################
        jpeg_save_path = os.path.join(output_dir, eachJPG.split('.')[0] + "." + thisfold +".pred.JPG")
        jpeg_save_path_individual = os.path.join(output_dir, eachJPG.split('.')[0] + "." + thisfold +".individual_pred.JPG")
        jpeg_save_path_ensemble = os.path.join(output_dir, eachJPG.split('.')[0] + "." + thisfold +".ensemble_pred.JPG")
        jpeg_save_path_indi_and_ensemble = os.path.join(output_dir, eachJPG.split('.')[0] + "." + thisfold +".indiEnsemble_pred.JPG")
        alignment_save_path = os.path.join(output_dir,  eachJPG.split('.')[0] + "." + thisfold +".alignment.pkl")
        mmpose_pickle_path = os.path.join(output_dir,  eachJPG.split('.')[0] + "." + thisfold +".mmpose.pkl")

        # if os.path.exists(jpeg_save_path):
        #     continue

        logging.info(fr"predicting image {cntFile}:\t{eachJPG}")
        xray_abs_path = os.path.join(img_folder, eachJPG)
        logging.info(fr"predicting image (abs path) {cntFile}:\t{img_folder}")
        logging.info(rf"the output image {cntFile} will be saved to:\t{jpeg_save_path}")

        baseName = os.path.basename(xray_abs_path)
        gtFile_name = os.path.splitext(baseName)[0] + ".png"
        gtFile_name = gtruthFolder + "/" + gtFile_name
        gtFile_name = gtFile_name.replace("\\", "/")

        logging.info(f"predicting image (abs path) {cntFile} with truth_file:{gtFile_name}")
        logging.info(f"predicting image (abs path) {cntFile} with truth_file:{os.path.basename(gtFile_name)}")

        if not os.path.exists(gtFile_name):
            logging.warning(f"truth file {gtFile_name} does not exists")
            break

        img_truth = cv2.imread(gtFile_name)
        img_bgr = cv2.imread(xray_abs_path)

        femur_truth = np.zeros(img_truth[:, :, 0].shape)
        femur_truth[np.where(img_truth[:, :, 0] == 1)] = 1
        tibia_truth = np.zeros(img_truth[:, :, 0].shape)
        tibia_truth[np.where(img_truth[:, :, 0] == 2)] = 1
        HEIGHT, WIDTH = img_bgr.shape[0:2]

        #########
        if (not os.path.exists(mmpose_pickle_path)):
            subprocess.run(rf'python mmpose_point_pred.py {thisfold} "{xray_abs_path}" "{mmpose_pickle_path}" {DEVICE}', shell=True,
                       check=True)

        #########
        resultsYolo = modelYolo(xray_abs_path, max_det=2)
        num_bbox = len(resultsYolo[0].boxes.cls)
        bboxes_xyxy = resultsYolo[0].boxes.xyxy.cpu().numpy().astype('uint32')
        bboxes_keypoints = resultsYolo[0].keypoints.data.cpu().numpy().astype('uint32')
        bbox_label = resultsYolo[0].names[0]
        #########
        with open(mmpose_pickle_path, 'rb') as f:  # 'rb' means read binary
            listPredKeypoints, modShapes = pickle.load(f)

        print(listPredKeypoints)
        # modShape = mmpose_point_pred.mmposeDraw(thisfold, xray_abs_path, DEVICE)
        # print(modShape)
        # continue
        #########
        FONTSIZE = img_bgr.shape[0] * 6 / 4000
        YPOS = int(round(img_bgr.shape[0] * 15 / 400))
        YPOS0 = int(YPOS*0)
        YPOS1 = int(YPOS*1)
        YPOS2 = int(round(YPOS * 2))
        YPOS3 = int(round(YPOS * 3))
        YPOS4 = int(round(YPOS * 4))

        img_bgr2 = img_bgr.copy()
        img_bgr2[:YPOS2, :, :] =0
        text_params = {
            'fontFace': cv2.FONT_HERSHEY_TRIPLEX,
            'fontScale': FONTSIZE,
            'color': (255, 255, 255),
            'thickness': 3,
            'lineType': cv2.LINE_AA
        }
        text_paramsOrig = text_params.copy()
        text_paramsOrig['color'] = (0, 255, 0)
        #########
        truth_mask_255 = np.zeros(img_truth.shape, dtype='uint8')
        truth_mask_255[img_truth[:, :, 0] == 1] = palette_dict[1]  # Femur color
        truth_mask_255[img_truth[:, :, 0] == 2] = palette_dict[2]  # Tibia color
        #pred_viz_truth = cv2.addWeighted(img_bgr, opacity, truth_mask_255, 1 - opacity, 0.0).astype('uint8')
        pred_viz_truth=truth_mask_255
        cv2.putText(pred_viz_truth, "truth mask", (20, YPOS2), **text_params)
        ############
        # listGroundTruths.append(img_truth[:, :, 0])

        listGT_Meta.append(img_truth[:, :, 0])
        # if (selected[cntFile-1] == 1):
        #     listGT_Meta_train.append(img_truth[:, :, 0])
        # else:
        #     listGT_Meta_test.append(img_truth[:, :, 0])

        masks=[]
        masksStacked_femur = np.zeros(img_bgr[:, :, 0].shape)#np.array([])
        masksStacked_tibia = np.zeros(img_bgr[:, :, 0].shape)
        listPred_Femur = []
        listPred_Tibia = []
        for i, modelName_sg in enumerate(segMODELS):
            logging.info(f"handling seg model {i}:{modelName_sg}")
            modeli = learnedModels_seg[modelName_sg]
            
            try:
                # Ensure mmseg scope before inference
                from mmengine.registry import init_default_scope
                init_default_scope('mmseg')
            except Exception as e:
                print(f"Scope warning (can be ignored): {str(e)}")
            
            resulti = inference_model(modeli, img_bgr)
            pred_mask_i = resulti.pred_sem_seg.data[0].cpu().numpy()
            table, iouFT = compute_ious(pred_mask_i, img_truth[:, :, 0], [1, 2], fileNum=cntFile, modelName=modelName_sg,
                                        fold=thisfold, metric="IOU", isChosen=1)
            # table, iouFT= compute_ious(pred_mask, img_truth[:, :, 0], [1, 2], fileNum=cntFile, modelName=modelNamei, fold=thisfold, metric="IOU", isChosen=1)
            # logging.info("\n" + table.get_string())

            listPred[i].append(pred_mask_i)

            femur_i = np.zeros(pred_mask_i.shape)
            tibia_i = np.zeros(pred_mask_i.shape)

            femur_i[np.where(pred_mask_i == 1)] = 1
            tibia_i[np.where(pred_mask_i == 2)] = 1

            listPred_Femur.append(femur_i)
            listPred_Tibia.append(tibia_i)
            if i==0:
                masksStacked_femur = femur_i.copy()
                masksStacked_tibia = tibia_i.copy()
            else:
                masksStacked_femur += femur_i
                masksStacked_tibia += tibia_i
            ############
            pred_mask_bgr = np.zeros(img_bgr.shape)
            for idx in palette_dict.keys():
                pred_mask_bgr[np.where(pred_mask_i == idx)] = palette_dict[idx]
            pred_mask_bgr = pred_mask_bgr.astype('uint8')
            pred_viz_individual = cv2.addWeighted(img_bgr2, opacity, pred_mask_bgr, 1 - opacity, 0.0)


            # print(f"YPOS:{YPOS}")
            # cv2.putText(pred_viz_individual, thisfold, [20, YPOS], **text_params)
            cv2.putText(pred_viz_individual, modelName_sg, [20, YPOS1], **text_params)
            fiou = "fIOU:" + str(iouFT[1])
            tiou = "tIOU:" + str(iouFT[2])

            logging.info(f"{thisfold}: femur fIOU for model {i} in file {cntFile}:{modelName_sg} is:{iouFT[1]}")
            logging.info(f"{thisfold}: tibia tIOU for model {i} in file {cntFile}:{modelName_sg} is:{iouFT[2]}")

            cv2.putText(pred_viz_individual, fiou, [20, YPOS2], **text_params)
            cv2.putText(pred_viz_individual, tiou, [20, YPOS3], **text_params)
            if i == 0:
                new_imageIndividual = np.concatenate((img_bgr2, pred_viz_truth, pred_viz_individual), axis=1)
                cv2.putText(new_imageIndividual, "original", [20, YPOS1], **text_paramsOrig)
            else:
                new_imageIndividual = np.concatenate((new_imageIndividual, pred_viz_individual), axis=1)
            ############

        predStacks = []
        num_thresholds = round(len(segMODELS)/2)

        femur_stacker_pred = predict_with_stacker("logistic_regression_model.Femur."+thisfold+".pkl", listPred_Femur)
        tibia_stacker_pred = predict_with_stacker("logistic_regression_model.Tibia."+thisfold+".pkl", listPred_Tibia)
        tab, iou_stackerF = compute_ious(femur_stacker_pred, femur_truth, [1],  fileNum = cntFile, modelName = "stacking", fold = thisfold, metric = "IOU", isChosen = 1)
        tab, iou_stackerT = compute_ious(tibia_stacker_pred, tibia_truth, [1],  fileNum = cntFile, modelName = "stacking", fold = thisfold, metric = "IOU", isChosen = 1)

        pred_mask_Stacker = np.zeros(femur_stacker_pred.shape)
        pred_mask_Stacker[np.where(femur_stacker_pred == 1)] = 1
        pred_mask_Stacker[np.where(tibia_stacker_pred == 1)] = 2

        pred_mask_Stacker_255 = np.zeros((femur_stacker_pred.shape[0], femur_stacker_pred.shape[1], 3), dtype='uint8')
        pred_mask_Stacker_255[femur_stacker_pred == 1] = palette_dict[1]  # Femur color
        pred_mask_Stacker_255[tibia_stacker_pred == 1] = palette_dict[2]  # Tibia color
        pred_viz_Stacker = cv2.addWeighted(img_bgr2, opacity, pred_mask_Stacker_255, 1 - opacity, 0.0).astype('uint8')
        # Add text annotations
        cv2.putText(pred_viz_Stacker, "stacking", (20, YPOS1), **text_params)
        cv2.putText(pred_viz_Stacker, f"fIOU:{iou_stackerF[1]}", (20, YPOS2), **text_params)
        cv2.putText(pred_viz_Stacker, f"tIOU:{iou_stackerT[1]}", (20, YPOS3), **text_params)

        ############
        # for i in [num_thresholds]:
        for i in range(6):
            # Create binary mask for metrics
            pred_mask_ensemb_12 = np.zeros(masksStacked_femur.shape)
            pred_mask_ensemb_12[np.where(masksStacked_femur > i)] = 1
            pred_mask_ensemb_12[np.where(masksStacked_tibia > i)] = 2
            predStacks.append(pred_mask_ensemb_12)

            # Calculate metrics
            threshold_txt = f">{i}"
            table, iou_ensemb = compute_ious(
                pred_mask_ensemb_12, 
                img_truth[:, :, 0], 
                [1, 2], 
                fileNum=cntFile, 
                modelName=threshold_txt,
                fold=thisfold, 
                metric="IOU", 
                isChosen=1
            )
            logging.info("\n" + table.get_string())
            
            # Create visualization
            # Convert single-channel mask to 3-channel for blending
            pred_mask_ensemb_255 = np.zeros((pred_mask_ensemb_12.shape[0], pred_mask_ensemb_12.shape[1], 3), dtype='uint8')
            pred_mask_ensemb_255[pred_mask_ensemb_12 == 1] = palette_dict[1]  # Femur color
            pred_mask_ensemb_255[pred_mask_ensemb_12 == 2] = palette_dict[2]  # Tibia color
            
            # Create segmentation visualization
            pred_viz_majority = cv2.addWeighted(img_bgr2, opacity, pred_mask_ensemb_255, 1 - opacity, 0.0).astype('uint8')
            
            # Add text annotations
            cv2.putText(pred_viz_majority, threshold_txt, (20, YPOS1), **text_params)
            cv2.putText(pred_viz_majority, f"fIOU:{iou_ensemb[1]}", (20, YPOS2), **text_params)
            cv2.putText(pred_viz_majority, f"tIOU:{iou_ensemb[2]}", (20, YPOS3), **text_params)
            
            # Log results
            logging.info(f"{thisfold}: femur fIOU for ensemble {threshold_txt} in file {cntFile} is:{iou_ensemb[1]}")
            logging.info(f"{thisfold}: tibia tIOU for ensemble {threshold_txt} in file {cntFile} is:{iou_ensemb[2]}")


            if i == 0:
                new_imageEnsemb = pred_viz_majority #np.concatenate((pred_viz_Stacker, pred_viz_majority), axis=1)
                # cv2.putText(new_imageEnsemb, "original", [20, YPOS], cv2.FONT_HERSHEY_DUPLEX, FONTSIZE, (0, 255, 0), 3,
                #             cv2.LINE_AA)
            else:
                new_imageEnsemb = np.concatenate((new_imageEnsemb, pred_viz_majority), axis=1)
            # Create final visualization
            # new_image2 = np.concatenate((img_bgr, pred_viz_majority, pred_viz_seg_yolo, pred_viz_seg_yolo_tibialines, pred_viz_seg_yolo_femurlines, pred_viz_seg_yolo_lines_femur2), axis=1)
            # cv2.putText(new_image2, "original", (20, YPOS), **text_params)

        # logiModel = logisticsStackingForOne(predStacks, img_truth[:, :, 0], [1,2], fileNum=cntFile,
        #                                 fold=thisfold)

################

        # pred_viz_seg_yolo_tibialines = pred_viz_seg_yolo.copy()
        # pred_viz_seg_yolo_femurlines = pred_viz_seg_yolo_tibialines.copy()
        dictAverageKpts = getDictAvrgCenter(listPredKeypoints, bboxes_keypoints, WIDTH)
        pred_viz_Stacker_pruned = pruningSeg(pred_mask_Stacker, dictAverageKpts, img_bgr2, img_truth, drawBox=False)
        pred_viz_seg_yolo = pred_viz_Stacker_pruned.copy()

        yoloDraw(num_bbox, bboxes_xyxy, bboxes_keypoints, bbox_label, pred_viz_seg_yolo)
        yoloPredKeypoints = [num_bbox, bboxes_xyxy, bboxes_keypoints, bbox_label]
        dictAverageKpts2 = mmposeDraw(pred_viz_seg_yolo, bboxes_keypoints, listPredKeypoints, landmarks)

        pred_viz_joints = pred_viz_Stacker_pruned.copy()
        pred_viz_joints = jointsDraw(pred_viz_joints, bboxes_keypoints, listPredKeypoints, landmarks)
        # calculate_average_joints(listPredKeypoints)
        # listShapes.append(modShape)
        # continue
        pred_viz_seg_yolo_tibialines = pred_viz_Stacker_pruned.copy()
        pred_viz_seg_yolo_femurlines = pred_viz_seg_yolo_tibialines.copy()
        tibiaAlignments = TibiaLine(pred_viz_seg_yolo_tibialines, bboxes_keypoints, pred_mask_Stacker)
        femurAlignments, pred_viz_seg_yolo_lines_femur2 = FemurLines(pred_viz_seg_yolo_femurlines, bboxes_keypoints,
                                                                     pred_mask_Stacker, dictAverageKpts)
        ################
        if not alignment_save_path is None:
            with open(alignment_save_path, 'wb') as fout:
                pickle.dump([tibiaAlignments, femurAlignments, xray_abs_path, dictAverageKpts,
                             resultsYolo, yoloPredKeypoints, listPredKeypoints, kptTruthFileFullJson, landmarks], fout)

        new_imageAlignments = np.concatenate((img_bgr2, pred_viz_Stacker, pred_viz_seg_yolo, pred_viz_joints, pred_viz_seg_yolo_tibialines,
                                     pred_viz_seg_yolo_femurlines, pred_viz_seg_yolo_lines_femur2), axis=1)
        cv2.putText(new_imageAlignments, "original", (20, YPOS1), **text_paramsOrig)
        if save:
            cv2.imwrite(jpeg_save_path_individual, new_imageIndividual)
            print(f"done saving {cntFile} for {thisfold}:{jpeg_save_path_individual}")
        if save:
            new_imageEnsemb2 = np.concatenate((new_imageEnsemb, pred_viz_Stacker, pred_viz_Stacker_pruned), axis=1)
            cv2.imwrite(jpeg_save_path_ensemble, new_imageEnsemb2)
            print(f"done saving {cntFile} for {thisfold}:{jpeg_save_path_ensemble}")
        if save:
            new_image_indi_and_Ensemb = np.concatenate((new_imageIndividual, new_imageEnsemb2), axis=0)
            cv2.imwrite(jpeg_save_path_indi_and_ensemble, new_image_indi_and_Ensemb)
            print(f"done saving {cntFile} for {thisfold}:{jpeg_save_path_indi_and_ensemble}")
        if save:
            cv2.imwrite(jpeg_save_path, new_imageAlignments)
            print(f"done saving {cntFile} for {thisfold}:{jpeg_save_path}")
        ################
        elapsed_time = time.time() - start_time
        completed = cntFile + 1
        remaining_iterations = total_iterations - completed
        eta = (elapsed_time / completed) * remaining_iterations if completed > 0 else 0

        print(f"ETA for {thisfold}: {eta:.2f} seconds")
    print(listShapes)

    # logiModel = logisticsStackingForMany(listPred, listGT_Meta, thisfold)


# Initialize the model
# model = init_model(config_file, checkpoint_file, device='cuda:0')
# model = init_segmentor(config_file, checkpoint_file, device='cuda:0')  # Change to 'cpu' if needed
if __name__ == '__main__':

    dictFOLDER = dict(fold1="Front_val1_train234", fold2="Front_val2_train134", fold3="Front_val3_train124",
                      fold4="Front_val4_train123")
    for fold in ["fold1", "fold2", "fold4", "fold3"]:
        image_folder = os.path.join("data", dictFOLDER[fold], "img_dir", "val")
        truthFolder = os.path.join("data", dictFOLDER[fold], "ann_dir", "val")

        logging.info(f"handling {fold} with image_path:{image_folder}")
        logging.info(f"handling {fold} with truthFolder:{truthFolder}")

        output_dir = os.path.join(r"outputs\pred-2024-dec","alignment.dec20."+fold)
        logging.info(f"the output folder is:{output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # cnt+=1
        # if cnt>5:
        #     break

        # break
        keyPt_Seg_and_Alignment_pred(fold, image_folder, truthFolder, output_dir, save=True)


