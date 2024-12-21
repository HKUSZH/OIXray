import mmcv
# from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.apis import init_model, inference_model

import numpy as np
import cv2
import glob
import os
from tqdm import tqdm

import logging
from datetime import datetime
import bestModels_front
from iouHelpers import *
from logisticStacking import *

import time

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_name = f"outputs\\log_{current_time}.log"

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file_name,
    filemode="w"  # Use "w" to overwrite the file or "a" to append
)

# Log an info message
logging.info("This info message is written to a file.")

MODELS = ["PSPNet", "UNet", "KNet", "FastSCNN", "DeepLabV3plus", "Segformer"]
# MODELS = ["PSPNet", "UNet", "FastSCNN", "DeepLabV3plus", "Segformer"]

FOLDS=["fold1", "fold2", "fold3", "fold4"]
cntMissing=0
for fold in FOLDS:
    for model in MODELS:
        thisConfig=bestModels.allConfigs[fold][model]
        thisModel = bestModels.allBestModels[fold][model]
        if not os.path.exists(thisConfig):
            logging.info(f"{thisConfig} does not exist")
            cntMissing+=1
        if not os.path.exists(thisModel):
            logging.info(f"{thisModel} does not exist")
            cntMissing += 1
logging.info(f"{cntMissing} model or config files are missing")


def predict_unlab(thisfold, img_folder, gtruthFolder, output_dir, save=False):
    foldNum=thisfold[4]

    theseConfig = bestModels.allConfigs[thisfold]
    theseModels = bestModels.allBestModels[thisfold]
    learnedModels= {}
    for i, modelNamei in enumerate(MODELS):
        logging.info(f"handling model {i}:{modelNamei}")
        config_file = theseConfig[modelNamei]
        checkpoint_file = theseModels[modelNamei]

        logging.info(fr"fold {thisfold}, model {i} ({modelNamei}): \t({config_file}, {checkpoint_file})")
        model = init_model(config_file, checkpoint_file, device='cuda:0')
        learnedModels[modelNamei]=model

    jpeg_files = glob.glob(os.path.join(img_folder, '*.jpg')) + glob.glob(os.path.join(img_folder, '*.jpeg'))
    total_iterations=len(jpeg_files)
    start_time = time.time()
    listGT_Meta_train=[]
    listGT_Meta_test = []
    listPred = [[], [], [],[], [], []]
    listValidate = [[], [], [], [], [], []]

    np.random.seed(42)
    selected = np.random.choice([0, 1], size=len(jpeg_files), p=[0.25, 0.75])
    # listIOUs = []
    cntFile = 0
    for eachJPG in tqdm(os.listdir(img_folder)):
        cntFile+=1
        # if cntFile > 20:
        #      break
        if not eachJPG.endswith("JPG"):
            continue



        save_path = os.path.join(output_dir, 'new.pred-' + eachJPG.split('.')[0] + thisfold + ".JPG")

        logging.info(fr"predicting image {cntFile}:\t{eachJPG}")
        xray_abs_path = os.path.join(img_folder, eachJPG)
        logging.info(fr"predicting image (abs path) {cntFile}:\t{img_folder}")
        logging.info(rf"the output image {cntFile} will be saved to:\t{save_path}")


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
        # gt_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # gt_mask2 = cv2.imread(label_path)
        # configs=[config_file_PSP1, config_file_UNET1, config_file_KNET1, config_file_FASTSCNN1, config_file_DLabV1, config_file_SegF1]
        # checkpoints=[checkpoint_file_PSP1, checkpoint_file_UNET1, checkpoint_file_KNET1, checkpoint_file_FASTSCNN1, checkpoint_file_DLabV1, checkpoint_file_SegF1]
        # configsNames = ["frPSP2", "frUNET2", "frKNET2", "frFASTSCNN2", "frDLabV2", "frSegFormer2"]

        # listGroundTruths.append(img_truth[:, :, 0])

        if (selected[cntFile-1] == 1):
            listGT_Meta_train.append(img_truth[:, :, 0])
        else:
            listGT_Meta_test.append(img_truth[:, :, 0])

        masks=[]
        masksStacked_femur = np.array([])
        masksStacked_tibia = np.array([])
        for i, modelNamei in enumerate(MODELS):
            logging.info(f"handling model {i}:{modelNamei}")
            config_file = theseConfig[modelNamei]
            checkpoint_file = theseModels[modelNamei]


            logging.info(f"fold {thisfold}, model {i} ({modelNamei}): ({config_file}, {checkpoint_file})")
            modeli = learnedModels[modelNamei] #init_model(config_file, checkpoint_file, device='cuda:0')

            resulti = inference_model(modeli, img_bgr)
            pred_mask = resulti.pred_sem_seg.data[0].cpu().numpy()
            table, iouFT= compute_ious(pred_mask, img_truth[:, :, 0], [1, 2], fileNum=cntFile, modelName=modelNamei, fold=thisfold, metric="IOU", isChosen=selected[cntFile-1])
            logging.info("\n" + table.get_string())

            if (selected[cntFile - 1] == 1):
                listPred[i].append(pred_mask)
            else:
                listValidate[i].append(pred_mask)

            femur = np.zeros(pred_mask.shape)
            tibia = np.zeros(pred_mask.shape)
            femur[np.where(pred_mask == 1)] = 1
            tibia[np.where(pred_mask == 2)] = 1

            if i==0:
                masksStacked_femur = femur
                masksStacked_tibia = tibia
            else:
                masksStacked_femur += femur
                masksStacked_tibia += tibia

            pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
            masks.append(pred_mask)

            for idx in palette_dict.keys():
                pred_mask_bgr[np.where(pred_mask == idx)] = palette_dict[idx]
            pred_mask_bgr = pred_mask_bgr.astype('uint8')

            # 叠加可视化效果
            pred_viz = cv2.addWeighted(img_bgr, opacity, pred_mask_bgr, 1 - opacity, 0.0)
            FONTSIZE=img_bgr.shape[0]*4/4000
            YPOS = int(round(img_bgr.shape[0] * 10 / 400))
            YPOS2 = int(round(YPOS * 2))
            YPOS3 = int(round(YPOS * 3))
            YPOS4 = int(round(YPOS * 4))
            # print(f"YPOS:{YPOS}")
            cv2.putText(pred_viz, thisfold, [20, YPOS], cv2.FONT_HERSHEY_TRIPLEX, FONTSIZE, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(pred_viz, modelNamei, [20, YPOS2], cv2.FONT_HERSHEY_TRIPLEX, FONTSIZE, (255, 255, 255), 3, cv2.LINE_AA)
            fiou = "fIOU:" + str(iouFT[1])
            tiou = "tIOU:" + str(iouFT[2])

            logging.info(f"{thisfold}: femur fIOU for model {i} in file {cntFile}:{modelNamei} is:{iouFT[1]}")
            logging.info(f"{thisfold}: tibia tIOU for model {i} in file {cntFile}:{modelNamei} is:{iouFT[2]}")

            cv2.putText(pred_viz, fiou,[20, YPOS3], cv2.FONT_HERSHEY_TRIPLEX, FONTSIZE, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(pred_viz, tiou, [20, YPOS4], cv2.FONT_HERSHEY_TRIPLEX, FONTSIZE, (255, 255, 255), 3, cv2.LINE_AA)
            if i==0:
                new_image1 = np.concatenate((img_bgr, pred_viz), axis=1)
                cv2.putText(new_image1, "original", [20, YPOS], cv2.FONT_HERSHEY_TRIPLEX, FONTSIZE, ( 0, 255,0), 3, cv2.LINE_AA)
            else:
                new_image1 = np.concatenate((new_image1, pred_viz), axis=1)

        predStacks=[]
        for i in range(len(MODELS)):
            pred_mask_ensemb_i = np.zeros((masksStacked_femur.shape[0], masksStacked_femur.shape[1], 3))
            pred_mask_ensemb_i[np.where(masksStacked_femur > i)] = palette_dict[1]
            pred_mask_ensemb_i[np.where(masksStacked_tibia > i)] = palette_dict[2]
            pred_mask_ensemb_i = pred_mask_ensemb_i.astype('uint8')

            pred_mask_ensemb_12 = np.zeros(masksStacked_femur.shape)
            pred_mask_ensemb_12[np.where(masksStacked_femur > i)] = 1
            pred_mask_ensemb_12[np.where(masksStacked_tibia > i)] = 2

            predStacks.append(pred_mask_ensemb_12)
            # if(selected[cntFile-1]==1):
            #     listPred[i].append(pred_mask_ensemb_12)
            # else:
            #     listValidate[i].append(pred_mask_ensemb_12)
            # iouEnsembi = compute_ious(pred_mask_ensemb_12, img_truth[:, :, 0], [1, 2])
            thresholdTxt=">"+str(i)
            table, iouEnsembi = compute_ious(pred_mask_ensemb_12, img_truth[:, :, 0], [1, 2], fileNum=cntFile, modelName=thresholdTxt,
                                        fold=thisfold, metric="IOU", isChosen=selected[cntFile-1])
            logging.info("\n" + table.get_string())

            pred_viz = cv2.addWeighted(img_bgr, opacity, pred_mask_ensemb_i, 1 - opacity, 0.0)
            FONTSIZE = img_bgr.shape[0] * 4 / 4000
            YPOS = int(round(img_bgr.shape[0] * 10 / 400))
            YPOS2 = int(round(YPOS * 2))
            YPOS3 = int(round(YPOS * 3))
            # print(f"YPOS:{YPOS}")

            cv2.putText(pred_viz, thresholdTxt,
                        [20, YPOS], cv2.FONT_HERSHEY_TRIPLEX, FONTSIZE, (255, 255, 255), 3,
                        cv2.LINE_AA)
            fiou = "fIOU:" + str(iouEnsembi[1])
            tiou = "tIOU:" + str(iouEnsembi[2])
            logging.info(f"{thisfold}: femur fIOU for ensemble {thresholdTxt}  in file {cntFile} is:{iouEnsembi[1]}")
            logging.info(f"{thisfold}: tibia tIOU for ensemble {thresholdTxt}  in file {cntFile} is:{iouEnsembi[2]}")

            cv2.putText(pred_viz, fiou,
                        [20, YPOS2], cv2.FONT_HERSHEY_TRIPLEX, FONTSIZE, (255, 255, 255), 3,
                        cv2.LINE_AA)
            cv2.putText(pred_viz, tiou,
                        [20, YPOS3], cv2.FONT_HERSHEY_TRIPLEX, FONTSIZE, (255, 255, 255), 3,
                        cv2.LINE_AA)

            if i == 0:
                new_image2 = np.concatenate((img_bgr, pred_viz), axis=1)
                cv2.putText(new_image2, "original", [20, YPOS], cv2.FONT_HERSHEY_TRIPLEX, FONTSIZE, (0, 255, 0), 3,
                            cv2.LINE_AA)
            else:
                new_image2 = np.concatenate((new_image2, pred_viz), axis=1)

        new_image12 = np.concatenate((new_image1, new_image2), axis=0)

        # logiModel = logisticsStackingForOne(predStacks, img_truth[:, :, 0], [1,2], fileNum=cntFile,
        #                                 fold=thisfold)

        if save:
            cv2.imwrite(save_path, new_image12)
            print(f"done saving {cntFile} for {thisfold}:{save_path}")

        elapsed_time = time.time() - start_time
        completed = cntFile + 1
        remaining_iterations = total_iterations - completed
        eta = (elapsed_time / completed) * remaining_iterations if completed > 0 else 0

        print(f"ETA for {thisfold}: {eta:.2f} seconds")

    logiModel = logisticsStackingForMany(listPred, listGT_Meta_train, listValidate, listGT_Meta_test, thisfold, choice=selected)


# Initialize the model
# model = init_model(config_file, checkpoint_file, device='cuda:0')
# model = init_segmentor(config_file, checkpoint_file, device='cuda:0')  # Change to 'cpu' if needed
if __name__ == '__main__':


    # image_path = r"E:\mmseg-pkchen\mmsegmentation\inputs\Val1_train234\img_dir\val"
    # image_path = r"testing\img_dir"


    # label_dir = r"E:\mmseg-pkchen\mmsegmentation\inputs\Val1_train234\ann_dir\val"
    dictFOLDER = dict(fold1="Front_val1_train234", fold2="Front_val2_train134", fold3="Front_val3_train124",
                      fold4="Front_val4_train123")
    for fold in FOLDS:
        image_folder = os.path.join("data", dictFOLDER[fold], "img_dir", "val")
        truthFolder = os.path.join("data", dictFOLDER[fold], "ann_dir", "val")

        logging.info(f"handling {fold} with image_path:{image_folder}")
        logging.info(f"handling {fold} with truthFolder:{truthFolder}")

        output_dir = os.path.join(r"outputs\pred-2024-dec",fold)
        logging.info(f"the output folder is:{output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # cnt+=1
        # if cnt>5:
        #     break




        # break
        predict_unlab(fold, image_folder, truthFolder, output_dir, save=False)


