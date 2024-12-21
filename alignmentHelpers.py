import os
import cv2
import numpy as np
from scipy.optimize import minimize
import bestModels_front
import pickle
######################
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector
######################
from iouHelpers import *
from prettytable import PrettyTable

FOLDS=["fold1", "fold2", "fold3", "fold4"]
segMODELS = ["PSPNet", "UNet", "KNet", "FastSCNN", "DeepLabV3plus", "Segformer"]
pointMODELS = ["faster_r_cnn", "hourglass", "Hrnet", "mobilenet", "rtmpose", "shufflenet"]
KOLORS = dict(hourglass=(200, 0, 200),
              Hrnet=(0, 0, 0),
              mobilenet=(0, 200, 200),
              rtmpose=(42, 42, 165),
              yolo=(255, 42, 165),
              shufflenet=(0, 0, 200))
KOLORjson = dict(left=[0, 0, 0], right=[255, 255, 255])
def checkModelAndConfigs(logging):
    cntMissing1 = 0
    for fold in FOLDS:
        for model in segMODELS:
            thisConfig = bestModels_front.allConfigs[fold][model]
            thisModel = bestModels_front.allBestModels[fold][model]
            if not os.path.exists(thisConfig):
                logging.info(f"{thisConfig} does not exist")
                cntMissing1 += 1
            if not os.path.exists(thisModel):
                logging.info(f"{thisModel} does not exist")
                cntMissing1 += 1
        cntMissing2 = 0
        for model in pointMODELS:
            thisConfig = bestModels_front.mmpose_configs[fold][model]
            thisModel = bestModels_front.mmpose_models[fold][model]
            if not os.path.exists(thisConfig):
                logging.info(f"{thisConfig} does not exist")
                cntMissing2 += 1
            if not os.path.exists(thisModel):
                logging.info(f"{thisModel} does not exist")
                cntMissing2 += 1
    print(f"{cntMissing1} seg model or config files are missing")
    print(f"{cntMissing2} point model or config files are missing")
    logging.info(f"{cntMissing1} seg model or config files are missing")
    logging.info(f"{cntMissing2} point model or config files are missing")

####
####
def distance_to_line(x_i, y_i, k, x0, y0):
    """
    Calculate perpendicular distance from point (x_i, y_i) to a line
    that passes through (x0, y0) with slope k
    """
    # For near-vertical lines, use horizontal distance
    if abs(k) > 1e3:
        return abs(x_i - x0)
    
    # For other lines: use point-line distance formula
    # Line equation: y - y0 = k(x - x0)
    numerator = abs(k * (x_i - x0) - (y_i - y0))
    denominator = np.sqrt(1 + k**2)
    return numerator / denominator


# Clip the points to ensure they stay within the image
def clip_point(x, y, width, height):
    return max(0, min(x, width - 1)), max(0, min(y, height - 1))
def clip_line(x1, y1, x2, y2, width, height):
    # Ensure all points are within the image boundaries
    x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
    x2, y2 = max(0, min(width - 1, x2)), max(0, min(height - 1, y2))
    return x1, y1, x2, y2
def TibiaLine(pred_viz_seg_yolo_tibialines, bboxes_keypoints, pred_mask_ensemb_12):
    HEIGHT = pred_mask_ensemb_12.shape[0]
    WIDTH = pred_mask_ensemb_12.shape[1]

    femurPoints = np.where(pred_mask_ensemb_12 == 1)
    tibiaPoints = np.where(pred_mask_ensemb_12 == 2)

    numTibiaPts = len(tibiaPoints[0])
    downSamplSiz = 1000
    np.random.seed(42)
    indTibia = np.random.randint(0, numTibiaPts, size=downSamplSiz)

    # Create subsampled points array using the random indices
    tibia_subsampl = [
        tibiaPoints[0][indTibia],  # y coordinates
        tibiaPoints[1][indTibia]   # x coordinates
    ]

    # Visualize subsampled points
    for i in range(downSamplSiz):
        cv2.circle(pred_viz_seg_yolo_tibialines,
                  (int(tibia_subsampl[1][i]), int(tibia_subsampl[0][i])),  # (x, y)
                  radius=1,
                  color=(0, 0, 0),  # black color
                  thickness=-1)  # Filled circle

    femurTop = np.min(femurPoints[0])
    femurLow = np.max(femurPoints[0])

    tibiaTop = np.min(tibiaPoints[0])
    tibiaTopx =tibiaPoints[1][np.argmin(tibiaPoints[0])]
    tibiaLow = np.max(tibiaPoints[0])
    HEIGHT20 = int(HEIGHT * 0.1)

    tibiaAlignments={}
    if len(bboxes_keypoints) == 2:
        midPoint = (np.mean(bboxes_keypoints[0], axis=0)[0] + np.mean(bboxes_keypoints[1], axis=0)[0]) / 2
        
        leftTibiaX = np.array([tibia_subsampl[1][kk] for kk in np.where(tibia_subsampl[1] < midPoint)[0]])
        leftTibiaY = np.array([tibia_subsampl[0][kk] for kk in np.where(tibia_subsampl[1] < midPoint)[0]])
        rightTibiaX = np.array([tibia_subsampl[1][kk] for kk in np.where(tibia_subsampl[1] > midPoint)[0]])
        rightTibiaY = np.array([tibia_subsampl[0][kk] for kk in np.where(tibia_subsampl[1] > midPoint)[0]])

        sides=dict(left=[leftTibiaX, leftTibiaY], right=[rightTibiaX, rightTibiaY])
        for side in sides:
            data_points =np.array(sides[side]).T
            sideTibiaX = data_points[:, 0] #sides[side][0]
            sideTibiaY = data_points[:, 1]# sides[side][1]

            if (len(sideTibiaX)<10) | (len(sideTibiaY)<10):
                continue

            m_alt, c_alt = np.polyfit(sideTibiaY, sideTibiaX, 1)  # Swapping x and y
            y1, x1 = max(0, tibiaTop - HEIGHT20), int(m_alt * max(0, tibiaTop - HEIGHT20) + c_alt)  # rt at x = 0
            y2, x2 = min(HEIGHT, tibiaLow + HEIGHT20), int(
                m_alt * min(HEIGHT, tibiaLow + HEIGHT20) + c_alt)  # End at x = image width
            # x1, y1 = clip_point(x1, y1, WIDTH, HEIGHT)
            # x2, y2 = clip_point(x2, y2, WIDTH, HEIGHT)

            print(f"Slope (m_alt): {m_alt}")
            print(f"Intercept (c_alt): {c_alt}")

            color = (255, 0, 255)  # Magenta color
            thickness = 1  # Line thickness
            cv2.line(pred_viz_seg_yolo_tibialines, (x1, y1), (x2, y2), color, thickness=2)

            tibiaProxPt = [tibiaTopx, tibiaTop]  # Fixed point for second line

            distances_to_p2 = []
            for point in np.array(sides[side]).T:
                x_i, y_i = point
                dist_to_p2 = np.sqrt((x_i - tibiaProxPt[0]) ** 2 + (y_i - tibiaProxPt[1]) ** 2)
                distances_to_p2.append(dist_to_p2)

            # Convert to numpy arrays for easier manipulation
            distances_to_p2 = np.array(distances_to_p2)

            # Get thresholds for point selection
            pUpper_threshold = np.percentile(distances_to_p2, 55)  # Top 60% closest to p2
            pLower_threshold = np.percentile(distances_to_p2, 60)  #

            # Create masks for point selection
            pUpper_points_mask = distances_to_p2 <= pUpper_threshold
            pLower_points_mask = distances_to_p2 > pLower_threshold

            upperTibiaX, upperTibiaY = getFemurPoints(data_points, pUpper_points_mask, img_obj=pred_viz_seg_yolo_tibialines,
                                                    KOL=(255, 0, 0))  # magenta
            lowerTibiaX, lowerTibiaY = getFemurPoints(data_points, pLower_points_mask,
                                                      img_obj=pred_viz_seg_yolo_tibialines,
                                                      KOL=(255, 0, 0))  # magenta
            kUpper, interceptUpper = np.polyfit(upperTibiaY, upperTibiaX, 1)  # Swapping x and y
            y1, x1 = max(0, tibiaTop - HEIGHT20), int(kUpper * max(0, tibiaTop - HEIGHT20) + interceptUpper)  # rt at x = 0
            y2, x2 = min(HEIGHT, tibiaLow), int(
                kUpper * min(HEIGHT, tibiaLow) + interceptUpper)  # End at x = image width
            cv2.line(pred_viz_seg_yolo_tibialines, (x1, y1), (x2, y2), [0,0,0], thickness=2)

            kLower, interceptLower = np.polyfit(lowerTibiaY, lowerTibiaX, 1)  # Swapping x and y
            y1, x1 = max(0, tibiaTop), int(kLower * max(0, tibiaTop) + interceptLower)  # rt at x = 0
            y2, x2 = min(HEIGHT, tibiaLow + HEIGHT20), int(
                kLower * min(HEIGHT, tibiaLow + HEIGHT20) + interceptLower)  # End at x = image width
            cv2.line(pred_viz_seg_yolo_tibialines, (x1, y1), (x2, y2), [255, 255, 255], thickness=2)

            intersectionTibia,  degTibia= draw_intersection_point(pred_viz_seg_yolo_tibialines,
                                                   kUpper, interceptUpper,
                                                   kLower, interceptLower,
                                                   WIDTH, HEIGHT, side, AngleName="Tibia")
            tibiaAlignment=dict(intersectionTibia=intersectionTibia,
                                 degTibia=degTibia)

            tibiaAlignments[side]=tibiaAlignment

    return tibiaAlignments

def FemurLines(pred_viz_seg_yolo_lines_femur, bboxes_keypoints, pred_mask_ensemb_12, dictAvrgKpt):
    pred_viz_seg_yolo_lines_femur2 = pred_viz_seg_yolo_lines_femur.copy()
    HEIGHT = pred_mask_ensemb_12.shape[0]
    femurPoints = np.where(pred_mask_ensemb_12 == 1)
    tibiaPoints = np.where(pred_mask_ensemb_12 == 2)

    numFemurPts = len(femurPoints[0])
    downSamplSiz = 4000
    np.random.seed(42)
    indFemur = np.random.randint(0, numFemurPts, size=downSamplSiz)

    # Create subsampled points array using the random indices
    femur_subsampl = [
        femurPoints[0][indFemur],  # y coordinates
        femurPoints[1][indFemur]  # x coordinates
    ]

    # Visualize subsampled points
    for i in range(downSamplSiz):
        cv2.circle(pred_viz_seg_yolo_lines_femur,
                   (int(femur_subsampl[1][i]), int(femur_subsampl[0][i])),  # (x, y)
                   radius=1,
                   color=(0, 0, 0),  # black color
                   thickness=-1)  # Filled circle


    if len(bboxes_keypoints) == 2:
        midPoint = (np.mean(bboxes_keypoints[0], axis=0)[0] + np.mean(bboxes_keypoints[1], axis=0)[0]) / 2
        print(f"midPoint: {midPoint}")
        print(bboxes_keypoints)

        xyFemurTupleLeft = np.array([[fx for i, fx in enumerate(femurPoints[1]) if femurPoints[1][i]<midPoint],
                                     [fy for i, fy in enumerate(femurPoints[0]) if femurPoints[1][i]<midPoint]])
        xyFemurTupleRight = np.array([[fx for i, fx in enumerate(femurPoints[1]) if femurPoints[1][i]>midPoint],
                                     [fy for i, fy in enumerate(femurPoints[0]) if femurPoints[1][i]>midPoint]])

        femurs=dict(left=xyFemurTupleLeft, right=xyFemurTupleRight)
        # if len(bboxes_keypoints) == 1:
        #     midPoint = pred_mask_ensemb_12.shape[1]
        #     xyFemurTupleLeft = np.array([[fx for i, fx in enumerate(femurPoints[1]) if femurPoints[1][i] < midPoint],
        #                                  [fy for i, fy in enumerate(femurPoints[0]) if femurPoints[1][i] < midPoint]])
        #     femurs = dict(left=xyFemurTupleLeft)

        femurAlignments={}
        for side in femurs.keys():
            thisFemur = femurs[side]
            if(len(thisFemur[1])<10):
                continue
            femurTop = np.min(thisFemur[1])
            femurLow = np.max(thisFemur[1])

            femurLow2 = max(0, int(femurLow - (femurLow - femurTop) * 0.1))

            if (np.mean(bboxes_keypoints[1], axis=0)[0] < midPoint):
                rectSides = dict(left=1, right=0)
            else:
                rectSides = dict(left=0, right=1)

            # proxCentrY_side = bboxes_keypoints[rectSides[side]][0, 1]
            # proxCentrX_side = bboxes_keypoints[rectSides[side]][0, 0]
            print(dictAvrgKpt[side])
            proxCentrY_side = int(dictAvrgKpt[side]['hip'][1])
            proxCentrX_side = int(dictAvrgKpt[side]['hip'][0])

            distFemurX_side = [thisFemur[0][i] for i, fy in enumerate(thisFemur[1]) if (fy>femurLow2 and fy<femurLow) ]
            distFemurY_side = [fy for i, fy in enumerate(thisFemur[1]) if (fy > femurLow2 and fy < femurLow)]

            if(len(distFemurX_side)==0 or len(distFemurY_side)==0):
                continue
            # distCentrY_side = int(np.mean(distFemurY_side))
            # distCentrX_side= int(np.mean(distFemurX_side))
            distCentrY_side = int(dictAvrgKpt[side]['knee'][1])
            distCentrX_side = int(dictAvrgKpt[side]['knee'][0])

            # tibiaTop = np.min(np.where(tibia > 0)[0])
            # tibiaLow = np.max(np.where(tibia > 0)[0])
            # HEIGHT20 = int(HEIGHT * 0.1)

            pred_viz_seg_yolo_lines_femur = cv2.drawMarker(pred_viz_seg_yolo_lines_femur, (distCentrX_side, distCentrY_side), [0, 0, 255], markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
            pred_viz_seg_yolo_lines_femur = cv2.drawMarker(pred_viz_seg_yolo_lines_femur,(proxCentrX_side, proxCentrY_side), [0, 0, 255],  markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
            print(f"proxCentrX_side, proxCentrY_side {proxCentrX_side}, {proxCentrY_side}")
            # pred_viz_seg_yolo_lines_femur = cv2.drawMarker(pred_viz_seg_yolo_lines_femur, (distCentrX_right, distCentrY_right), [255, 255, 0], markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)

            # axisFemur = np.where(skeletonFemur > 0)
            if side == "left":
                axisFemur_side_y = [femur_subsampl[0][i] for i,fy in enumerate(femur_subsampl[0]) if femur_subsampl[1][i] < midPoint]
                axisFemur_side_x = [femur_subsampl[1][i] for i,fy in enumerate(femur_subsampl[0]) if femur_subsampl[1][i] < midPoint]
            elif side == "right":
                axisFemur_side_y = [femur_subsampl[0][i] for i, fy in enumerate(femur_subsampl[0]) if femur_subsampl[1][i] > midPoint]
                axisFemur_side_x = [femur_subsampl[1][i] for i, fy in enumerate(femur_subsampl[0]) if femur_subsampl[1][i] > midPoint]

            femurAlignment=fitFemurLines([proxCentrX_side, proxCentrY_side],
                          [distCentrX_side, distCentrY_side],
                          np.array((axisFemur_side_x, axisFemur_side_y)).T,
                          pred_viz_seg_yolo_lines_femur2, pos=side)

            femurAlignments[side]=femurAlignment

        return [femurAlignments, pred_viz_seg_yolo_lines_femur2]
    return [None, pred_viz_seg_yolo_lines_femur2]

def drawFittedLine(pred_viz_seg_yolo_lines_femur, slope1, intercept1, height, KOL=(0, 255, 0), THICK=3):
    y1, x1 = 0, int(intercept1)  # rt at x = 0
    y2, x2 = height, int(slope1 * height + intercept1)  # End at x = image width
    # Calculate and draw first line (femoral head - green)
    cv2.line(pred_viz_seg_yolo_lines_femur,
             (x1, y1), (x2, y2),
             KOL, THICK)  # Green for first line



def getFemurPoints(data_points, flagVec, img_obj, KOL):
    distFemurX=[]
    distFemurY=[]
    for point in data_points[flagVec]:
        distFemurX.append(point[0])
        distFemurY.append(point[1])
        cv2.circle(img_obj,
                       (int(point[0]), int(point[1])),
                       radius=2,
                       color=KOL,  # Blue for line 2
                       thickness=-1)
    return distFemurX, distFemurY
def fitFemurLines(proxCentrXY, distCentrXY, oneFemur, pred_viz_seg_yolo_lines_femur2, pos="left"):
    height, width = pred_viz_seg_yolo_lines_femur2.shape[:2]
    data_points = oneFemur
    p1 = proxCentrXY  # Fixed point for first line
    p2 = distCentrXY  # Fixed point for second line

    # Calculate initial distances to both fixed points
    distances_to_p1 = []
    distances_to_p2 = []
    for point in data_points:
        x_i, y_i = point
        # Use Euclidean distance for point selection
        dist_to_p1 = np.sqrt((x_i - p1[0])**2 + (y_i - p1[1])**2)
        dist_to_p2 = np.sqrt((x_i - p2[0])**2 + (y_i - p2[1])**2)
        distances_to_p1.append(dist_to_p1)
        distances_to_p2.append(dist_to_p2)

    # Convert to numpy arrays for easier manipulation
    distances_to_p1 = np.array(distances_to_p1)
    distances_to_p2 = np.array(distances_to_p2)

    # Get thresholds for point selection
    p1_threshold = np.percentile(distances_to_p1, 15)  # Top 10% closest to p1
    p2_threshold = np.percentile(distances_to_p2, 55)  # Top 60% closest to p2
    p3_threshold = np.percentile(distances_to_p2, 50)  #
    p4_threshold = np.percentile(distances_to_p2, 80)

    # Create masks for point selection
    p1_points_mask = distances_to_p1 <= p1_threshold
    p2_points_mask = distances_to_p2 <= p2_threshold
    p3_points_mask = (distances_to_p2 > p3_threshold) & (distances_to_p2 <= p4_threshold)

    competing_points_mask = ~(p1_points_mask | p2_points_mask)

    proxFemurX, proxFemurY = getFemurPoints(data_points, p1_points_mask, img_obj=pred_viz_seg_yolo_lines_femur2, KOL=(255, 0, 255)) #magenta
    upperFemurX, upperFemurY = getFemurPoints(data_points, p3_points_mask, img_obj=pred_viz_seg_yolo_lines_femur2, KOL=(255, 255, 0)) #cyan
    distFemurX, distFemurY = getFemurPoints(data_points, p2_points_mask, img_obj=pred_viz_seg_yolo_lines_femur2, KOL=(255, 0, 0)) #blue

    slope_lowerShaft, intercept_lowerShaft = np.polyfit(distFemurY, distFemurX, 1)
    drawFittedLine(pred_viz_seg_yolo_lines_femur2, slope_lowerShaft, intercept_lowerShaft, height, KOL=[255, 0, 0], THICK=3) #Blue

    slope_upperShaft, intercept_upperShaft = np.polyfit(upperFemurY, upperFemurX, 1)
    drawFittedLine(pred_viz_seg_yolo_lines_femur2, slope_upperShaft, intercept_upperShaft, height, KOL=[255, 0, 255], THICK=3) #magenta

    slope1, intercept1 = np.polyfit(proxFemurY, proxFemurX, 1)
    drawFittedLine(pred_viz_seg_yolo_lines_femur2, slope1, intercept1, height, KOL=[0, 255, 255], THICK=1)

    # Fit y = mx + b and draw directly with cv2.line
    slope2, intercept2 = np.polyfit(proxFemurX, proxFemurY, 1)
    x1, x2 = 0, width  # Use full width of image
    y1 = int(slope2 * x1 + intercept2)
    y2 = int(slope2 * x2 + intercept2)
    cv2.line(pred_viz_seg_yolo_lines_femur2, (x1, y1), (x2, y2), [0, 0, 255], thickness=1)  # Red, thickness=3

    slope_ransac, intercept_ransac, inliers_ransac = fit_line_ransac(proxFemurX, proxFemurY, img=pred_viz_seg_yolo_lines_femur2, residual_threshold=5.0  )

    slope_pca, intercept_pca = fit_line_pca(proxFemurX, proxFemurY, img_pca=pred_viz_seg_yolo_lines_femur2)

    angle_rad_NSA1 = np.arctan2(slope_pca, 1) - np.arctan2(slope_upperShaft, 1)  # Since we fitted x = my + b
    angle_rad_NSA2 = np.arctan2(1/slope_pca, 1) - np.arctan2(1/slope_upperShaft, 1)  # Since we fitted x = my + b
    angle_rad_NSA3 = np.arctan2(1/slope_pca, 1) - np.arctan2(slope_upperShaft, 1)  # Since we fitted x = my + b
    angle_rad_NSA4 = np.arctan2(slope_pca, 1) - np.arctan2(1/slope_upperShaft, 1)  # Since we fitted x = my + b

    print(f"angle_rad_NSA1: {angle_rad_NSA1}")
    print(f"angle_rad_NSA1 (deg): {np.degrees(angle_rad_NSA1)}")
    print(f"angle_rad_NSA2: {angle_rad_NSA2}")
    print(f"angle_rad_NSA2 (deg): {np.degrees(angle_rad_NSA2)}")
    print(f"angle_rad_NSA3: {angle_rad_NSA3}")
    print(f"angle_rad_NSA3 (deg): {np.degrees(angle_rad_NSA3)}")
    print(f"angle_rad_NSA4: {angle_rad_NSA4}")
    print(f"angle_rad_NSA4 (deg): {np.degrees(angle_rad_NSA4)}")
    intersectionFemur, angle_deg = draw_intersection_point(pred_viz_seg_yolo_lines_femur2,
                                           1/slope_pca, -intercept_pca/slope_pca,
                                           slope_upperShaft, intercept_upperShaft,
                                           width, height, pos, "NSA")
    intersectionFemurCora, angle_FemurCora = draw_intersection_point(pred_viz_seg_yolo_lines_femur2,
                                                           slope_lowerShaft, intercept_lowerShaft,
                                                           slope_upperShaft, intercept_upperShaft,
                                                           width, height, pos, "Femur")
    femurAlignments=dict(intersectionNSA=intersectionFemur,
                 NSA=angle_deg,
                 slope_pca=slope_pca,
                 intercept_pca=intercept_pca,
                 slope_upperShaft=slope_upperShaft,
                 intercept_upperShaft=intercept_upperShaft,
                 slope_lowerShaft=slope_lowerShaft,
                 intercept_lowerShaft=intercept_lowerShaft,
                 intersectionFemurCora=intersectionFemurCora,
                 angle_FemurCora=angle_FemurCora,
                 ransac=[slope_ransac, intercept_ransac, inliers_ransac]
                 )

    return femurAlignments

    def modified_objective(params, data_points, p1, p2, p1_mask, p2_mask, competing_mask):
        k1, k2 = params
        total_distance = 0

        # Points assigned to line 1
        for point in data_points[p1_mask]:
            dist = distance_to_line(point[0], point[1], k1, p1[0], p1[1])
            total_distance += dist

        # Points assigned to line 2
        for point in data_points[p2_mask]:
            dist = distance_to_line(point[0], point[1], k2, p2[0], p2[1])
            total_distance += dist

        # Competing points - assign to nearest line
        for point in data_points[competing_mask]:
            dist1 = distance_to_line(point[0], point[1], k1, p1[0], p1[1])
            dist2 = distance_to_line(point[0], point[1], k2, p2[0], p2[1])
            total_distance += min(dist1, dist2)

        return total_distance

    # Initial optimization to determine point assignments
    initial_guesses = [
        [0.5, 10.0],    # Moderate slopes
        [-0.5, 10.0],   # One steeper
        [0.5, -10.0],  # Moderate slopes
        [-0.5, -10.0],
        [0.5, 100.0],  # Very different slopes
        [-0.5, 100.0],  #
    ]

    best_result = None
    best_score = float('inf')

    for i, guess in enumerate(initial_guesses):
        result = minimize(modified_objective, guess,
                         args=(data_points, p1, p2, p1_points_mask, p2_points_mask, competing_points_mask),
                         method='Nelder-Mead')
        k1, k2 = result.x
        drawFittedLine(pred_viz_seg_yolo_lines_femur2, 1 / k1, (p1[0] - p1[1] / k1), height, KOL=[0, 0, 0], THICK=1)
        drawFittedLine(pred_viz_seg_yolo_lines_femur2, 1 / k2, (p2[0] - p2[1] / k2), height, KOL=[255, 255, 255], THICK=1)

        if result.fun < best_score:
            best_score = result.fun
            best_result = result

    initial_slopes = best_result.x
    k1, k2 = initial_slopes

    # Determine final point assignments
    femoralHeadX = []
    femoralHeadY = []
    FemurX = []
    FemurY = []

    for i, point in enumerate(data_points):
        x_i, y_i = point
        if p1_points_mask[i]:  # Points pre-assigned to femoral head
            femoralHeadX.append(x_i)
            femoralHeadY.append(y_i)
        elif p2_points_mask[i]:  # Points pre-assigned to femur
            FemurX.append(x_i)
            FemurY.append(y_i)
        else:  # Competing points - assign based on distance
            dist1 = distance_to_line(x_i, y_i, k1, p1[0], p1[1])
            dist2 = distance_to_line(x_i, y_i, k2, p2[0], p2[1])
            if dist1 < dist2:
                femoralHeadX.append(x_i)
                femoralHeadY.append(y_i)
            else:
                FemurX.append(x_i)
                FemurY.append(y_i)

    # Refit each line using only its assigned points
    slope1, intercept1 = np.polyfit(femoralHeadY, femoralHeadX, 1)
    slope2, intercept2 = np.polyfit(FemurY, FemurX, 1)

    # Color points based on their final assignment
    for point in zip(femoralHeadX, femoralHeadY):
        cv2.circle(pred_viz_seg_yolo_lines_femur2,
                  (int(point[0]), int(point[1])),
                  radius=4,
                  color=(0, 255, 0),  # Green for line 1
                  thickness=-1)

    for point in zip(FemurX, FemurY):
        cv2.circle(pred_viz_seg_yolo_lines_femur2,
                  (int(point[0]), int(point[1])),
                  radius=2,
                  color=(255, 0, 0),  # Blue for line 2
                  thickness=-1)

    drawFittedLine(pred_viz_seg_yolo_lines_femur2, slope1, intercept1, height, KOL=[0, 0, 255], THICK=1) # x=my+c
    drawFittedLine(pred_viz_seg_yolo_lines_femur2, slope2, intercept2, height, KOL=[0, 255, 0], THICK=3) #y=mx + c

    # Calculate and draw second line (femur shaft - blue)
    # y1, x1 = 0, int(intercept2)  # rt at x = 0
    # y2, x2 = height, int(slope2 * height + intercept2)  # End at x = image width
    #
    # cv2.line(pred_viz_seg_yolo_lines_femur,
    #          (x1, y1), (x2, y2),
    #          (255, 0, 0), 2)  # Blue for second line

    print(f"Initial slopes - k1: {k1:.2f}, k2: {k2:.2f}")
    print(f"Final slopes - k1: {slope1:.2f}, k2: {slope2:.2f}")

    # Calculate angle between lines
    angle_rad = np.arctan2(slope2, 1) - np.arctan2(slope1, 1)  # Since we fitted x = my + b


    # Draw the angle value on the image
    # text_x = int(width * 0.1)  # 10% from left
    # text_y = int(height * 0.1)  # 10% from top
    intersection, angle_deg = draw_intersection_point(pred_viz_seg_yolo_lines_femur2,
                                           slope1, intercept1,
                                           slope2, intercept2,
                                           width, height,
                                           pos)


    return [slope1, slope2, np.angle(angle_rad)]


def jointsDraw(pred_viz_joints, yolo_keypoints, listkeypoints:dict, landmarks:dict):
    HEIGHT = pred_viz_joints.shape[0]
    WIDTH = pred_viz_joints.shape[1]
    RADIUS = int(np.ceil(HEIGHT / 500))
    FONTSIZE = HEIGHT / 2000
    text_params = {
        'fontFace': cv2.FONT_HERSHEY_DUPLEX,
        'fontScale': FONTSIZE,
        'color': (0, 0, 0),
        'thickness': 2,
        'lineType': cv2.LINE_AA
    }

    # Define target box dimensions
    box_height = HEIGHT // 3
    box_width = WIDTH // 2
    
    # Create zoom_img with same dimensions as original
    zoom_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Calculate region size maintaining aspect ratio
    zoom_factor = 4 #box_height / region_height  # Use height for zoom factor
    region_height = box_height // zoom_factor #min(box_height, box_width) // 4
    region_width = box_width // zoom_factor  #int(region_height * (WIDTH / HEIGHT))

    # Setup for keypoints
    listkeypoints6 = listkeypoints.copy()
    # yolo_keypoints2=yolo_keypoints.copy()
    # yolopts = yolo_keypoints2[:, :, 0:2]
    # m1= np.mean(yolopts[0], axis=0)[0]
    # m2= np.mean(yolopts[1], axis=0)[0]
    # if m1>m2:
    #     tmp=yolopts[0].copy()
    #     yolopts[0]=yolopts[1]
    #     yolopts[1]=tmp
    #
    # listkeypoints6["yolo"] = yolopts
    listkeypoints6["yolo"] = yolo_keypoints[:, :, 0:2].copy()
    X, modelSrc, centerNames, clusterLabels, kMeansCenters, joint_mapping, centerName = calculate_average_joints_by_Kmeans(listkeypoints6, WIDTH)
    lowerCenterName = [s.lower() for s in centerNames]

    
    # Define colors and markers (same as mmposeDraw)
    MARKERS = dict(hourglass=cv2.MARKER_DIAMOND,
                  Hrnet=cv2.MARKER_STAR,
                  mobilenet=cv2.MARKER_CROSS,
                  rtmpose=cv2.MARKER_SQUARE,
                  shufflenet=cv2.MARKER_TRIANGLE_UP)

    # Border parameters
    border_thickness = max(2, int(zoom_factor/4))
    border_color = (255, 255, 255)
    
    # Process each side and joint
    for side_idx, side in enumerate(['left', 'right']):
        for joint_idx, joint in enumerate(['hip', 'knee', 'ankle']):
            center_idx = joint_mapping[side][joint]
            center = kMeansCenters[center_idx]
            x_center, y_center = int(center[0]), int(center[1])
            sidejoint=side + joint
            indsidejoint=[i for i, s in enumerate(lowerCenterName) if s == sidejoint]
            Xsidejoint_ij=X[indsidejoint, :]
            modelSrc_ij=np.array(modelSrc)[indsidejoint].tolist()
            print("    ***    ")

            table = PrettyTable()
            table.field_names = ['No.', 'model','Name', 'X', 'Y']
            for j in range(Xsidejoint_ij.shape[0]):
                table.add_row([j, modelSrc_ij[j], sidejoint, Xsidejoint_ij[j, 0], Xsidejoint_ij[j, 1]])

            print(table)
            # Calculate source region around joint center
            src_x1 = x_center - region_width // 2
            src_x2 = src_x1 + region_width
            src_y1 = y_center - region_height // 2
            src_y2 = src_y1 + region_height
            
            # Create padded region with correct aspect ratio
            padded_region = np.zeros((region_height, region_width, 3), dtype=np.uint8)
            
            # Calculate valid source coordinates and padding
            valid_src_x1 = max(0, src_x1)
            valid_src_x2 = min(WIDTH, src_x2)
            valid_src_y1 = max(0, src_y1)
            valid_src_y2 = min(HEIGHT, src_y2)
            
            # Calculate padding amounts
            pad_left = max(0, -src_x1)
            pad_right = max(0, src_x2 - WIDTH)
            pad_top = max(0, -src_y1)
            pad_bottom = max(0, src_y2 - HEIGHT)
            
            # Copy valid region to padded array
            valid_region = pred_viz_joints[valid_src_y1:valid_src_y2, valid_src_x1:valid_src_x2]
            padded_region[pad_top:region_height-pad_bottom, pad_left:region_width-pad_right] = valid_region
            
            # Calculate destination coordinates for this box
            dst_x1 = side_idx * box_width
            dst_y1 = joint_idx * box_height
            
            # Resize padded region maintaining aspect ratio
            zoomed_width = int(region_width * zoom_factor)
            zoomed_height = int(region_height * zoom_factor)
            resized_region = cv2.resize(padded_region, (zoomed_width, zoomed_height))
            
            # Center the zoomed region in the box
            x_offset = (box_width - zoomed_width) // 2
            y_offset = (box_height - zoomed_height) // 2
            
            # Place the resized region
            dst_x1_centered = dst_x1 + x_offset
            dst_y1_centered = dst_y1 + y_offset
            zoom_img[dst_y1_centered:dst_y1_centered+zoomed_height, 
                    dst_x1_centered:dst_x1_centered+zoomed_width] = resized_region
            
            # Draw border around the zoomed region
            cv2.rectangle(zoom_img,
                         (dst_x1_centered, dst_y1_centered),
                         (dst_x1_centered+zoomed_width-1, dst_y1_centered+zoomed_height-1),
                         color=border_color,
                         thickness=border_thickness,
                         lineType=cv2.LINE_AA)
            for i in range(Xsidejoint_ij.shape[0]):
                point=Xsidejoint_ij[i].tolist()
                model=modelSrc_ij[i]
                zoom_x = dst_x1_centered + ((point[0] - src_x1) * zoom_factor)
                zoom_y = dst_y1_centered + ((point[1] - src_y1) * zoom_factor)
                if (dst_x1_centered <= zoom_x < dst_x1_centered + zoomed_width) and \
                        (dst_y1_centered <= zoom_y < dst_y1_centered + zoomed_height):
                    jitterX = np.random.randint(low=-HEIGHT/500, high=HEIGHT/500, size=10)[0]
                    jitterY = np.random.randint(low=-HEIGHT/500, high=HEIGHT / 500, size=10)[0]
                    cv2.circle(zoom_img,
                               (int(zoom_x+jitterX), int(zoom_y+jitterY)),
                               radius=int(RADIUS * 2* zoom_factor / 4),  # Same size as mmposeDraw
                               color=KOLORS[model],
                               thickness=-1)
                    # cv2.putText(zoom_img, model[0], (int(zoom_x), int(zoom_y)), **text_params)

                # if (dst_x1_centered <= zoom_x < dst_x1_centered + zoomed_width) and \
                #         (dst_y1_centered <= zoom_y < dst_y1_centered + zoomed_height):
                #     cv2.circle(zoom_img,
                #                (int(zoom_x), int(zoom_y)),
                #                radius=int(4 * zoom_factor / 4),  # Same size as mmposeDraw
                #                color=KOLORS[model],
                #                thickness=-1)
                #     cv2.putText(zoom_img, model[0], (int(zoom_x), int(zoom_y)), **text_params)
            # Transform and draw points with same size as mmposeDraw
            # for model, points in listkeypoints6.items():
            #     if points is None:
            #         continue
            #
            #     for leg_idx, leg in enumerate(points):
            #         if (leg_idx == 0 and side == 'left') or (leg_idx == 1 and side == 'right'):
            #             point = leg[joint_idx]
            #             # Transform point coordinates to zoomed space
            #             zoom_x = dst_x1_centered + ((point[0] - src_x1) * zoom_factor)
            #             zoom_y = dst_y1_centered + ((point[1] - src_y1) * zoom_factor)
            #
            #             if (dst_x1_centered <= zoom_x < dst_x1_centered+zoomed_width) and \
            #                (dst_y1_centered <= zoom_y < dst_y1_centered+zoomed_height):
            #                 cv2.circle(zoom_img,
            #                          (int(zoom_x), int(zoom_y)),
            #                          radius=int(4 * zoom_factor/4),  # Same size as mmposeDraw
            #                          color=KOLORS[model],
            #                          thickness=-1)
            #                 cv2.putText(zoom_img, model[0], (int(zoom_x), int(zoom_y)), **text_params)

            
            # Draw kmeans center
            zoom_center_x = dst_x1_centered + zoomed_width//2
            zoom_center_y = dst_y1_centered + zoomed_height//2
            cv2.drawMarker(zoom_img, 
                         (int(zoom_center_x), int(zoom_center_y)), 
                         [255, 0, 0], 
                         markerType=cv2.MARKER_TILTED_CROSS, 
                         markerSize=int(RADIUS * 6 ),  # Same size as mmposeDraw
                         thickness=3)
            
            # Draw landmark if available
            if landmarks and landmarks[side].get(joint):
                lm_x, lm_y = landmarks[side][joint]
                zoom_lm_x = dst_x1_centered + ((lm_x - src_x1) * zoom_factor)
                zoom_lm_y = dst_y1_centered + ((lm_y - src_y1) * zoom_factor)
                if (dst_x1_centered <= zoom_lm_x < dst_x1_centered+zoomed_width) and \
                   (dst_y1_centered <= zoom_lm_y < dst_y1_centered+zoomed_height):
                    cv2.drawMarker(zoom_img, 
                                 (int(zoom_lm_x), int(zoom_lm_y)), 
                                 KOLORjson[side],
                                 markerType=cv2.MARKER_CROSS, 
                                 markerSize=int(RADIUS * 6),  # Same size as mmposeDraw
                                 thickness=3)
    
    return zoom_img

def getDictAvrgCenter(listkeypoints, yolo_keypoints, WIDTH):
    listkeypoints6 = listkeypoints.copy()
    listkeypoints6["yolo"] = yolo_keypoints[:, :, 0:2]
    # dictAvrgKpt = calculate_average_joints(listkeypoints)
    X, modelSrc, centerNames, clusterLabels, kMeansCenters, joint_mapping, centerName = calculate_average_joints_by_Kmeans(
        listkeypoints6, WIDTH)
    dictAvrgKpt = dict(left=dict(hip=kMeansCenters[joint_mapping['left']['hip']].tolist(),
                                 knee=kMeansCenters[joint_mapping['left']['knee']].tolist(),
                                 ankle=kMeansCenters[joint_mapping['left']['ankle']].tolist()),
                       right=dict(hip=kMeansCenters[joint_mapping['right']['hip']].tolist(),
                                  knee=kMeansCenters[joint_mapping['right']['knee']].tolist(),
                                  ankle=kMeansCenters[joint_mapping['right']['ankle']].tolist()))
    return dictAvrgKpt
def mmposeDraw(img_bgr, yolo_keypoints, listkeypoints:dict, landmarks:dict):
    listkeypoints6 = listkeypoints.copy()
    listkeypoints6["yolo"] = yolo_keypoints[:, :, 0:2]
    # dictAvrgKpt = calculate_average_joints(listkeypoints)
    X, modelSrc, centerNames, clusterLabels, kMeansCenters, joint_mapping, centerName=calculate_average_joints_by_Kmeans(listkeypoints6, img_bgr.shape[1])
    dictAvrgKpt=dict(left=dict(hip=kMeansCenters[joint_mapping['left']['hip']].tolist(),
                               knee=kMeansCenters[joint_mapping['left']['knee']].tolist(),
                               ankle=kMeansCenters[joint_mapping['left']['ankle']].tolist()),
                     right=dict(hip=kMeansCenters[joint_mapping['right']['hip']].tolist(),
                               knee=kMeansCenters[joint_mapping['right']['knee']].tolist(),
                               ankle=kMeansCenters[joint_mapping['right']['ankle']].tolist()))
    print(dictAvrgKpt)
    MARKERS=dict(hourglass=cv2.MARKER_DIAMOND,
                 Hrnet=cv2.MARKER_STAR,
                 mobilenet=cv2.MARKER_CROSS,
                 rtmpose=cv2.MARKER_SQUARE,
                 shufflenet= cv2.MARKER_TRIANGLE_UP)
    # KOLORS=dict(hourglass=(200, 0, 200),
    #              Hrnet=(0, 0, 0),
    #              mobilenet=(0, 200, 200),
    #              rtmpose=(42, 42, 165),
    #              yolo=(255, 42, 165),
    #              shufflenet= (0, 0, 200))
    lineHeight=img_bgr.shape[0]/60
    RADIUS = int(np.ceil(img_bgr.shape[0] / 500))
    fontSize = img_bgr.shape[0] / 1500

    YPOS = int(img_bgr.shape[0] - (-1) * lineHeight - 20 * lineHeight)
    YPOS2 = int(YPOS + 0.25 * lineHeight)
    XPOS = int(img_bgr.shape[1] / 80) * 2
    # cv2.drawMarker(img_bgr, (XPOS, YPOS), [255, 0, 0], markerType=cv2.MARKER_TRIANGLE_UP, markerSize=int(RADIUS*5),
    #                thickness=2)
    cv2.circle(img_bgr, (XPOS, YPOS), radius=int(RADIUS*3.5), color=KOLORS['yolo'], thickness=-1)
    img_bgr = cv2.putText(img_bgr, "YOLO", (XPOS * 2, YPOS2),
                          cv2.FONT_HERSHEY_DUPLEX, int(fontSize), [255, 255, 255], thickness=2)

    for i in range(kMeansCenters.shape[0]):
        img_bgr = cv2.drawMarker(img_bgr, [int(kMeansCenters[i, 0]), int(kMeansCenters[i, 1])], color=[255, 0, 0],
                             markerType=cv2.MARKER_TILTED_CROSS, markerSize=int(RADIUS * 4),
                             thickness=2)
    for i, modelPt in enumerate(listkeypoints.keys()):
        numHip = 0
        numknee = 0
        numAnkle = 0
        legs = listkeypoints[modelPt]
        for j in range(legs.shape[0]):
            leg = legs[j, :, :]
            hip = leg[0, :]
            numHip+=1
            knee = leg[1, :]
            numknee += 1
            ankle = leg[2, :]
            numAnkle += 1
            # img_bgr = cv2.drawMarker(img_bgr, hip, [0, 0, 255], markerType=MARKERS[modelPt], markerSize=20, thickness=2)
            # img_bgr = cv2.drawMarker(img_bgr, knee, [0, 255, 255], markerType=MARKERS[modelPt], markerSize=20, thickness=2)
            # img_bgr = cv2.drawMarker(img_bgr, ankle, [255, 0, 255], markerType=MARKERS[modelPt], markerSize=20, thickness=2)
            img_bgr = cv2.circle(img_bgr, hip,  radius=RADIUS, color=KOLORS[modelPt], thickness= -1)
            img_bgr = cv2.circle(img_bgr, knee,  radius=RADIUS, color=KOLORS[modelPt], thickness=-1)
            img_bgr = cv2.circle(img_bgr, ankle, radius=RADIUS, color=KOLORS[modelPt], thickness=-1)

        YPOS=int(img_bgr.shape[0]-i*lineHeight - 20*lineHeight)
        YPOS2 = int(YPOS+0.25*lineHeight)
        cv2.circle(img_bgr, [XPOS, YPOS], radius=int(RADIUS*3.5), color=KOLORS[modelPt], thickness=-1)
        img_bgr = cv2.putText(img_bgr, modelPt+": "+",".join(str(x) for x in [numHip, numknee, numAnkle]),(XPOS*2, YPOS2),
                              cv2.FONT_HERSHEY_DUPLEX, int(fontSize), [255, 255, 255], thickness=2)

    if not landmarks is None:
        for side in landmarks.keys():
            legTruth = landmarks[side]
            legAvrg = dictAvrgKpt[side]
            if(len(legTruth)==0):
                continue
            for LOC in legTruth.keys():
                print(LOC)
                print(legTruth[LOC])


                if side in KOLORjson.keys():
                    kol = KOLORjson[side]
                else:
                    kol = [255, 0, 255]
                if not legTruth[LOC] is None:
                    img_bgr = cv2.drawMarker(img_bgr, [int(legTruth[LOC][0]), int(legTruth[LOC][1])], color=kol, markerType=cv2.MARKER_CROSS, markerSize=int(RADIUS*4),
                                         thickness=2)
                # if not legAvrg[LOC] is None:
                #     img_bgr = cv2.drawMarker(img_bgr, [int(legAvrg[LOC][0]), int(legTruth[LOC][1])], color=[0, 0, 255],
                #                          markerType=cv2.MARKER_TILTED_CROSS, markerSize=int(RADIUS * 4),
                #                          thickness=2)
    else:
        YPOS = int(img_bgr.shape[0] - (-2) * lineHeight - 20 * lineHeight)
        img_bgr = cv2.putText(img_bgr, "no kpt json",
                              (XPOS * 2, YPOS),
                              cv2.FONT_HERSHEY_DUPLEX, int(fontSize), [255, 255, 255], thickness=2)
    return dictAvrgKpt
def yoloDraw(num_bbox, bboxes_xyxy, bboxes_keypoints, bbox_label, img_bgr):

    RADIUS = int(np.ceil(img_bgr.shape[0] / 500))
    # 关键点 BGR 配色
    kpt_color_map = {
        0: {'name': '0', 'color': [0, 255, 0], 'radius': RADIUS}, #hip
        1: {'name': '1', 'color': [0,  0, 255], 'radius': RADIUS}, #ankle
        2: {'name': '2', 'color': [255, 0, 0], 'radius': RADIUS}, #knee
        3: {'name': '3', 'color': [255, 250, 0], 'radius': RADIUS},
        4: {'name': '4', 'color': [255, 0, 255], 'radius': RADIUS},
        5: {'name': '5', 'color': [0, 255, 255], 'radius': RADIUS},
        6: {'name': '6', 'color': [255, 165, 0], 'radius': RADIUS},
        7: {'name': '7', 'color': [128, 0, 128], 'radius': RADIUS},
        8: {'name': '8', 'color': [164, 42, 42], 'radius': RADIUS},
        9: {'name': '9', 'color': [128, 128, 0], 'radius': RADIUS},
    }

    ###
    for idx in range(num_bbox):  # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度

        # for kpt_id in kpt_color_map:
        for kpt_id in range(bbox_keypoints.shape[0]):
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]

            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为充）
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y),  radius=RADIUS, color=KOLORS["yolo"], thickness=-1)
            # img_bgr = cv2.drawMarker(img_bgr, (kpt_x, kpt_y), [255, 0, 0], markerType=cv2.MARKER_TRIANGLE_UP, markerSize=kpt_radius,
            #                          thickness=2)
            # img_bgr = cv2.drawMarker(img_bgr, hip, [0, 0, 255], markerType=MARKERS[modelPt], markerSize=20, thickness=2)
            # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
#             kpt_label = str(kpt_id) # 写关键点类别 ID
#             kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称
#             img_bgr = cv2.putText(img_bgr, kpt_label, (kpt_x+kpt_labelstr['offset_x'], kpt_y+kpt_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color, kpt_labelstr['font_thickness'])
def pruningSeg(pred_mask_Stacker, dictAverageKpts, img_bgr, img_truth, drawBox):
    SHAPE = pred_mask_Stacker.shape
    pred_mask_Stacker_pruned = np.zeros(SHAPE)
    pred_mask_Stacker_pruned_255 = np.zeros((pred_mask_Stacker_pruned.shape[0], pred_mask_Stacker_pruned.shape[1], 3),
                                            dtype='uint8')

    FONTSIZE = img_bgr.shape[0] * 6 / 4000
    YPOS = int(round(img_bgr.shape[0] * 15 / 400))
    YPOS1 = int(YPOS * 1)
    YPOS2 = int(round(YPOS * 2))
    YPOS3 = int(round(YPOS * 3))
    YPOS4 = int(round(YPOS * 4))
    text_params = {
        'fontFace': cv2.FONT_HERSHEY_TRIPLEX,
        'fontScale': FONTSIZE,
        'color': (255, 255, 255),
        'thickness': 3,
        'lineType': cv2.LINE_AA
    }
    WIDTH=SHAPE[1]
    halfWIDTH=int(WIDTH/2)
    # pred_mask_Stacker_pruned = np.zeros(SHAPE)
    femur_stacker = np.zeros(SHAPE)
    tibia_stacker = np.zeros(SHAPE)

    femur_stacker[np.where(pred_mask_Stacker == 1)] = 1
    tibia_stacker[np.where(pred_mask_Stacker == 2)] = 1
    posFemur_stacker = np.where(femur_stacker > 0)
    posTibia_stacker = np.where(tibia_stacker > 0)
    posTibia_stackerL = np.where((tibia_stacker == 1) & (np.arange(WIDTH) < halfWIDTH)[None, :])
    posTibia_stackerR = np.where((tibia_stacker == 1) & (np.arange(WIDTH) >= halfWIDTH)[None, :])
    posFemur_stackerL = np.where((femur_stacker == 1) & (np.arange(WIDTH) < halfWIDTH)[None, :])
    posFemur_stackerR = np.where((femur_stacker == 1) & (np.arange(WIDTH) >= halfWIDTH)[None, :])   

    if not check_dict_keys(dictAverageKpts):
        return pred_mask_Stacker_pruned_255

    tibiaLenL=np.sqrt(sum((np.array(dictAverageKpts['left']['knee']) - np.array(dictAverageKpts['left']['ankle']))**2))
    tibiaLenR=np.sqrt(sum((np.array(dictAverageKpts['right']['knee']) - np.array(dictAverageKpts['right']['ankle']))**2))
    femurLenL=np.sqrt(sum((np.array(dictAverageKpts['left']['hip']) - np.array(dictAverageKpts['left']['knee']))**2))
    femurLenR=np.sqrt(sum((np.array(dictAverageKpts['right']['hip']) - np.array(dictAverageKpts['right']['knee']))**2))
    print(f"tibiaLenL={tibiaLenL}")
    print(f"tibiaLenR={tibiaLenR}")
    print(f"femurLenL={femurLenL}")
    print(f"femurLenR={femurLenR}")
    lowerBoundTibiaL = int(dictAverageKpts['left']['ankle'][1] + tibiaLenL*0.08)
    upperBoundTibiaL = int(dictAverageKpts['left']['knee'][1] - tibiaLenL * 0.08)
    lowerBoundTibiaR = int(dictAverageKpts['right']['ankle'][1] + tibiaLenR * 0.08)
    upperBoundTibiaR = int(dictAverageKpts['right']['knee'][1] - tibiaLenR * 0.08)
    print(f"lowerBoundTibiaL={lowerBoundTibiaL}")
    print(f"upperBoundTibiaL={upperBoundTibiaL}")
    print(f"lowerBoundTibiaR={lowerBoundTibiaR}")
    print(f"upperBoundTibiaR={upperBoundTibiaR}")

    lowerBoundFemurL = int(dictAverageKpts['left']['knee'][1] + femurLenL * 0.08)
    upperBoundFemurL = int(dictAverageKpts['left']['hip'][1] - femurLenL * 0.10)
    lowerBoundFemurR = int(dictAverageKpts['right']['knee'][1] + femurLenR * 0.08)
    upperBoundFemurR = int(dictAverageKpts['right']['hip'][1] - femurLenR * 0.10)
    print(f"lowerBoundFemurL={lowerBoundFemurL}")
    print(f"upperBoundFemurL={upperBoundFemurL}")
    print(f"lowerBoundFemurR={lowerBoundFemurR}")
    print(f"upperBoundFemurR={upperBoundFemurR}")

    posTibia_stackerL_pruned = (
        posTibia_stackerL[0][(posTibia_stackerL[0] > upperBoundTibiaL) & (posTibia_stackerL[0] < lowerBoundTibiaL)],
        posTibia_stackerL[1][(posTibia_stackerL[0] > upperBoundTibiaL) & (posTibia_stackerL[0] < lowerBoundTibiaL)])
    posTibia_stackerR_pruned = (
        posTibia_stackerR[0][(posTibia_stackerR[0] > upperBoundTibiaR) & (posTibia_stackerR[0] < lowerBoundTibiaR)],
        posTibia_stackerR[1][(posTibia_stackerR[0] > upperBoundTibiaR) & (posTibia_stackerR[0] < lowerBoundTibiaR)])
    posFemur_stackerL_pruned = (
        posFemur_stackerL[0][(posFemur_stackerL[0] > upperBoundFemurL) & (posFemur_stackerL[0] < lowerBoundFemurL)],
        posFemur_stackerL[1][(posFemur_stackerL[0] > upperBoundFemurL) & (posFemur_stackerL[0] < lowerBoundFemurL)])
    posFemur_stackerR_pruned = (
        posFemur_stackerR[0][(posFemur_stackerR[0] > upperBoundFemurR) & (posFemur_stackerR[0] < lowerBoundFemurR)],
        posFemur_stackerR[1][(posFemur_stackerR[0] > upperBoundFemurR) & (posFemur_stackerR[0] < lowerBoundFemurR)])
   
    pred_mask_Stacker_pruned[posTibia_stackerL_pruned] = 2
    pred_mask_Stacker_pruned[posTibia_stackerR_pruned] = 2
    pred_mask_Stacker_pruned[posFemur_stackerL_pruned] = 1
    pred_mask_Stacker_pruned[posFemur_stackerR_pruned] = 1

    tab, iou_stackerPruned_FT = compute_ious(pred_mask_Stacker_pruned, img_truth[:, :, 0], [1,2], fileNum='prune', modelName="stacker+pruning",
                                     fold='prune', metric="IOU", isChosen=1)
    pred_mask_Stacker_pruned_255[pred_mask_Stacker_pruned == 1] = palette_dict[1]  # Femur color
    pred_mask_Stacker_pruned_255[pred_mask_Stacker_pruned == 2] = palette_dict[2]  # Tibia color
    pred_viz_Stacker_pruned = cv2.addWeighted(img_bgr, opacity, pred_mask_Stacker_pruned_255, 1 - opacity, 0.0).astype('uint8')
    # Add text annotations
    cv2.putText(pred_viz_Stacker_pruned, "stacking+pruning", (20, YPOS1), **text_params)
    cv2.putText(pred_viz_Stacker_pruned, f"fIOU:{iou_stackerPruned_FT[1]}", (20, YPOS2), **text_params)
    cv2.putText(pred_viz_Stacker_pruned, f"tIOU:{iou_stackerPruned_FT[2]}", (20, YPOS3), **text_params)

    if drawBox:
        cv2.rectangle(pred_viz_Stacker_pruned, pt1=(20, upperBoundFemurL),  pt2=(halfWIDTH - 20, lowerBoundFemurL),  # bottom-right corner
                      color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)  # anti-aliased line
        cv2.rectangle(pred_viz_Stacker_pruned, pt1=(100, upperBoundTibiaL), pt2=(halfWIDTH - 10, lowerBoundTibiaL),
                      color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)  # anti-aliased line
        cv2.rectangle(pred_viz_Stacker_pruned, pt1=(halfWIDTH + 20, upperBoundFemurR), pt2=(WIDTH - 20, lowerBoundFemurR),
                      color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)  # anti-aliased line
        cv2.rectangle(pred_viz_Stacker_pruned, pt1=(halfWIDTH + 10, upperBoundTibiaR), pt2=(WIDTH - 100, lowerBoundTibiaR),
                      color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)  # anti-aliased line

    return pred_viz_Stacker_pruned


def draw_intersection_point(img, slope1, intercept1, slope2, intercept2, width, height, pos="left", AngleName="alpha"):
    """
    Calculate and draw the intersection point of two lines in x = ky + c form
    """
    FONTSIZE = height * 3 / 4000
    # Calculate intersection point
    angle_deg = 180
    if abs(slope1 - slope2) > 1e-6:  # Lines are not parallel
        # For lines in form x = ky + c:
        # k1y + c1 = k2y + c2
        # y(k1 - k2) = c2 - c1
        intersect_y = (intercept2 - intercept1) / (slope1 - slope2)
        intersect_x = slope1 * intersect_y + intercept1

        # Convert to screen coordinates and check bounds
        screen_x = int(intersect_x)
        screen_y = int(intersect_y)
        
        if (0 <= screen_x <= width) and (0 <= screen_y <= height):
            # Calculate angle between lines
            # For x = ky + c form, use arctan(k) instead of arctan2(1, k)
            angle_rad = abs(np.arctan(slope1) - np.arctan(slope2))
            angle_deg = np.degrees(angle_rad)
            
            # Normalize angle to be between 0 and 180

            if AngleName=="NSA":
                angle_deg = max(angle_deg, 180 - angle_deg)
            else:
                angle_deg = min(angle_deg, 180 - angle_deg)
            # Draw intersection point
            cv2.circle(img,
                      (screen_x, screen_y),
                      radius=15,
                      color=(0, 0, 255),  # Red
                      thickness=-1)

            # Position text
            offset_y = 30
            offset_x = -100 if pos == "right" else -140
            text_x = int(screen_x + offset_x)
            text_y = int(screen_y - offset_y)

            # Draw angle text
            text = fr"{AngleName}: {angle_deg:.1f} deg"
            font_scale = FONTSIZE
            font = cv2.FONT_HERSHEY_DUPLEX
            
            # Get text size for better positioning
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 2)
            
            # Adjust text position to stay within bounds
            text_x = max(5, min(width - text_width - 5, text_x))
            text_y = max(text_height + 5, min(height - 5, text_y))
            
            # Draw text with white background for better visibility
            cv2.putText(img, text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), 2)

            return [(screen_x, screen_y), angle_deg]
    
    return [(-1, -1), angle_deg]

def fit_line_pca(x_points, y_points, img_pca):
    """
    Fits a line to points using PCA
    Returns slope and intercept of the line
    """
    points = np.vstack((x_points, y_points)).T
    # Center the data
    points_mean = points.mean(axis=0)
    points_centered = points - points_mean

    # Compute PCA
    U, S, Vt = np.linalg.svd(points_centered)

    # Direction vector of the line is the first principal component
    direction = Vt[0]

    # Calculate slope and intercept
    slope_pca = direction[1] / direction[0]
    intercept_pca = points_mean[1] - slope_pca * points_mean[0]

    height, width = img_pca.shape[:2]
    x1, x2 = 0, width
    y1 = int(slope_pca * x1 + intercept_pca)
    y2 = int(slope_pca * x2 + intercept_pca)
    cv2.line(img_pca, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green line for PCA

    return slope_pca, intercept_pca


def fit_line_ransac(x_points, y_points, img=None, max_trials=100, residual_threshold=5.0,
                    visualize=True, line_color=(0, 0, 0), inlier_color=(0, 255, 0),
                    outlier_color=(0, 0, 255)):
    """
    Fit a line using RANSAC and optionally visualize the results

    Args:
        x_points (array-like): X coordinates of points
        y_points (array-like): Y coordinates of points
        img (numpy.ndarray, optional): Image to draw on. If None, no visualization
        max_trials (int): Maximum number of iterations for RANSAC
        residual_threshold (float): Maximum deviation for inliers (in pixels)
        visualize (bool): Whether to visualize results on img
        line_color (tuple): BGR color for the fitted line
        inlier_color (tuple): BGR color for inlier points
        outlier_color (tuple): BGR color for outlier points

    Returns:
        tuple: (slope, intercept, inlier_mask)
    """
    from sklearn.linear_model import RANSACRegressor

    # Reshape data for sklearn
    X = np.array(x_points).reshape(-1, 1)
    y = np.array(y_points)

    # Fit RANSAC
    ransac = RANSACRegressor(
        max_trials=max_trials,
        min_samples=2,
        residual_threshold=residual_threshold,
        random_state=42
    )
    ransac.fit(X, y)

    # Get the line parameters
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    # Visualize if requested and img is provided
    if visualize and img is not None:
        height, width = img.shape[:2]

        # Draw the RANSAC line
        x1, x2 = 0, width
        y1 = int(slope * x1 + intercept)
        y2 = int(slope * x2 + intercept)
        cv2.line(img, (x1, y1), (x2, y2), line_color, thickness=1)

        # Visualize inlier and outlier points
        inlier_mask = ransac.inlier_mask_

        # Draw points
        for idx in range(len(x_points)):
            color = inlier_color if inlier_mask[idx] else outlier_color
            cv2.circle(img,
                       (int(x_points[idx]), int(y_points[idx])),
                       radius=3,
                       color=color,
                       thickness=-1)

    return slope, intercept, ransac.inlier_mask_


import json


def read_landmarks_from_json(json_path):
    if json_path is None:
        return None
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Dictionary to store coordinates for both legs
    landmarks = {
        'left': {},
        'right': {}
    }

    # Process each shape in the JSON
    for shape in data['shapes']:
        # Get coordinates (first point for landmarks)
        points = shape['points'][0]  # Each landmark has only one point [x, y]
        x, y = points

        # Determine left/right based on x coordinate
        # Assuming left leg has smaller x coordinate
        side = 'left' if x < data['imageWidth'] / 2 else 'right'

        # Store coordinates based on label
        if shape['label'] == 'Femoral_head_center':
            landmarks[side]['hip'] = (x, y)
        elif shape['label'] == 'Tibial_plafond_center':
            landmarks[side]['knee'] = (x, y)
        elif shape['label'] == 'Tibial_spine_center':
            landmarks[side]['ankle'] = (x, y)

    print(landmarks)
    for side in ['left', 'right']:
        print(f"\n{side.upper()} LEG:")
        for joint, coords in landmarks[side].items():
            print(f"{joint}: (x={coords[0]:.1f}, y={coords[1]:.1f})")

    return landmarks
def calculate_average_joints_by_Kmeans(listkeypoints6, WIDTH):
    from sklearn.cluster import KMeans

    points = []
    pointSrc = []
    for modi, vali in listkeypoints6.items():
        dati = vali.reshape((vali.shape[0]*vali.shape[1], vali.shape[2]))
        points.append(dati)
        pointSrc.append([modi]*dati.shape[0])

    # Convert to numpy array and perform k-means
    X = np.vstack(points)
    modelSrc = sum(pointSrc, []) #np.array(pointSrc).flatten()

    kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
    kmeans.fit(X)
    kmeans6 = KMeans(n_clusters=6, random_state=42, n_init='auto').fit(X)
    kmeans5 = KMeans(n_clusters=5, random_state=42, n_init='auto').fit(X)
    kmeans4 = KMeans(n_clusters=4, random_state=42, n_init='auto').fit(X)
    kmeans3 = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(X)
    kmeans2 = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X)
    kmeans1 = KMeans(n_clusters=1, random_state=42, n_init='auto').fit(X)

    # Compare inertia (sum of squared distances to nearest centroid)
    scores= np.array([kmeans1.inertia_, kmeans2.inertia_, kmeans3.inertia_, kmeans4.inertia_, kmeans5.inertia_, kmeans6.inertia_])
    for i, score in enumerate(scores):
        print(f"Scores of KMeans model {i} (k={i+1}) : {round(score)}")
    KMEANSMODELS = [kmeans1, kmeans2, kmeans3, kmeans4, kmeans5, kmeans6]

    # Select better model
    best_model = KMEANSMODELS[np.argmin(scores)]
    clusterLabels = best_model.labels_  #
    import pandas as pd

    crosstab = pd.crosstab(clusterLabels, modelSrc, margins=True)
    # crosstab_pct = pd.crosstab(clusterLabels, modelSrc, margins=True, normalize='index') * 100

    # Print both tables
    print("Counts:")
    print(crosstab)

    # Get cluster centers
    kMeansCenters = best_model.cluster_centers_  # Shape: (6, 2)
    ##############
    # Identify joint indices (same as before)
    leftHip = np.argmin(np.prod(kMeansCenters, axis=1))
    rightAnkle = np.argmax(np.prod(kMeansCenters, axis=1))
    rightHip = np.argmin(np.prod(np.array([WIDTH - kMeansCenters[:, 0], kMeansCenters[:, 1]]).T, axis=1))
    leftAnkle = np.argmax(np.prod(np.array([WIDTH - kMeansCenters[:, 0], kMeansCenters[:, 1]]).T, axis=1))
    remainingPoints = list(set(range(len(kMeansCenters))) - set([leftHip, rightAnkle, rightHip, leftAnkle]))

    if kMeansCenters[remainingPoints[0], 0] < kMeansCenters[remainingPoints[1], 0]:
        leftKnee, rightKnee = remainingPoints
    else:
        rightKnee, leftKnee = remainingPoints

    joint_mapping = {
        'left': {'hip': leftHip, 'knee': leftKnee, 'ankle': leftAnkle},
        'right': {'hip': rightHip, 'knee': rightKnee, 'ankle': rightAnkle}
    }
    centerName=[""]*6
    DICT=dict(leftHip=leftHip, leftKnee=leftKnee, leftAnkle=leftAnkle, rightHip=rightHip, rightKnee=rightKnee, rightAnkle=rightAnkle)
    for k, v in DICT.items():
        centerName[v]=k

    centerNames=np.array(centerName)[clusterLabels]
    crosstab2 = pd.crosstab(modelSrc, centerNames, margins=True)
    print(crosstab2)
    ##############
    # Print centers
    for i, (center_x, center_y) in enumerate(kMeansCenters):
        print(f"Cluster {i} center: ({center_x:.1f}, {center_y:.1f})")

    return [X, modelSrc, centerNames, clusterLabels, kMeansCenters, joint_mapping, centerName]
def calculate_average_joints(listkeypoints):
    """
    Calculate average positions and maintain calculation strings for joints
    """
    # Initialize storage for valid points and calculation strings
    joints = {
        'left': {'hip': [], 'knee': [], 'ankle': [],
                 'hip_xstr': [], 'hip_ystr': [],
                 'knee_xstr': [], 'knee_ystr': [],
                 'ankle_xstr': [], 'ankle_ystr': []},
        'right': {'hip': [], 'knee': [], 'ankle': [],
                  'hip_xstr': [], 'hip_ystr': [],
                  'knee_xstr': [], 'knee_ystr': [],
                  'ankle_xstr': [], 'ankle_ystr': []}
    }

    # Collect all valid points
    for model_name, predictions in listkeypoints.items():
        if predictions is None or len(predictions) == 0:
            continue

        if len(predictions) == 2:
            # Process left leg
            left_leg = predictions[0]
            for i, joint in enumerate(['hip', 'knee', 'ankle']):
                x, y = left_leg[i]
                if x > 0 and y > 0:
                    joints['left'][joint].append((x, y))
                    joints['left'][f'{joint}_xstr'].append(f"{x:.1f}")
                    joints['left'][f'{joint}_ystr'].append(f"{y:.1f}")

            # Process right leg
            right_leg = predictions[1]
            for i, joint in enumerate(['hip', 'knee', 'ankle']):
                x, y = right_leg[i]
                if x > 0 and y > 0:
                    joints['right'][joint].append((x, y))
                    joints['right'][f'{joint}_xstr'].append(f"{x:.1f}")
                    joints['right'][f'{joint}_ystr'].append(f"{y:.1f}")

    # Calculate averages and format strings
    averages = {
        'left': {},
        'right': {}
    }

    for side in ['left', 'right']:
        for joint in ['hip', 'knee', 'ankle']:
            points = joints[side][joint]
            if points:
                # Calculate averages
                avg_x = sum(p[0] for p in points) / len(points)
                avg_y = sum(p[1] for p in points) / len(points)
                averages[side][joint] = (avg_x, avg_y)

                # Add calculation strings
                x_strs = joints[side][f'{joint}_xstr']
                y_strs = joints[side][f'{joint}_ystr']
                averages[side][f'{joint}_xstr'] = f"({'+'.join(x_strs)})/{len(x_strs)}"
                averages[side][f'{joint}_ystr'] = f"({'+'.join(y_strs)})/{len(y_strs)}"
            else:
                averages[side][joint] = None
                averages[side][f'{joint}_xstr'] = "no valid points"
                averages[side][f'{joint}_ystr'] = "no valid points"
                print(f"Warning: No valid points for {side} {joint}")

    return averages


# Example usage:
"""
listkeypoints = {
    'hourglass': np.array([[[100, 200], [150, 400], [200, 600]], 
                          [[300, 200], [350, 400], [400, 600]]]),
    'Hrnet': np.array([[[110, 210], [160, 410], [210, 610]], 
                       [[310, 210], [360, 410], [410, 610]]]),
}

averages = calculate_average_joints(listkeypoints)
for side in ['left', 'right']:
    for joint in ['hip', 'knee', 'ankle']:
        if averages[side][joint]:
            x, y = averages[side][joint]
            print(f"{side} {joint}:")
            print(f"  Position: ({x:.1f}, {y:.1f})")
            print(f"  X calculation: {averages[side][f'{joint}_xstr']}")
            print(f"  Y calculation: {averages[side][f'{joint}_ystr']}")
"""
def check_dict_keys(dict_to_check):
    """
    required_keys format: [('level1', 'level2'), ('left', 'hip'), ...]
    """
    # Example usage:
    required_keys = [
        ('left', 'hip'), ('left', 'knee'), ('left', 'ankle'),
        ('right', 'hip'), ('right', 'knee'), ('right', 'ankle')
    ]
    for level1, level2 in required_keys:
        if (dict_to_check[level1] is None) or (dict_to_check[level1][level2] is None):
            return False
    return True


