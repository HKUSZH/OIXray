import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from prettytable import PrettyTable

def compute_ious(pred, gt, class_ids, fileNum="", modelName="", fold="", metric="IOU", isChosen=0):
    bones = {1: "Femur", 2: "Tibia"}
    ious = {}
    table = PrettyTable()
    table.field_names = ["Category", "fold", "modelName", "fileNum", "isChosen", "classID", "boneName", "iou", "recall", "precision", "f1", "fmindex"] +   ["intersection",
        "union", "truth", "predicted", "truth_neg"]
    for class_id in class_ids:
        thisBone = bones[class_id]
        pred_mask = (pred == class_id)
        gt_mask = (gt == class_id)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        truth_neg = np.sum(gt_mask == False)
        truth_pos = np.sum(gt_mask == True)
        predicted = np.sum(pred_mask == True)

        iou = round(intersection / union, 4) if union != 0 else 0.0
        recall = round(intersection / truth_pos, 4) if union != 0 else 0.0
        precision = round(intersection / predicted, 4) if union != 0 else 0.0
        f1 = round(2 * (recall * precision) / (recall + precision), 4) if union != 0 else 0.0
        fmindex = round(np.sqrt(recall * precision), 4) if union != 0 else 0.0

        if np.random.random(1)>1.99:
            pred_mask1 = pred_mask.ravel()
            gt_mask1 = gt_mask.ravel()
            precision_sk = round(precision_score(gt_mask1, pred_mask1), 4)
            recall_sk = round(recall_score(gt_mask1, pred_mask1), 4)
            f1_sk = round(f1_score(gt_mask1, pred_mask1), 4)
        else:
            precision_sk = -1
            recall_sk = -1
            f1_sk = -1

        table.add_row(["FrontSeg", fold, modelName, fileNum, isChosen, class_id, thisBone, iou, recall, precision, f1, fmindex] + #, precision_sk, recall_sk, f1_sk]+
                      [intersection, union, truth_pos, predicted, truth_neg])

        # iou = calculate_iou_per_class(pred_mask, gt_mask, class_id,  fileNum, modelName, fold, metric)
        ious[class_id] = iou

    print(table)
    return table, ious #, [intersection, union, truth_pos, predicted, truth_neg]

def save_iou_results(filename, img_name, ious, palette):
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(f"{img_name} - ")
        iou_str = "-".join([f"{palette[class_id][0]}类 - {iou:.4f}" for class_id, iou in ious.items()])
        f.write(iou_str + "\n")

opacity=0.5
palette = [
        ['background', [0, 0, 0]],
        ['Femur', [0, 0, 255]],
        ['Tibia', [0, 255, 0]],
        ['FemurAxis', [255, 255, 255]],
        ['TibiaAxis', [255, 255, 255]]
    ]

gt_mapping = {
    0: 0,  # background
    38: 1,  # Femur
    75: 2  # Tibia
}

palette_dict = {idx: each[1] for idx, each in enumerate(palette)}
