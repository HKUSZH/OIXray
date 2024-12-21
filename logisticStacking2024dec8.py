import logging
import timeit

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
from prettytable import PrettyTable

def logisticsStackingForOne(predStacks, gtruth, class_ids, fileNum="", fold=""):
    table = PrettyTable()
    table.field_names = ["Category", "fold", "fileNum", "classID", "intercept", "coef1", "coef2", "coef3",
                         "coef4", "coef5", "coef6"]
    predStacks2 = np.stack(predStacks, axis=-1)
    modelMatrix = predStacks2.reshape(-1, 6)
    y_meta = gtruth.flatten()

    for class_id in class_ids:
        X_meta = np.zeros(modelMatrix.shape)
        X_meta[np.where(modelMatrix == class_id)]=1

        y_meta1 = np.zeros(y_meta.shape)
        y_meta1[np.where(y_meta == class_id)]=1


        stacker = LogisticRegression(max_iter=10000)
        stacker.fit(X_meta, y_meta1)
        print(stacker.coef_)
        COEFF = np.round(stacker.coef_[0], 4)
        INTERCEPT = np.round(stacker.intercept_[0], 4)
        table.add_row(
            ["Stacking", fold, fileNum, class_id, INTERCEPT, COEFF[0], COEFF[1], COEFF[2], COEFF[3], COEFF[4], COEFF[5]])

    print(table)
    # return(stacker)

def logisticsStackingForMany(listPred, listGT_Meta, thisfold):
    np.random.seed(42)
    selected = np.random.randint(0, 4, size=len(listPred[0]))
    print(selected)
    selectedList = selected.tolist()
    print(selectedList)


    BONES = {1: "Femur", 2: "Tibia"}

    for part in range(4):
        print(f"handling part {part}")
        numTr = len(np.where(selected != part)[0])
        numTst = len(np.where(selected == part)[0])

        y_meta_train = [gtruth.flatten() for gtruth, k in zip(listGT_Meta, selectedList) if k != part]  # Flatten ground truth
        print(f"len of y_meta_train: {len(y_meta_train)}")
        y_meta_train = np.concatenate(y_meta_train)

        y_meta_test = [gtruth.flatten() for gtruth, k in zip(listGT_Meta, selectedList) if k == part]  # Flatten ground truth
        print(f"len of y_meta_test: {len(y_meta_test)}")
        y_meta_test = np.concatenate(y_meta_test)
        ####
        X_meta_train = []
        for i in range(6):  # Loop over the 6 models
            predictions = listPred[i]  # Shape (40, X, H)
            xi = [predi.flatten() for predi, k in zip(predictions, selectedList) if k != part]  # Flatten ground truth
            print(f"len of xi train: {len(xi)}")
            xi = np.concatenate(xi)
            X_meta_train.append(xi)

        modelMatrix = np.stack(X_meta_train, axis=1)

        X_meta_test = []
        for i in range(6):  # Loop over the 6 models
            valsamples = listPred[i]  # Shape (40, X, H)
            vali = [val.flatten() for val, k in zip(valsamples, selectedList) if k == part]  # Flatten ground truth
            vali = np.concatenate(vali)
            X_meta_test.append(vali)
        valMatrix = np.stack(X_meta_test, axis=1)
        ####

        for class_id in [1, 2]:
            logging.info(f"handling part {part}, {BONES[class_id]}")

            y_train = np.zeros(y_meta_train.shape)
            y_train[np.where(y_meta_train == class_id)] = 1

            y_test = np.zeros(y_meta_test.shape)
            y_test[np.where(y_meta_test == class_id)] = 1

            modelMatrix_train =np.zeros(modelMatrix.shape)
            modelMatrix_train[np.where(modelMatrix == class_id)] = 1

            stacker = LogisticRegression(max_iter=1000, tol=1e-6, verbose=1)
            stacker.fit(modelMatrix_train, y_train)

            valMatrix_test = np.zeros(valMatrix.shape)
            valMatrix_test[np.where(valMatrix == class_id)] = 1

            predicted = stacker.predict(valMatrix_test)

            modelOut = 'logistic_regression_model.'+BONES[class_id]+'.'+thisfold+'part.'+str(part)+'.pkl'
            joblib.dump(stacker, modelOut)

            row_sumsTr = np.sum(modelMatrix_train, axis=1)
            row_sumsTst = np.sum(valMatrix_test, axis=1)

            def XTab(vec1, vec2, inMain):
                df = pd.DataFrame({"vec1": vec1, "vec2": vec2})
                crosstab_result = pd.crosstab(df['vec1'], df['vec2'])
                print(inMain)
                print(crosstab_result)
                print("***")
                pretty_table = PrettyTable()
                pretty_table.field_names = ['TableName', 'bones', 'fold', 'comp', 'vec1'] + crosstab_result.columns.tolist()
                for index, row in crosstab_result.iterrows():
                    # Add the row index as the first column, followed by the row values
                    pretty_table.add_row(['crossTab', BONES[class_id], thisfold, inMain]+[index] + row.tolist())
                print(pretty_table)
                logging.info("\n%s", pretty_table)
                return(crosstab_result)

            # random_indices = np.random.choice(y_train.shape[0], size=50000, replace=False)
            indtr = np.random.randint(0, y_train.shape[0] - 1, size=50000)
            indtst = np.random.randint(0, y_test.shape[0] - 1, size=5000000)

            xtab1 = XTab(row_sumsTr[indtr], y_train[indtr], "row_sumsTr vs y_train")
            xtab2 = XTab(row_sumsTst, y_test, "row_sumsTst vs y_test")
            xtab3 = XTab(predicted, y_test, "predictedFemur vs y_test")
            xtab4 = XTab(row_sumsTst>3, y_test, "row_sumsTst>3 vs y_test")
            TN3 = xtab3.loc[xtab3.index.tolist()[0], xtab3.columns.tolist()[0]]
            TP3 = xtab3.loc[xtab3.index.tolist()[1], xtab3.columns.tolist()[1]]
            FN3 = xtab3.loc[xtab3.index.tolist()[0], xtab3.columns.tolist()[1]]
            FP3 = xtab3.loc[xtab3.index.tolist()[1], xtab3.columns.tolist()[0]]
            IOU3= TP3/(TP3+FN3+FP3)

            TN4 = xtab4.loc[xtab4.index.tolist()[0], xtab4.columns.tolist()[0]]
            TP4 = xtab4.loc[xtab4.index.tolist()[1], xtab4.columns.tolist()[1]]
            FN4 = xtab4.loc[xtab4.index.tolist()[0], xtab4.columns.tolist()[1]]
            FP4 = xtab4.loc[xtab4.index.tolist()[1], xtab4.columns.tolist()[0]]
            IOU4 = TP4 / (TP4 + FN4 + FP4)

            pretty_table_tst = PrettyTable()
            pretty_table_tst.field_names = ['TableName', 'bones', 'fold', 'part', 'comp', 'TN', 'TP', "FN", "FP", "IOU", "numTr", "NumTst", "NumModel", "NumPixcelsTr", "NumPixcelsTst"]
            pretty_table_tst.add_row(['logisticRes', BONES[class_id], thisfold, part, "stacking vs y_test",TN3, TP3, FN3, FP3, IOU3, numTr, numTst, len(listPred), modelMatrix.shape[0], valMatrix.shape[0]])
            pretty_table_tst.add_row(['logisticRes', BONES[class_id], thisfold, part, "row_sumsTst>3 vs y_test",TN4, TP4, FN4, FP4, IOU4, numTr, numTst, len(listPred), modelMatrix.shape[0], valMatrix.shape[0]])

            pretty_table_model = PrettyTable()
            pretty_table_model.field_names = ['TableName', 'bones', 'fold', 'part', 'comp', "intercept"] +["PSPNet", "UNet", "KNet", "FastSCNN", "DeepLabV3plus", "Segformer"] + ["numTr", "NumTst", "NumModel", "NumPixcelsTr", "NumPixcelsTst"]
            pretty_table_model.add_row(
                ['stackingModel', BONES[class_id], thisfold, part, "predicted vs y_test"] +
                    [stacker.intercept_[0]]+stacker.coef_[0].tolist() +
                    [numTr, numTst, len(listPred), modelMatrix.shape[0], valMatrix.shape[0]])

            MODELS = ["PSPNet", "UNet", "KNet", "FastSCNN", "DeepLabV3plus", "Segformer"]
            for j in range(6):
                xtabj = XTab(valMatrix_test[:, j], y_test, MODELS[j] + " vs y_test (gt)")
                TNj = xtabj.loc[xtabj.index.tolist()[0], xtabj.columns.tolist()[0]]
                TPj = xtabj.loc[xtabj.index.tolist()[1], xtabj.columns.tolist()[1]]
                FNj = xtabj.loc[xtabj.index.tolist()[0], xtabj.columns.tolist()[1]]
                FPj = xtabj.loc[xtabj.index.tolist()[1], xtabj.columns.tolist()[0]]
                IOUj = TPj / (TPj + FNj + FPj)
                pretty_table_tst.add_row(
                    ['individual', BONES[class_id], thisfold, part, MODELS[j] +" vs y_test", TNj, TPj, FNj, FPj, IOUj,
                     numTr, numTst, len(listPred), modelMatrix.shape[0], valMatrix.shape[0]])

            print(pretty_table_tst)
            print(pretty_table_model)
            logging.info("\n%s", pretty_table_model)
            logging.info("\n%s", pretty_table_tst)


def predict_with_stacker(model_path, listPred, class_id=1):
    """
    Load a trained stacking model and make predictions based on ensemble predictions

    Args:
        model_path (str): Path to the saved model pickle file
        listPred (list): List of 6 model predictions [model1_pred, model2_pred, ...]
                        where each prediction contains binary values (0,1)
                        from PSPNet, UNet, KNet, FastSCNN, DeepLabV3plus, Segformer
        class_id (int): Class ID to predict (1 for Femur, 2 for Tibia)

    Returns:
        predicted: Binary prediction mask (0 or 1)
    """
    # Load the trained model
    print(f"loading from {model_path}")
    stacker = joblib.load(model_path)

    # Stack predictions from different models
    X_meta = []
    for i in range(6):  # Loop over the 6 models
        predictions = listPred[i]  # Contains binary values 0,1
        xi = predictions.flatten()
        X_meta.append(xi)

    # Create input matrix
    valMatrix = np.stack(X_meta, axis=1)

    # Make prediction
    predicted = stacker.predict(valMatrix)

    # Reshape back to original dimensions
    predicted = predicted.reshape(listPred[0].shape)

    return predicted
