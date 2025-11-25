import os
import time
import psutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Img getter
def imgGet(imgp):
    arr = np.array(Image.open(imgp).convert("L"))
    return {
        "mean": np.mean(arr), "std": np.std(arr), "max": np.max(arr), "min": np.min(arr), "nonzero": np.count_nonzero(arr), "entropy": -np.sum((arr / 255) * np.log2((arr / 255) + 1e-9))
    }

#Builds the data
def bData(fold):
    ret = []
    for lab in sorted(os.listdir(fold)):
        path = os.path.join(fold, lab)
        if not os.path.isdir(path):
            continue
        for fname in os.listdir(path):
            if fname.endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(path, fname)
                feat = imgGet(fpath)
                feat["label"] = int(lab)
                ret.append(feat)
    return pd.DataFrame(ret)

def mPerf(func, *args, **kwargs):
    pro = psutil.Process(os.getpid())
    mem1 = pro.memory_info().rss / (1024 ** 2)
    cpu1 = pro.cpu_percent(interval=None)
    st = time.time()
    ret = func(*args, **kwargs)
    laten = time.time() - st
    cpu2 = pro.cpu_percent(interval=None)
    mem2 = pro.memory_info().rss / (1024 ** 2)
    return ret, {
        "cpu": cpu2 - cpu1, "memory": mem2 - mem1, "latency": laten
    }

#Prints Data
def printData(nm, fam, yt, yp, train, pred):
    print(f"\n{nm} Malware Family {fam}")
    print(classification_report(yt, yp, zero_division=0))
    cm = confusion_matrix(yt, yp)

    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"True Negative: {tnr:.4f}")
    print(f"False Positive: {fpr:.4f}")
    print(f"False Negative: {fnr:.4f}")
    print(f"True Positive: {tpr:.4f}")

    print("\nTraining Performance")
    print(f"CPU: {train['cpu']:.2f}%")
    print(f"Memory Overhead: {train['memory']:.2f} MB")
    print(f"Latency: {train['latency']:.4f} sec")

    print("\nPrediction Performance")
    print(f"CPU: {pred['cpu']:.2f}%")
    print(f"Memory Overhead: {pred['memory']:.2f} MB")
    print(f"Latency: {pred['latency']:.4f} sec")

#Analizes Data
def runAna(path):
    train = bData(os.path.join(path, "train"))
    test = bData(os.path.join(path, "test"))

    for family in range(1, 10):
        trainSub = train[train["label"].isin([family, 0])].copy()
        testSub = test[test["label"].isin([family, 0])].copy()
        trainSub["label"] = trainSub["label"].map({0: 0, family: 1})
        testSub["label"] = testSub["label"].map({0: 0, family: 1})

        xt, yt = trainSub.drop("label", axis=1), trainSub["label"]
        xte, yte = testSub.drop("label", axis=1), testSub["label"]

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42
        )

        xgb, xgbt = mPerf(xgb.fit, xt, yt)
        xgbp, xgbpp = mPerf(xgb.predict, xte)
        printData("XGBoost", family, yte, xgbp, xgbt, xgbpp)

        svm = SVC(probability=True)
        svm, svmt = mPerf(svm.fit, xt, yt)
        svmp, svmpp = mPerf(svm.predict, xte)
        printData("SVM", family, yte, svmp, svmt, svmpp)


if __name__ == "__main__":
    runAna("./MNIST")