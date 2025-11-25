import os
import time
import psutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Img getter
def extract_features(imgp):
    arr = np.array(Image.open(imgp).convert("L"))
    return {
        "mean": np.mean(arr), "std": np.std(arr), "max": np.max(arr), "min": np.min(arr), "nonzero": np.count_nonzero(arr), "entropy": -np.sum((arr / 255) * np.log2((arr / 255) + 1e-9))
    }

#Builds Data
def bData(fold):
    ret = []
    for fname in os.listdir(fold):
        if not fname.endswith(".png"):
            continue
        path = os.path.join(fold, fname)
        feat = extract_features(path)
        feat["label"] = 0 if fname.startswith("benign_") else 1
        ret.append(feat)
    return pd.DataFrame(ret)

#Measures sys prefomance
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
def printData(name, yt, yp, train, pred):
    print(f"\n{name}")
    print(classification_report(yt, yp))

    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()

    tnr = tn / (tn + fp) 
    fpr = fp / (fp + tn) 
    fnr = fn / (fn + tp) 
    tpr = tp / (tp + fn) 

    print(f"True Negative: {tnr:.4f}")
    print(f"False Positive: {fpr:.4f}")
    print(f"False Negative: {fnr:.4f}")
    print(f"True Positive: {tpr:.4f}")

    print("\nTraining Performance")
    print(f"CPU: {train['cpu']:.2f}%")
    print(f"Memory Overhead: {train['memory']:.2f} MB")
    print(f"Latency: {train['latency']:.2f} sec")

    print("\nPrediction Performance")
    print(f"CPU: {pred['cpu']:.2f}%")
    print(f"Memory Overhead: {pred['memory']:.2f} MB")
    print(f"Latency: {pred['latency']:.2f} sec")

def runAna(path):
    df = bData(path)
    X = df.drop("label", axis=1)
    y = df["label"]

    Xt, Xte, yt, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBClassifier()
    xgb, xgbt = mPerf(xgb.fit, Xt, yt)
    xgbp, xgbpp = mPerf(xgb.predict, Xte)
    printData("XGBoost", yte, xgbp, xgbt, xgbpp)

    svm = SVC()
    svm, svmt = mPerf(svm.fit, Xt, yt)
    svmp, svmpp = mPerf(svm.predict, Xte)
    printData("SVM", yte, svmp, svmt, svmpp)

#Main
if __name__ == "__main__":
    runAna("./Dataset")