import pandas as pd
import numpy as np
from PIL import Image
import os
import psutil
import time
import shutil

def api2png(api, width=64):
    hold = {api: i for i, api in enumerate(set(api))}
    arr = np.array([hold[api] for api in api], dtype=np.uint8)
    height = int(np.ceil(len(arr) / width))
    arr = np.pad(arr, (0, height * width - len(arr)), mode="constant")
    arr = arr.reshape((height, width))
    return Image.fromarray(arr, mode="L")

def csvbatches(csv, dir, bs=1000, width=64):
    df = pd.read_csv(csv)

    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

    process = psutil.Process()

    for start in range(0, len(df), bs):
        end = min(start + bs, len(df))
        batch = df.iloc[start:end]

        memb = process.memory_info().rss / (1024**2)
        cpu_times_before = process.cpu_times()
        cpu_time_before = cpu_times_before.user + cpu_times_before.system

        t0 = time.perf_counter()

        for idx, row in batch.iterrows():
            api = [str(x) for x in row.dropna().tolist()]
            if not api:
                continue
            img = api2png(api, width=width)
            img.save(os.path.join(dir, f"sample_{idx}.png"))

        latency = time.perf_counter() - t0

        cputa = process.cpu_times()
        cputa = cputa.user + cputa.system
        cpup = ((cputa - cpu_time_before) / latency) * 100 if latency > 0 else 0.0

        mema = process.memory_info().rss / (1024**2)

        print(f"Batch {start//bs + 1}:")
        print(f"Samples processed: {end - start}")
        print(f"CPU: {cpup:.2f}%")
        print(f"Memory: {mema - memb:.2f} MB")
        print(f"Latency: {latency:.4f} sec\n")


#Main
if __name__ == "__main__":
    dir = "./api_sequences_malware_datasets"
    samp = os.path.join(dir, "VirusSample.csv")
    share = os.path.join(dir, "VirusShare.csv")
    odir = os.path.join(dir, "newApiPng")
    csvbatches(samp, odir)