import pandas as pd
import numpy as np
df = pd.read_csv("data/soc-sign-bitcoinotc.csv", header=None)
print(df.head())
df.columns = ["n1", "n2", "w", "timestamp"]

n1_sorted = df["n1"].astype(int).to_numpy()
n2_sorted = df["n2"].astype(int).to_numpy()

srt = np.concatenate((n1_sorted, n2_sorted))#print(df.head())
srt.sort()
print(len(np.unique(srt)))
i = 1
prev = srt[0]
dicti = {1:i}
for elem in srt:
    if elem != prev:

        dicti[elem] = i
        prev = elem
        i+=1

df[["n1", "n2"]] = df[["n1", "n2"]].replace(dicti)

print(df.head())
df.to_csv("data/bitcoin_updated.csv", header = False, index = False)