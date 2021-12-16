import pandas as pd
from sklearn.utils import shuffle


def actFunc(n):
    if n >= 0:
        return 1
    else:
        return 0


df = pd.read_csv(r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment 7\orGate.csv")

df = shuffle(df)
df.reset_index(inplace=True, drop=True)
print(df)

W = [0.0 for asd in range(3)]
h = 0

max_epoch = 5

print(W)

for i in range(max_epoch):
    Wc = W.copy()
    for j in range(4):
        sum = 0
        for k in range(2):
            sum += W[k + 1] * df.iloc[j, k]

        h = actFunc(W[0] * 1 + sum)
        print(h)

        W[0] = W[0] + (df.iloc[j, 2] - h) * 1

        for k in range(1, 3):
            W[k] = W[k] + (df.iloc[j, 2] - h) * df.iloc[j, k]

        print(W)

    if Wc == W:
        break
