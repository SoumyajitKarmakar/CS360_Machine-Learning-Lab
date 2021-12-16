import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df["target"] = pd.Series(sklearn_dataset.target)
    return df


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def maximumF(a, b, c):
    if (a >= b) and (a >= c):
        largest = a
        fl = 0
    elif (b >= a) and (b >= c):
        largest = b
        fl = 1
    else:
        largest = c
        fl = 2

    return fl


df_iris = sklearn_to_df(datasets.load_iris())

df_iris = shuffle(df_iris)
df_iris.reset_index(inplace=True, drop=True)

# print(df_iris)

# Normalising the dataframe.

for i in range(4):
    maxValue = df_iris.iloc[:, i].max()
    df_iris.iloc[:, i] = df_iris.iloc[:, i].apply(lambda x: x / maxValue)

# print(df_iris)


ts = 0.2

X_train, X_test = train_test_split(df_iris, test_size=ts)

X_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)

# print(X_test)

max_epoch = 10

W0 = [0.0 for asd in range(5)]
W1 = [0.0 for asd in range(5)]
W2 = [0.0 for asd in range(5)]

AccM = 0.0

PreM = [0.0] * 3
ReM = [0.0] * 3

X_train["target"] = np.where(X_train["target"] == 0, 1, 0)
size = X_train.shape[0]
h = 0

for i in range(max_epoch):
    Wc = W0.copy()
    for j in range(size):
        sum = 0
        for k in range(4):
            sum += W0[k + 1] * X_train.iloc[j, k]

        h = sigmoid(W0[0] * 1 + sum)
        # print(h)

        W0[0] = W0[0] + (X_train.iloc[j, 4] - h) * 1

        for k in range(1, 5):
            W0[k] = W0[k] + (X_train.iloc[j, 4] - h) * X_train.iloc[j, k]

        # print(W0)

    if Wc == W0:
        break

X_train["target"] = np.where(X_train["target"] == 1, 1, 0)
size = X_train.shape[0]
h = 0

for i in range(max_epoch):
    Wc = W1.copy()
    for j in range(size):
        sum = 0
        for k in range(4):
            sum += W1[k + 1] * X_train.iloc[j, k]

        h = sigmoid(W1[0] * 1 + sum)
        # print(h)

        W1[0] = W1[0] + (X_train.iloc[j, 4] - h) * 1

        for k in range(1, 5):
            W1[k] = W1[k] + (X_train.iloc[j, 4] - h) * X_train.iloc[j, k]

        # print(W1)

    if Wc == W1:
        break

X_train["target"] = np.where(X_train["target"] == 2, 1, 0)
size = X_train.shape[0]
h = 0

for i in range(max_epoch):
    Wc = W2.copy()
    for j in range(size):
        sum = 0
        for k in range(4):
            sum += W2[k + 1] * X_train.iloc[j, k]

        h = sigmoid(W2[0] * 1 + sum)
        # print(h)

        W2[0] = W2[0] + (X_train.iloc[j, 4] - h) * 1

        for k in range(1, 5):
            W2[k] = W2[k] + (X_train.iloc[j, 4] - h) * X_train.iloc[j, k]

        # print(W2)

    if Wc == W2:
        break


# Test set

size = X_test.shape[0]
h_test = [0.0] * size
C0 = 0.0
C1 = 0.0
C2 = 0.0

for j in range(size):
    sum = 0
    for k in range(4):
        sum += W0[k + 1] * X_test.iloc[j, k]

    C0 = sigmoid(W0[0] * 1 + sum)

    sum = 0
    for k in range(4):
        sum += W1[k + 1] * X_test.iloc[j, k]

    C1 = sigmoid(W1[0] * 1 + sum)

    sum = 0
    for k in range(4):
        sum += W2[k + 1] * X_test.iloc[j, k]

    C2 = sigmoid(W2[0] * 1 + sum)

    # print(C0, C1, C2)

    h_test[j] = maximumF(C0, C1, C2)

# print(h_test)
# print(X_test.iloc[:, 13])
# print("Wait")

# input("Wait")

CM = [[0 for asdd in range(3)] for fgh in range(3)]

for j in range(size):
    if X_test.iloc[j, 4] == 0:
        if h_test[j] == 0:
            CM[0][0] += 1
        elif h_test[j] == 1:
            CM[1][0] += 1
        elif h_test[j] == 2:
            CM[2][0] += 1
    elif X_test.iloc[j, 4] == 1:
        if h_test[j] == 0:
            CM[0][1] += 1
        elif h_test[j] == 1:
            CM[1][1] += 1
        elif h_test[j] == 2:
            CM[2][1] += 1
    elif X_test.iloc[j, 4] == 2:
        if h_test[j] == 0:
            CM[0][2] += 1
        elif h_test[j] == 1:
            CM[1][2] += 1
        elif h_test[j] == 2:
            CM[2][2] += 1

# print(CM)
print()

AccM = (CM[0][0] + CM[1][1] + CM[2][2]) / (
    CM[0][0]
    + CM[0][1]
    + CM[0][2]
    + CM[1][0]
    + CM[1][1]
    + CM[1][2]
    + CM[2][0]
    + CM[2][1]
    + CM[2][2]
)

print("Accuracy : ", AccM)

if (CM[0][0] + CM[0][1] + CM[0][2]) != 0:
    PreM[0] = (CM[0][0]) / (CM[0][0] + CM[0][1] + CM[0][2])
else:
    PreM[0] = None
if (CM[1][0] + CM[1][1] + CM[1][2]) != 0:
    PreM[1] = (CM[1][1]) / (CM[1][0] + CM[1][1] + CM[1][2])
else:
    PreM[1] = None
if (CM[2][0] + CM[2][1] + CM[2][2]) != 0:
    PreM[2] = (CM[2][2]) / (CM[2][0] + CM[2][1] + CM[2][2])
else:
    PreM[2] = None

print("Presision of the 3 classes : ", PreM)

if (CM[0][0] + CM[1][0] + CM[2][0]) != 0:
    ReM[0] = (CM[0][0]) / (CM[0][0] + CM[1][0] + CM[2][0])
else:
    ReM[0] = None
if (CM[0][1] + CM[1][1] + CM[2][1]) != 0:
    ReM[1] = (CM[1][1]) / (CM[0][1] + CM[1][1] + CM[2][1])
else:
    ReM[1] = None
if (CM[0][2] + CM[1][2] + CM[2][2]) != 0:
    ReM[2] = (CM[2][2]) / (CM[0][2] + CM[1][2] + CM[2][2])
else:
    ReM[2] = None

print("Recall of the 3 classes : ", ReM)
