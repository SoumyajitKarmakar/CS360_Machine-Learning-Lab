import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold


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


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df["target"] = pd.Series(sklearn_dataset.target)
    return df


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


df_wine = sklearn_to_df(datasets.load_wine())

df_wine = shuffle(df_wine)
df_wine.reset_index(inplace=True, drop=True)


# Normalising the dataframe.

for i in range(0, 13):
    maxValue = df_wine.iloc[:, i].max()
    df_wine.iloc[:, i] = df_wine.iloc[:, i].apply(lambda x: x / maxValue)

# print(df_wine)

dfC = df_wine.copy(deep=True)


# Set these 4 hyperparameters before each run.
a = 0.1
r = 0.0001
max_epoch = 100
W0 = [0.1] * 14
W1 = [0.1] * 14
W2 = [0.1] * 14

p = -1
s = 0.0
Jc = 0.0
Jp = 100000.0
flag = 0


OAcc = 0.0
AccM = [0.0] * 5

PreM = [0.0] * 3
ReM = [0.0] * 3

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(dfC):

    p += 1
    dfC["target"] = np.where(dfC["target"] == 0, 1, 0)
    X_train, X_test = dfC.iloc[train_index, :], dfC.iloc[test_index, :]

    size = X_train.shape[0]
    h_train = [0] * size

    for i in range(max_epoch):
        for j in range(size):
            sum = 0
            for k in range(0, 13):
                sum += W0[k + 1] * X_train.iloc[j, k]

            h_train[j] = sigmoid(W0[0] * 1 + sum)
            # print(h_train)

        Jc = 0.0
        for j in range(size):
            Jc += X_train.iloc[j, 13] * np.log(h_train[j]) + (
                1 - X_train.iloc[j, 13]
            ) * np.log(1 - h_train[j])
        Jc = -1 * (Jc / size)
        # print(Jc)

        if abs(Jc - Jp) < r:
            # print("Training set error : ", Jc)
            flag = 1
            break

        # if Jc > Jp:
        #   print("Model does not converge")
        #   flag = 1
        #   break

        Jp = Jc

        # print(W0)

        # For W[0]
        s = 0.0
        for k in range(size):
            s += (X_train.iloc[k, 13] - h_train[k]) * 1

        W0[0] = W0[0] + a * (1 / size) * s

        for j in range(1, 14):
            s = 0.0
            for k in range(size):
                s += (X_train.iloc[k, 13] - h_train[k]) * X_train.iloc[k, j - 1]

            W0[j] = W0[j] + a * (1 / size) * s

    # print(W0)

    dfC = df_wine.copy(deep=True)
    dfC["target"] = np.where(dfC["target"] == 1, 1, 0)
    X_train, X_test = dfC.iloc[train_index, :], dfC.iloc[test_index, :]

    size = X_train.shape[0]
    h_train = [0] * size

    for i in range(max_epoch):
        for j in range(size):
            sum = 0
            for k in range(0, 13):
                sum += W1[k + 1] * X_train.iloc[j, k]

            h_train[j] = sigmoid(W1[0] * 1 + sum)
            # print(h_train)

        Jc = 0.0
        for j in range(size):
            Jc += X_train.iloc[j, 13] * np.log(h_train[j]) + (
                1 - X_train.iloc[j, 13]
            ) * np.log(1 - h_train[j])
        Jc = -1 * (Jc / size)
        # print(Jc)

        if abs(Jc - Jp) < r:
            # print("Training set error : ", Jc)
            flag = 1
            break

        # if Jc > Jp:
        #   print("Model does not converge")
        #   flag = 1
        #   break

        Jp = Jc

        # print(W0)

        # For W[0]
        s = 0.0
        for k in range(size):
            s += (X_train.iloc[k, 13] - h_train[k]) * 1

        W1[0] = W1[0] + a * (1 / size) * s

        for j in range(1, 14):
            s = 0.0
            for k in range(size):
                s += (X_train.iloc[k, 13] - h_train[k]) * X_train.iloc[k, j - 1]

            W1[j] = W1[j] + a * (1 / size) * s

    # print(W1)

    dfC = df_wine.copy(deep=True)
    dfC["target"] = np.where(dfC["target"] == 2, 1, 0)
    X_train, X_test = dfC.iloc[train_index, :], dfC.iloc[test_index, :]

    size = X_train.shape[0]
    h_train = [0] * size

    for i in range(max_epoch):
        for j in range(size):
            sum = 0
            for k in range(0, 13):
                sum += W2[k + 1] * X_train.iloc[j, k]

            h_train[j] = sigmoid(W2[0] * 1 + sum)
            # print(h_train)

        Jc = 0.0
        for j in range(size):
            Jc += X_train.iloc[j, 13] * np.log(h_train[j]) + (
                1 - X_train.iloc[j, 13]
            ) * np.log(1 - h_train[j])
        Jc = -1 * (Jc / size)
        # print(Jc)

        if abs(Jc - Jp) < r:
            # print("Training set error : ", Jc)
            flag = 1
            break

        # if Jc > Jp:
        #   print("Model does not converge")
        #   flag = 1
        #   break

        Jp = Jc

        # print(W0)

        # For W[0]
        s = 0.0
        for k in range(size):
            s += (X_train.iloc[k, 13] - h_train[k]) * 1

        W2[0] = W2[0] + a * (1 / size) * s

        for j in range(1, 14):
            s = 0.0
            for k in range(size):
                s += (X_train.iloc[k, 13] - h_train[k]) * X_train.iloc[k, j - 1]

            W2[j] = W2[j] + a * (1 / size) * s

    # print(W2)
    # input("Wait")

    # Test set

    dfC = df_wine.copy(deep=True)
    # dfC['target'] = np.where(dfC['target'] == 2, 1, 0)
    X_train, X_test = dfC.iloc[train_index, :], dfC.iloc[test_index, :]

    size = X_test.shape[0]
    h_test = [0.0] * size
    C0 = 0.0
    C1 = 0.0
    C2 = 0.0

    for j in range(size):
        sum = 0
        for k in range(0, 13):
            sum += W0[k + 1] * X_test.iloc[j, k]

        C0 = sigmoid(W0[0] * 1 + sum)

        sum = 0
        for k in range(0, 13):
            sum += W1[k + 1] * X_test.iloc[j, k]

        C1 = sigmoid(W1[0] * 1 + sum)

        sum = 0
        for k in range(0, 13):
            sum += W2[k + 1] * X_test.iloc[j, k]

        C2 = sigmoid(W2[0] * 1 + sum)

        # print(C0, C1, C2)

        h_test[j] = maximumF(C0, C1, C2)

    # print(h_test)
    # print(X_test.iloc[:, 13])
    # print("Wait")

    # input("Wait")

    CM = [[0 for asd in range(3)] for fgh in range(3)]

    for j in range(size):
        if X_test.iloc[j, 13] == 0:
            if h_test[j] == 0:
                CM[0][0] += 1
            elif h_test[j] == 1:
                CM[1][0] += 1
            elif h_test[j] == 2:
                CM[2][0] += 1
        elif X_test.iloc[j, 13] == 1:
            if h_test[j] == 0:
                CM[0][1] += 1
            elif h_test[j] == 1:
                CM[1][1] += 1
            elif h_test[j] == 2:
                CM[2][1] += 1
        elif X_test.iloc[j, 13] == 2:
            if h_test[j] == 0:
                CM[0][2] += 1
            elif h_test[j] == 1:
                CM[1][2] += 1
            elif h_test[j] == 2:
                CM[2][2] += 1

    # print(CM)
    print()

    AccM[p] = (CM[0][0] + CM[1][1] + CM[2][2]) / (
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

    print("Fold ", p + 1, " Accuracy : ", AccM[p])

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

    print("Fold ", p + 1, " Presision : ", PreM)

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

    print("Fold ", p + 1, " Recall : ", ReM)


print()
OAcc = (AccM[0] + AccM[1] + AccM[2] + AccM[3] + AccM[4]) / 5
print("Overall accuracy is ", OAcc)
