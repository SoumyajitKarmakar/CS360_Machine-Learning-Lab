import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


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

# print(df_iris.columns)

# sns.pairplot(df_iris, hue="target", height = 2, palette = 'colorblind');

# Normalising the dataframe.

scaler = MinMaxScaler()

df_iris[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
] = scaler.fit_transform(
    df_iris[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    ]
)

# print(df_iris)

# One Hot Enconding

df_orig = df_iris.copy(deep=True)

df_iris["c0"] = np.where(df_iris["target"] == 0, 1, 0)
df_iris["c1"] = np.where(df_iris["target"] == 1, 1, 0)
df_iris["c2"] = np.where(df_iris["target"] == 2, 1, 0)
df_iris = df_iris.drop("target", 1)

# print(df_iris)

dfC = df_iris.copy(deep=True)

# Set these 4 hyperparameters before each run.
a = 0.1
r = 0.001
max_epoch = 50
W0 = [0] * 5
W1 = [0.3] * 5
W2 = [0.1] * 5

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
    X_train, X_test = dfC.iloc[train_index, :], dfC.iloc[test_index, :]

    # Train Set

    size = X_train.shape[0]
    d = [0] * 3

    for i in range(max_epoch):
        j = 0
        # print(W0)
        # print(W1)
        # print(W2)
        for j in range(size):
            s = 0
            for k in range(4):
                s += W0[k + 1] * X_train.iloc[j, k]

            d[0] = sigmoid(W0[0] * 1 + s)

            s = 0
            for k in range(4):
                s += W1[k + 1] * X_train.iloc[j, k]

            d[1] = sigmoid(W1[0] * 1 + s)

            s = 0
            for k in range(4):
                s += W2[k + 1] * X_train.iloc[j, k]

            d[2] = sigmoid(W2[0] * 1 + s)

            W0[0] = W0[0] + a * (X_train.iloc[j, 4] - d[0]) * 1 * d[0] * (1 - d[0])
            for k in range(1, 5):
                W0[k] = W0[k] + a * (X_train.iloc[j, 4] - d[0]) * X_train.iloc[
                    j, k
                ] * d[0] * (1 - d[0])

            W1[0] = W1[0] + a * (X_train.iloc[j, 5] - d[1]) * 1 * d[1] * (1 - d[1])
            for k in range(1, 5):
                W1[k] = W1[k] + a * (X_train.iloc[j, 5] - d[1]) * X_train.iloc[
                    j, k
                ] * d[1] * (1 - d[1])

            W2[0] = W2[0] + a * (X_train.iloc[j, 6] - d[2]) * 1 * d[2] * (1 - d[2])
            for k in range(1, 5):
                W2[k] = W2[k] + a * (X_train.iloc[j, 6] - d[2]) * X_train.iloc[
                    j, k
                ] * d[2] * (1 - d[2])

    # Test Set

    size = X_test.shape[0]
    h_test = [0.0] * size
    C0 = 0.0
    C1 = 0.0
    C2 = 0.0

    # print(X_test)

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

    print(h_test)
    CM = [[0 for asdd in range(3)] for fgh in range(3)]

    for j in range(size):
        if X_test.iloc[j, 4] == 1:
            if h_test[j] == 0:
                CM[0][0] += 1
            elif h_test[j] == 1:
                CM[1][0] += 1
            elif h_test[j] == 2:
                CM[2][0] += 1
        elif X_test.iloc[j, 5] == 1:
            if h_test[j] == 0:
                CM[0][1] += 1
            elif h_test[j] == 1:
                CM[1][1] += 1
            elif h_test[j] == 2:
                CM[2][1] += 1
        elif X_test.iloc[j, 6] == 1:
            if h_test[j] == 0:
                CM[0][2] += 1
            elif h_test[j] == 1:
                CM[1][2] += 1
            elif h_test[j] == 2:
                CM[2][2] += 1

    print(CM)

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

    print("Accuracy of fold ", p, " : ", AccM[p])

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

    print("Presision of the 3 classes in fold ", p, " : ", PreM)

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

    print("Recall of the 3 classes in fold ", p, " : ", ReM)

OAcc = (AccM[0] + AccM[1] + AccM[2] + AccM[3] + AccM[4]) / 5

print("Overall Accuracy is : ", OAcc)
