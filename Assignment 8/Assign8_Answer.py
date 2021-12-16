import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


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

scaler = MinMaxScaler()

df_wine[
    [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280/od315_of_diluted_wines",
        "proline",
    ]
] = scaler.fit_transform(
    df_wine[
        [
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280/od315_of_diluted_wines",
            "proline",
        ]
    ]
)

# for i in range(0, 13):
#   maxValue = df_wine.iloc[:, i].max()
#   df_wine.iloc[:, i] = df_wine.iloc[:, i].apply(lambda x: x / maxValue)

# print(df_wine)

# One Hot Enconding

df_wine["c0"] = np.where(df_wine["target"] == 0, 1, 0)
df_wine["c1"] = np.where(df_wine["target"] == 1, 1, 0)
df_wine["c2"] = np.where(df_wine["target"] == 2, 1, 0)
df_wine = df_wine.drop("target", 1)

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
    X_train, X_test = dfC.iloc[train_index, :], dfC.iloc[test_index, :]

    size = X_train.shape[0]
    d = [0] * 3

    for i in range(max_epoch):
        j = 0
        for j in range(size):
            s = 0
            for k in range(13):
                s += W0[k + 1] * X_train.iloc[j, k]

            d[0] = sigmoid(W0[0] * 1 + s)

            s = 0
            for k in range(13):
                s += W1[k + 1] * X_train.iloc[j, k]

            d[1] = sigmoid(W1[0] * 1 + s)

            s = 0
            for k in range(13):
                s += W2[k + 1] * X_train.iloc[j, k]

            d[2] = sigmoid(W2[0] * 1 + s)

            W0[0] = W0[0] + a * (X_train.iloc[j, 13] - d[0]) * 1 * d[0] * (1 - d[0])
            for k in range(1, 13):
                W0[k] = W0[k] + a * (X_train.iloc[j, 13] - d[0]) * X_train.iloc[
                    j, k
                ] * d[0] * (1 - d[0])

            W1[0] = W1[0] + a * (X_train.iloc[j, 14] - d[1]) * 1 * d[1] * (1 - d[1])
            for k in range(1, 13):
                W1[k] = W1[k] + a * (X_train.iloc[j, 14] - d[1]) * X_train.iloc[
                    j, k
                ] * d[1] * (1 - d[1])

            W2[0] = W2[0] + a * (X_train.iloc[j, 15] - d[2]) * 1 * d[2] * (1 - d[2])
            for k in range(1, 13):
                W2[k] = W2[k] + a * (X_train.iloc[j, 15] - d[2]) * X_train.iloc[
                    j, k
                ] * d[2] * (1 - d[2])
