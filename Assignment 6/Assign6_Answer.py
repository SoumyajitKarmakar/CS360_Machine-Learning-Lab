import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df["target"] = pd.Series(sklearn_dataset.target)
    return df


df_iris = sklearn_to_df(datasets.load_iris())

df_iris.drop(df_iris[df_iris["target"] == 2].index, inplace=True)

df_iris = shuffle(df_iris)
# df_iris.reset_index(inplace=True, drop=True)


# Normalising the dataframe.
max = df_iris["sepal length (cm)"].max()
df_iris["sepal length (cm)"] = df_iris["sepal length (cm)"].apply(lambda x: x / max)

max = df_iris["sepal width (cm)"].max()
df_iris["sepal width (cm)"] = df_iris["sepal width (cm)"].apply(lambda x: x / max)

max = df_iris["petal length (cm)"].max()
df_iris["petal length (cm)"] = df_iris["petal length (cm)"].apply(lambda x: x / max)

max = df_iris["petal width (cm)"].max()
df_iris["petal width (cm)"] = df_iris["petal width (cm)"].apply(lambda x: x / max)

# df_iris


# Splitting the dataframe.
val_size = 0.3
test_size = 0.2

X_train, X_remain = train_test_split(df_iris, test_size=(val_size + test_size))
new_test_size = np.around(test_size / (val_size + test_size), 2)
new_val_size = 1.0 - new_test_size

X_val, X_test = train_test_split(X_remain, test_size=new_test_size)

X_train.reset_index(inplace=True, drop=True)
X_val.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)

# Set these 4 hyperparameters before each run.
a = 0.1
r = 0.001
max_epoch = 100
W = [0.3, 0.3, 0.3, 0.3, 0.3]

size = X_train.shape[0]
h_train = [0] * size
s = 0.0
Jc = 0.0
Jp = 100000.0
flag = 0

# Uncomment this only during the training of the finalised model. p denotes the number of elements to remove from the begining of the plot to make it prettier.
list1 = []
p = 0

for i in range(max_epoch):
    for i in range(size):
        h_train[i] = sigmoid(
            W[0] * 1
            + W[1] * X_train.at[i, "sepal length (cm)"]
            + W[2] * X_train.at[i, "sepal width (cm)"]
            + W[3] * X_train.at[i, "petal length (cm)"]
            + W[4] * X_train.at[i, "petal width (cm)"]
        )

    Jc = 0.0
    for i in range(size):
        Jc += X_train.at[i, "target"] * np.log(h_train[i]) + (
            1 - X_train.at[i, "target"]
        ) * np.log(1 - h_train[i])
    Jc = -1 * (Jc / size)
    # print(Jc)

    if abs(Jc - Jp) < r:
        print("Training set error : ", Jc)
        flag = 1
        break

    if Jc > Jp:
        print("Model does not converge")
        flag = 1
        break

    Jp = Jc

    # Uncomment this only during the training of the finalised model.
    list1.append(Jc)

    # For W[0]
    s = 0.0
    for i in range(size):
        s += (X_train.at[i, "target"] - h_train[i]) * 1

    W[0] = W[0] + a * (1 / size) * s

    # For W[1]
    s = 0.0
    for i in range(size):
        s += (X_train.at[i, "target"] - h_train[i]) * X_train.at[i, "sepal length (cm)"]

    W[1] = W[1] + a * (1 / size) * s

    # For W[2]
    s = 0.0
    for i in range(size):
        s += (X_train.at[i, "target"] - h_train[i]) * X_train.at[i, "sepal width (cm)"]

    W[2] = W[2] + a * (1 / size) * s

    # For W[3]
    s = 0.0
    for i in range(size):
        s += (X_train.at[i, "target"] - h_train[i]) * X_train.at[i, "petal length (cm)"]

    W[3] = W[3] + a * (1 / size) * s

    # For W[4]
    s = 0.0
    for i in range(size):
        s += (X_train.at[i, "target"] - h_train[i]) * X_train.at[i, "petal width (cm)"]

    W[4] = W[4] + a * (1 / size) * s

if flag == 0:
    print("Training set error : ", Jc)

list2 = [t for t in range(len(list1))]
plt.plot(list2[p:], list1[p:])
plt.show()

# Validation set

size = X_val.shape[0]
h_val = [0] * size

for i in range(size):
    h_val[i] = sigmoid(
        W[0] * 1
        + W[1] * X_val.at[i, "sepal length (cm)"]
        + W[2] * X_val.at[i, "sepal width (cm)"]
        + W[3] * X_val.at[i, "petal length (cm)"]
        + W[4] * X_val.at[i, "petal width (cm)"]
    )

Jc = 0.0
for i in range(size):
    Jc += X_val.at[i, "target"] * np.log(h_val[i]) + (
        1 - X_val.at[i, "target"]
    ) * np.log(1 - h_val[i])

Jc = -1 * (Jc / size)

print("Validation set error : ", Jc)

# Test set

size = X_test.shape[0]
h_test = [0] * size

for i in range(size):
    h_test[i] = sigmoid(
        W[0] * 1
        + W[1] * X_test.at[i, "sepal length (cm)"]
        + W[2] * X_test.at[i, "sepal width (cm)"]
        + W[3] * X_test.at[i, "petal length (cm)"]
        + W[4] * X_test.at[i, "petal width (cm)"]
    )

Jc = 0.0
for i in range(size):
    Jc += X_test.at[i, "target"] * np.log(h_test[i]) + (
        1 - X_test.at[i, "target"]
    ) * np.log(1 - h_test[i])

Jc = -1 * (Jc / size)

print("Test set error : ", Jc)
