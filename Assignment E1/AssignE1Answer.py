import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

boston_data = datasets.load_boston(return_X_y=False)
X = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
y = pd.Series(boston_data.target)

train_size = 0.3

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.3, shuffle=False)

test_size = 0.8571
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rem, y_rem, test_size=0.8571, shuffle=False
)

# Select alpha, rho and max epochs here
a = 0.1
r = 0.5
max = 10

# How many elements to remove from the begining of the plot (just to make the plot prettier, since the initial values are pretty abrupt).
p = 5


data = X_train.copy()
output = y_train.copy()
size = data.shape[0]
W = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
h = [0] * size
Jp = 10000000.0
Jc = 0.0
flag = 0
list1 = []

for k in range(max):

    # print(W)
    for i in range(size):
        h[i] = (
            W[0] * 1
            + W[1] * data.at[i, "CRIM"]
            + W[2] * data.at[i, "ZN"]
            + W[3] * data.at[i, "INDUS"]
            + W[4] * data.at[i, "CHAS"]
            + W[5] * data.at[i, "NOX"]
            + W[6] * data.at[i, "RM"]
            + W[7] * data.at[i, "AGE"]
            + W[8] * data.at[i, "DIS"]
            + W[9] * data.at[i, "RAD"]
            + W[10] * data.at[i, "TAX"]
            + W[11] * data.at[i, "PTRATIO"]
            + W[12] * data.at[i, "B"]
            + W[13] * data.at[i, "LSTAT"]
        )

    for i in range(size):
        Jc += (h[i] - output[i]) ** 2

    Jc = (1 / (2 * size)) * Jc
    # print(Jc)

    list1.append(Jc)

    if abs(Jc - Jp) < r:
        print("Training set MSE : ", Jc)
        flag = 1
        break

    # if Jc > Jp:
    #     print("MSE does not converge")
    #     flag = 1
    #     break

    Jp = Jc

    # For W[0]

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * 1

    W[0] = W[0] - a * (1 / size) * s

    # For W[1]

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "CRIM"]

    W[1] = W[1] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "ZN"]

    W[2] = W[2] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "INDUS"]

    W[3] = W[3] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "CHAS"]

    W[4] = W[4] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "NOX"]

    W[5] = W[5] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "RM"]

    W[6] = W[6] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "AGE"]

    W[7] = W[7] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "DIS"]

    W[8] = W[8] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "RAD"]

    W[9] = W[9] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "TAX"]

    W[10] = W[10] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "PTRATIO"]

    W[11] = W[11] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "B"]

    W[12] = W[12] - a * (1 / size) * s

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "LSTAT"]

    W[13] = W[13] - a * (1 / size) * s


if flag == 0:
    print("Training set MSE : ", Jc)

list2 = [t for t in range(len(list1))]
plt.plot(list2[p:], list1[p:])
plt.show()


# Validation set

data = X_valid.copy()
output = y_valid.copy()
size = data.shape[0]
Jc = 0
# print(data)

for i in range(size):
    h[i] = (
        W[0] * 1
        + W[1] * data.at[i + 151, "CRIM"]
        + W[2] * data.at[i + 151, "ZN"]
        + W[3] * data.at[i + 151, "INDUS"]
        + W[4] * data.at[i + 151, "CHAS"]
        + W[5] * data.at[i + 151, "NOX"]
        + W[6] * data.at[i + 151, "RM"]
        + W[7] * data.at[i + 151, "AGE"]
        + W[8] * data.at[i + 151, "DIS"]
        + W[9] * data.at[i + 151, "RAD"]
        + W[10] * data.at[i + 151, "TAX"]
        + W[11] * data.at[i + 151, "PTRATIO"]
        + W[12] * data.at[i + 151, "B"]
        + W[13] * data.at[i + 151, "LSTAT"]
    )
    # h[i] = 1

for i in range(size):
    Jc += (h[i] - output[i + 151]) ** 2

Jc = (1 / (2 * size)) * Jc
print("Validation set MSE : ", Jc)


# Test set

data = X_test.copy()
output = y_test.copy()
size = data.shape[0]
Jc = 0
print(size)

for i in range(size):
    h.append(
        W[0] * 1
        + W[1] * data.at[i + 201, "CRIM"]
        + W[2] * data.at[i + 201, "ZN"]
        + W[3] * data.at[i + 201, "INDUS"]
        + W[4] * data.at[i + 201, "CHAS"]
        + W[5] * data.at[i + 201, "NOX"]
        + W[6] * data.at[i + 201, "RM"]
        + W[7] * data.at[i + 201, "AGE"]
        + W[8] * data.at[i + 201, "DIS"]
        + W[9] * data.at[i + 201, "RAD"]
        + W[10] * data.at[i + 201, "TAX"]
        + W[11] * data.at[i + 201, "PTRATIO"]
        + W[12] * data.at[i + 201, "B"]
        + W[13] * data.at[i + 201, "LSTAT"]
    )


for i in range(size):
    Jc += (h[i] - output[i + 201]) ** 2

Jc = (1 / (2 * size)) * Jc
print("Test set MSE : ", Jc)
