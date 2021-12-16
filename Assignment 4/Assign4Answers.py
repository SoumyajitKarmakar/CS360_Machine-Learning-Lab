import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment4\DataSet.csv")

train_size = 0.5

X = df.drop(columns=["Y"]).copy()
y = df["Y"]

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.5, shuffle=False)

test_size = 0.6
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rem, y_rem, test_size=0.6, shuffle=False
)


# Select alpha, rho and max epochs here
a = 0.001
r = 0.001
max = 10000

# How many elements to remove from the begining of the plot (just to make the plot prettier, since the initial values are pretty abrupt).
p = 5


data = X_train.copy()
output = y_train.copy()
size = data.shape[0]
W = [1, 1]
h = [0] * size
Jp = 10000000.0
Jc = 0.0
flag = 0
list1 = []

for k in range(max):

    # print(W)
    for i in range(size):
        h[i] = W[0] * 1 + W[1] * data.at[i, "X"]

    for i in range(size):
        Jc += (h[i] - output[i]) ** 2

    Jc = (1 / (2 * size)) * Jc
    # print(Jc)

    list1.append(Jc)

    if abs(Jc - Jp) < r:
        print("Training set MSE : ", Jc)
        flag = 1
        break

    if Jc > Jp:
        print("MSE does not converge")
        flag = 1
        break

    Jp = Jc

    # For W[0]

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * 1

    W[0] = W[0] - a * (1 / size) * s

    # For W[1]

    s = 0.0
    for i in range(size):
        s += (h[i] - output[i]) * data.at[i, "X"]

    W[1] = W[1] - a * (1 / size) * s

if flag == 0:
    print("Training set MSE : ", Jc)

list2 = [t for t in range(len(list1))]
plt.plot(list2[p:], list1[p:])
plt.show()

# For training data with rho as 0.001,
# When alpha = 0.1, MSE does not converge
# When alpha = 0.001, MSE conerges to, 393.79391423822346
# When alpha = 0.0001, MSE converges to, 401.15557518802973
# When alpha = 0.05, MSE does not converge
# When alpha = 1, MSE does not converge


# Validation set

data = X_valid.copy()
output = y_valid.copy()
size = data.shape[0]


for i in range(size):
    h[i] = W[0] * 1 + W[1] * data.at[i + 31, "X"]

for i in range(size):
    Jc += (h[i] - output[i + 31]) ** 2

Jc = (1 / (2 * size)) * Jc
print("Validation set MSE : ", Jc)

# For validation data,
# When alpha = 0.1, MSE is, 57395847.61678633
# When alpha = 0.001, MSE is, 417.0361655220013
# When alpha = 0.0001, MSE is, 440.74625342277557
# When alpha = 0.05, MSE is, 14116254.07535305
# When alpha = 1, MSE is, 5824091139.376694


# Test set

data = X_test.copy()
output = y_test.copy()
size = data.shape[0]

for i in range(size):
    h[i] = W[0] * 1 + W[1] * data.at[i + 43, "X"]

for i in range(size):
    Jc += (h[i] - output[i + 43]) ** 2

Jc = (1 / (2 * size)) * Jc
print("Test set MSE : ", Jc)

# For test data,
# When alpha = 0.1, MSE is, 45934793.66175249
# When alpha = 0.001, MSE is, 1286.0130764796413
# When alpha = 0.0001, MSE is, 1343.7383376717517
# When alpha = 0.05, MSE is, 11263742.770794854
# When alpha = 1, MSE is, 4673606528.374931
