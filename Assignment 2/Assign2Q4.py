import pandas as pd


file = r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment2\student_scores.csv"
data = pd.read_csv(file)


def func(n):
    return 5 * n - 9


hours = data["Hours"]
scores = data["Scores"]

sqSum = 0
for i in range(1, 25):
    sqSum += (scores[i] - func(hours[i])) ** 2

print("The sum square is {}".format(sqSum))
