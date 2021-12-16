import pandas as pd


file = r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment2\student_scores.csv"
data = pd.read_csv(file)

print("Row-wise values :- ")

print("Means of each row is : ")
means = data.mean(axis=1)
for i in range(25):
    print(means[i])

print("Medians of each row is : ")
medians = data.median(axis=1)
for i in range(25):
    print(medians[i])

print("Modes of each row is : ")
modes = data.mode(axis=1)[0]
for i in range(25):
    print(modes[i])


print("\nColumn-wise values :- ")

print("Hours column : ")
mean = data["Hours"].mean()
median = data["Hours"].median()
mode = data["Hours"].mode()[0]
print("Mean is {}".format(mean))
print("Median is {}".format(median))
print("Mode is {}".format(mode))

print("Scores column : ")
mean = data["Scores"].mean()
median = data["Scores"].median()
mode = data["Scores"].mode()[0]
print("Mean is {}".format(mean))
print("Median is {}".format(median))
print("Mode is {}".format(mode))


print("\nOverall values :- ")
print("Mean is {}".format(data.stack().mean()))
print("Median is {}".format(data.stack().median()))
print("Mode is {}".format(data.stack().mode()[0]))
