import matplotlib.pyplot as plt
import csv

x = []
y = []

with open(
    r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment3\student_scores.csv", "r"
) as csvfile:
    csvfile.readline()
    lines = csv.reader(csvfile, delimiter=",")
    for row in lines:
        x.append(float(row[0]))
        y.append(int(row[1]))

x.sort()
# print(x)

plt.plot(x, y, color="g", linestyle="dashed", marker="o", label="Score")

plt.xticks(rotation=25)
plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("Hours spent vs Score obtained", fontsize=20)
plt.grid()
plt.legend()
plt.show()
