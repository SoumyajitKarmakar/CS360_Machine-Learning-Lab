import csv


rows = []
columns = []


with open(
    r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment2\student_scores.csv", "r"
) as csvfile:

    dataset = csv.reader(csvfile)

    columns = next(dataset)

    for row in dataset:
        rows.append(row)


print("Number of features = {}".format(len(columns) - 1))
print("Number of patterns = {}".format(dataset.line_num - 1))

min = 100
max = 0

for row in rows:
    value = int(row[1])
    if value > max:
        max = value
    if value < min:
        min = value

print("Range is {}".format(max - min))
