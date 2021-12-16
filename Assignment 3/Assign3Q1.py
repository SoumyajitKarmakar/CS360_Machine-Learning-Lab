import csv


rows = []
columns = []


with open(
    r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment3\A3_Monday.csv", "r"
) as csvfile:

    dataset = csv.reader(csvfile)

    columns = next(dataset)

    for row in dataset:
        rows.append(row)


print("Number of features = {}".format(len(columns) - 2))
print("Number of patterns = {}".format(dataset.line_num - 1))

min = 100
max = 0

for row in rows:
    # print(int(float(row[7])))
    value = int(float(row[7]))
    if value > max:
        max = value
    if value < min:
        min = value

print("Range is {}".format(max - min))
