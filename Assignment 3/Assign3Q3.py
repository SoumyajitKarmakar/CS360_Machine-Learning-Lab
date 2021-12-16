import pandas as pd


file = r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment3\A3_Monday.csv"
data = pd.read_csv(file)

dataNoLast = data.copy(deep=True)
dataNoLast.drop(columns=dataNoLast.columns[-1], axis=1, inplace=True)

print("Row-wise values :- ")

print("Means of each row is : ")
means = dataNoLast.mean(axis=1)
for i in range(100):
    print(means[i])

print("Medians of each row is : ")
medians = dataNoLast.median(axis=1)
for i in range(100):
    print(medians[i])

print("Modes of each row is : ")
modes = dataNoLast.mode(axis=1)[0]
for i in range(100):
    print(modes[i])

print("Standard Deviation of each row is : ")
sds = dataNoLast.std(axis=1)
for i in range(100):
    print(sds[i])


print("\nColumn-wise values :- ")

print("bedrooms column : ")
mean = data["bedrooms"].mean()
median = data["bedrooms"].median()
mode = data["bedrooms"].mode()[0]
sds = data["bedrooms"].std()
print("Mean is {}".format(mean))
print("Median is {}".format(median))
print("Mode is {}".format(mode))
print("Standard Deviation is {}".format(sds))

print("grade column : ")
mean = data["grade"].mean()
median = data["grade"].median()
mode = data["grade"].mode()[0]
sds = data["grade"].std()
print("Mean is {}".format(mean))
print("Median is {}".format(median))
print("Mode is {}".format(mode))
print("Standard Deviation is {}".format(sds))

print("sqft_living column : ")
mean = data["sqft_living"].mean()
median = data["sqft_living"].median()
mode = data["sqft_living"].mode()[0]
sds = data["sqft_living"].std()
print("Mean is {}".format(mean))
print("Median is {}".format(median))
print("Mode is {}".format(mode))
print("Standard Deviation is {}".format(sds))

print("bathrooms column : ")
mean = data["bathrooms"].mean()
median = data["bathrooms"].median()
mode = data["bathrooms"].mode()[0]
sds = data["bathrooms"].std()
print("Mean is {}".format(mean))
print("Median is {}".format(median))
print("Mode is {}".format(mode))
print("Standard Deviation is {}".format(sds))

print("floors column : ")
mean = data["floors"].mean()
median = data["floors"].median()
mode = data["floors"].mode()[0]
sds = data["floors"].std()
print("Mean is {}".format(mean))
print("Median is {}".format(median))
print("Mode is {}".format(mode))
print("Standard Deviation is {}".format(sds))

print("yr_built column : ")
mean = data["yr_built"].mean()
median = data["yr_built"].median()
mode = data["yr_built"].mode()[0]
sds = data["yr_built"].std()
print("Mean is {}".format(mean))
print("Median is {}".format(median))
print("Mode is {}".format(mode))
print("Standard Deviation is {}".format(sds))

print("price column : ")
mean = data["price"].mean()
median = data["price"].median()
mode = data["price"].mode()[0]
sds = data["price"].std()
print("Mean is {}".format(mean))
print("Median is {}".format(median))
print("Mode is {}".format(mode))
print("Standard Deviation is {}".format(sds))

print("\nOverall values :- ")
print("Mean is {}".format(dataNoLast.stack().mean()))
print("Median is {}".format(dataNoLast.stack().median()))
print("Mode is {}".format(dataNoLast.stack().mode()[0]))
print("Standard Deviation is {}".format(dataNoLast.stack().std()))
