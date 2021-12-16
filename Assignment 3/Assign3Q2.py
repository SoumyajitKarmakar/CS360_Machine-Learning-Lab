import pandas as pd


file = r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment3\A3_Monday.csv"
data = pd.read_csv(file)


for i in range(1, 10):
    training_data = data.sample(frac=i / 10, random_state=25)
    testing_data = data.drop(training_data.index)
    print(f"No. of training examples: {training_data.shape[0]}")
    print(f"No. of testing examples: {testing_data.shape[0]}")
