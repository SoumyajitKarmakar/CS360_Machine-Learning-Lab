import pandas as pd
import matplotlib.image as plot


# loadDataFrame = pd.read_csv(
#     r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\imageToCSV.csv",
#     skipfooter=1,
#     engine="python",
# )


loadDataFrame = pd.read_csv(
    r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\imageToCSV.csv"
)

loadDataFrame = loadDataFrame.iloc[:, :-1]

matrix2D = loadDataFrame.values

# print(matrix2D.shape)


loaded_mat = matrix2D.reshape(
    matrix2D.shape[0],
    matrix2D.shape[1] // 3,
    3,
)
