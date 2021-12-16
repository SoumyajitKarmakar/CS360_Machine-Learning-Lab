import numpy as np
import pandas as pd
import matplotlib.image as plot

img = plot.imread(r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\img.jpg")

if img.shape[2] == 3:
    img = img.reshape(img.shape[0], -1)


dataFrame = pd.DataFrame(img)


dataFrame.to_csv(r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\imageToCSV.csv")
