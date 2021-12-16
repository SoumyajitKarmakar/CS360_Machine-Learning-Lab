import numpy as num


mat1 = num.full((2, 5), 0)
mat2 = num.full((2, 5), 1)
mat3 = num.full((2, 5), 5)

num.savetxt(
    r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\matrix1.txt", mat1, fmt="%d"
)
num.savetxt(
    r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\matrix2.txt", mat2, fmt="%d"
)
num.savetxt(
    r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\matrix3.txt", mat3, fmt="%d"
)


load1 = num.loadtxt(r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\matrix1.txt")
load2 = num.loadtxt(r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\matrix2.txt")
load3 = num.loadtxt(r"C:\Users\karma\Desktop\5th_Sem\CS360\Assignment1\matrix3.txt")

print(load1)
print(load2)
print(load3)
