import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import numpy as np

# VARIANT 2
# Kramer, Gauss, Zeidel
# 9x1  + 14х2 - 15х3 + 23х4 = 5
# 16x1 + 2х2  - 22х3 + 29х4 = 8
# 18x1 + 20х2 - 3х3  + 32х4 = 9
# 10x1 + 12х2 - 16х3 + 10х4 = 4


class Matrix:
    def __init__(self, n, m):
        self.__value = []
        self.__n = n
        self.__m = m
        for i in range(n):
            self.__value.append([])
            for j in range(m):
                if i == j:
                    self.__value[i].append(1)
                else:
                    self.__value[i].append(0)

    def __str__(self):
        res = ""
        for i in range(self.__n):
            res += str(self.__value[i])
            res += '\n'
        return res

    def init_by_array(self, array):
        self.__value = array.copy()


matrixA = Matrix(4, 4)
matrixB = Matrix(4, 1)
print(matrixA)
print(matrixB)

A = [
    [9, 14, -15, 23],
    [16, 2, -22, 29],
    [18, 20, -3, 32],
    [10, 12, -16, 10],
]
B = [5, 8, 9, 4]

matrixA.init_by_array(A)
matrixB.init_by_array(B)

print(matrixA)
print(matrixB)
