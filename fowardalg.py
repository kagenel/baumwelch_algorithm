#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class FowardAlgorithm():
    def __init__(self):
        self.A = np.zeros(1)
        self.B = np.zeros(1)
        self.row = []
        self.x_list = []
        self.c = 0  # 状態数
        self.n = len(self.x_list)  # 観測回数
        self.alpha = [[0 for i in range(self.c)] for j in range(self.n)]
        self.Px = 0.0

    def set(self, A, B, r, x_list):
        self.A = A
        self.B = B
        self.row = r
        self.x_list = x_list
        self.c = B.shape[0]  # 状態数
        self.n = len(x_list)  # 観測回数
        self.alpha = [[0 for i in range(self.c)] for j in range(self.n)]

    def b(self, w, x):
        return self.B[w, self.x_list[x]]

    def calc_Px(self):
        # STEP 1
        for i in range(self.c):
            self.alpha[0][i] = self.row[i] * self.b(i, 0)

        # STEP 2
        for n in range(1, self.n):
            for j in range(self.c):
                asum = 0
                for i in range(self.c):
                    asum += self.alpha[n - 1][i] * self.A[i, j]
                self.alpha[n][j] = asum * self.b(j, n)
            # print("{}:{}".format(n, self.alpha[n]))

        # STEP 3
        Px = 0
        for i in range(self.c):
            Px += self.alpha[self.n - 1][i]
        self.Px = Px


if __name__ == '__main__':
    A = np.asarray([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
    B = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    row = [1 / 3, 1 / 3, 1 / 3]
    x = [0, 1, 0]
    fw = FowardAlgorithm()
    fw.set(A, B, row, x)
    fw.calc_Px()
    print(fw.Px)
