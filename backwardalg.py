#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class BackwardAlgorithm():
    def __init__(self):
        self.A = np.zeros(1)
        self.B = np.zeros(1)
        self.row = []
        self.x_list = []
        self.c = 0  # 状態数
        self.n = len(self.x_list)  # 観測回数
        self.beta = [[0 for i in range(self.c)] for j in range(self.n)]
        self.Px = 0.0

    def __init__(self, A, B, r, x_list):
        self.A = A
        self.B = B
        self.row = r
        self.x_list = x_list
        self.c = B.shape[0]  # 状態数
        self.n = len(x_list)  # 観測回数
        self.beta = [[0 for i in range(self.c)] for j in range(self.n)]
        self.Px = 0.0

    def b(self, w, x):
        return self.B[w, self.x_list[x]]

    def calc_Px(self):
        self.A = np.asarray([[0.15, 0.60, 0.25], [0.25, 0.15, 0.60],
                             [0.60, 0.25, 0.15]])
        self.B = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        self.rho = [1.0, 0.0, 0.0]
        # STEP 1
        for i in range(self.c):
            self.beta[self.n - 1][i] = 1
        # print("{}:{}".format(self.n-1, self.beta[self.n-1]))
        # STEP 2
        for t in reversed(range(self.n - 1)):
            for i in range(self.c):
                _bb = 0
                for j in range(self.c):
                    _bb += self.A[i, j] * self.b(j,
                                                 t + 1) * self.beta[t + 1][j]
                self.beta[t][i] = _bb
            # print("{}:{}".format(t, self.beta[t]))

        # STEP 3
        Px = 0
        for i in range(self.c):
            Px += self.row[i] * self.b(i, 0) * self.beta[0][i]
        self.Px = Px


if __name__ == '__main__':
    A = np.asarray([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
    B = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    row = [1 / 3, 1 / 3, 1 / 3]
    x = [0, 1, 0]
    bw = BackwardAlgorithm()
    bw.set(A, B, row, x)
    bw.calc_Px()
    print(bw.beta)
    print(bw.Px)
