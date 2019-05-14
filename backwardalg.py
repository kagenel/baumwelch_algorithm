#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from fowardalg import FowardAlgorithm

class BackwardAlgorithm():
    def __init__(self, A, B, r, x_list):
        self.A = A
        self.B = B
        self.rho = r
        self.x_list = x_list
        self.c = B.shape[0]  # 状態数
        self.n = len(x_list)  # 観測回数
        self.beta = [[0 for i in range(self.c)] for j in range(self.n)]
        self.Px = 0.0
        self.scale = []

    def b(self, w, x):
        return self.B[w, self.x_list[x]]

    def calc_Px(self):
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
            Px += self.rho[i] * self.b(i, 0) * self.beta[0][i]
        self.Px = Px

    def calc_Px_scale(self, scale):
        # STEP 1
        for i in range(self.c):
            self.beta[self.n - 1][i] = 1
        # scale
        self.scale = scale
        for i in range(self.c):
            self.beta[self.n - 1][i] /= self.scale[-1]

        # STEP 2
        for t in reversed(range(self.n - 1)):
            for i in range(self.c):
                _bb = 0
                for j in range(self.c):
                    _bb += self.A[i, j] * self.b(j,
                                                 t + 1) * self.beta[t + 1][j]
                self.beta[t][i] = _bb / self.scale[t+1]

        # STEP 3
        Px = 1
        for i in range(self.c):
            Px *= self.scale[i]
        self.Px = Px

if __name__ == '__main__':
    A = np.asarray([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
    B = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    rho = [1 / 3, 1 / 3, 1 / 3]
    x = [0, 1, 0]
    bw = BackwardAlgorithm(A, B, rho, x)
    fw = FowardAlgorithm(A, B, rho, x)
    fw.calc_Px_scale()
    bw.calc_Px_scale(fw.scale)
    #print(bw.beta)
    print(bw.Px)
