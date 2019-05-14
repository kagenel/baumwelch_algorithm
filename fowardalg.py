#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class FowardAlgorithm():
    def __init__(self, A, B, r, x_list):
        self.A = A
        self.B = B
        self.rho = r
        self.x_list = x_list
        self.c = B.shape[0]  # 状態数
        self.n = len(x_list)  # 観測回数
        self.alpha = [[0 for i in range(self.c)] for j in range(self.n)]
        self.Px = 0.0
        self.scale = []

    def b(self, w, x):
        return self.B[w, self.x_list[x]]

    def calc_Px(self):
        # STEP 1
        for i in range(self.c):
            self.alpha[0][i] = self.rho[i] * self.b(i, 0)

        # STEP 2
        for n in range(1, self.n):
            for j in range(self.c):
                asum = 0.0
                for i in range(self.c):
                    asum += self.alpha[n - 1][i] * self.A[i, j]
                self.alpha[n][j] = asum * self.b(j, n)
            # print("{}:{}".format(n, self.alpha[n]))

        # STEP 3
        Px = 0
        for i in range(self.c):
            Px += self.alpha[self.n - 1][i]
        self.Px = Px

    def calc_Px_scale(self):
        # STEP 1
        for i in range(self.c):
            self.alpha[0][i] = self.rho[i] * self.b(i, 0)
        # scale
        self.scale.clear()
        asum = 0.0
        for i in range(self.c):
            asum += self.alpha[0][i]
        self.scale += [asum]
        for i in range(self.c):

            self.alpha[0][i] /= asum

        # STEP 2
        for n in range(1, self.n):
            for j in range(self.c):
                asum = 0.0
                for i in range(self.c):
                    asum += self.alpha[n - 1][i] * self.A[i, j]
                self.alpha[n][j] = asum * self.b(j, n)
            
            # scale
            asum = 0.0
            for j in range(self.c):
                asum += self.alpha[n][j]
            self.scale += [asum]
            for j in range(self.c):
                self.alpha[n][j] /= asum

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
    #x = [np.random.randint(0,2) for i in range(10000)]
    fw = FowardAlgorithm(A, B, rho, x)
    fw.calc_Px_scale()
    print(fw.Px)
