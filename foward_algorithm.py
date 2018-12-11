#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class FowardAlgorithm():
    def __init__(self):
        self.n = 3
        self.c = 3
        self.alpha = [[0 for i in range(self.c)] for j in range(self.n)]
        self.row = [1/3, 1/3, 1/3]
        self.A = np.asarray([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
        self.B = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
        #self.s = [0, 0, 0]
        self.x = [0, 1, 0]

    def b(self, w, x):
        return self.B[w, self.x[x]]
    
    def calc_Px(self):
        # STEP 1
        for i in range(self.c):
            self.alpha[0][i] = self.row[i] * self.b(i, 0)

        # STEP 2
        for n in range(1, self.n):
            for j in range(self.c):
                asum = 0
                for i in range(self.c):
                    asum += self.alpha[n-1][i] * self.A[i, j]
                self.alpha[n][j] = asum * self.b(j, n)
            # print("{}:{}".format(n, self.alpha[n]))

        # STEP 3
        Px = 0
        for i in range(self.c):
            Px += self.alpha[self.n-1][i]
        # print(Px)
        return Px

    def main(self):
        self.calc_Px()      # Fowardアルゴリズム

if __name__ == '__main__':
    fw = FowardAlgorithm()
    fw.main()
