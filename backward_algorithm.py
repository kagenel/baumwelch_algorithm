#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class BackwardAlgorithm():
    def __init__(self):
        self.n = 3
        self.c = 3
        self.beta = [[0 for i in range(self.c)] for j in range(self.n)]
        self.row = [1/3, 1/3, 1/3]
        self.A = np.asarray([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
        self.B = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
        #self.s = [1, 2, 0]
        self.x = [0, 1, 0]

    def b(self, w, x):
        return self.B[w, self.x[x]]

    def calc_Px(self):
        # STEP 1
        for i in range(self.c):
            self.beta[self.n-1][i] = 1
        # print("{}:{}".format(self.n-1, self.beta[self.n-1]))
        # STEP 2
        for t in reversed(range(self.n-1)):
            for i in range(self.c):
                _bb = 0
                for j in range(self.c):
                    _bb += self.A[i, j] * self.b(j, t+1) * self.beta[t+1][j]
                self.beta[t][i] = _bb
            # print("{}:{}".format(t, self.beta[t]))

        # STEP 3
        Px = 0
        for i in range(self.c):
            Px += self.row[i] * self.b(i, 0) * self.beta[0][i]
        # print(Px)

    def main(self):
        print(self.beta)
        self.calc_Px()

if __name__ == '__main__':
    bw = BackwardAlgorithm()
    bw.main()
