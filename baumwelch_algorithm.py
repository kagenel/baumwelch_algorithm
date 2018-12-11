#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy
from foward_algorithm import FowardAlgorithm
from backward_algorithm import BackwardAlgorithm

class BaumwelchAlgorithm():
    def __init__(self):
        self.n = 3
        self.c = 3
        self.m = 2
        self.row = [1.0, 0.0, 0.0]
        self.A = np.asarray([[0.15, 0.60, 0.25], [0.25, 0.15, 0.60], [0.60, 0.25, 0.15]])
        self.B = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        self.Rrow = [0.0, 0.0, 0.0]
        self.Aa = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.Bb = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        self.s = [1, 2, 0]
        self.x = [0, 1, 0]

        self.fw = FowardAlgorithm()
        self.bw = BackwardAlgorithm()

        self.alpha = [[0 for i in range(self.c)] for j in range(self.n)]

    def b(self, w, x):
        return self.B[w, self.x[x]]

    def delta(self, x, v):
        if self.x[x] == v:
            return 1.0
        else:
            return 0

    # (Foward)
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

        # STEP 3
        Px = 0
        for i in range(self.c):
            Px += self.alpha[self.n-1][i]
        print(Px)
        return Px


    def estimate_HMM(self):
        # STEP 1
        self.A = np.asarray([[0.15, 0.60, 0.25], [0.25, 0.15, 0.60], [0.60, 0.25, 0.15]])
        self.B = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        self.row = [1.0, 0.0, 0.0]

        for n in range(100):
            self.fw.calc_Px()
            self.bw.calc_Px()
            _alpha = self.fw.alpha
            _beta = self.bw.beta

            # STEP 2
            for i in range(self.c):
                for j in range(self.c):
                    _xi = 0.0
                    _gamma = 0.0
                    for t in range(self.n-1):
                        _xi += _alpha[t][i] * self.A[i, j] * self.b(j, t+1) * _beta[t+1][j]
                        _gamma += _alpha[t][i] * _beta[t][i]
                    self.Aa[i, j] = _xi / _gamma

            for j in range(self.c):
                for k in range(self.m):
                    _del = 0.0
                    _gamma = 0.0
                    for t in range(self.n):
                        _del += self.delta(t, k) * _alpha[t][j] * _beta[t][j]
                        _gamma += _alpha[t][j] * _beta[t][j]
                    self.Bb[j, k] = _del / _gamma

            for i in range(self.c):
                _al = 0.0
                for j in range(self.c):
                    _al += _alpha[self.n-1][j]
                self.Rrow[i] = (_alpha[0][i] * _beta[0][i]) / _al

            # STEP 3
            self.A = copy.copy(self.Aa)
            self.B = copy.copy(self.Bb)   
            self.row = copy.copy(self.Rrow)   

            print(self.B)
            
            # _Px = np.log(self.calc_Px())
            # if np.abs(_Px) < 0.001:
            #     print(_Px)
            #     break

    def main(self):
        self.estimate_HMM()

if __name__ == '__main__':
    bm = BaumwelchAlgorithm()
    bm.main()
