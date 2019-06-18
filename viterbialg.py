#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class ViterbiAlgorithm():
    def __init__(self, A, B, r, x_list):
        self.A = A
        self.B = B
        self.rho = r
        self.x_list = x_list
        self.c = B.shape[0]  # 状態数
        self.n = len(x_list)  # 観測回数
        self.psi = [[0 for i in range(self.c)] for j in range(self.n)]
        self.Psi = [[0 for i in range(self.c)] for j in range(self.n)]
        self.Px = 0
        self.i = [0 for i in range(self.n)]
        self.s_star = [0 for i in range(self.n)]

    def b(self, w, x):
        return self.B[w, self.x_list[x]]

    def max(self, _list):
        p = 0.0
        for i in _list:
            if p < i:
                p = i
        return p

    def argmax(self, _list):
        _index = -1
        p = 0.0
        for i, j in enumerate(_list):
            if p < j:
                p = j
                _index = i
        return _index

    def calc_Px(self):
        # STEP 1
        for i in range(self.c):
            self.psi[0][i] = self.rho[i] * self.b(i, 0)
            self.Psi[0][i] = 0.0
        
        # STEP 2
        for t in range(1, self.n):
            _plist = []
            for j in range(self.c):
                for i in range(self.c):
                    _plist.append(self.psi[t-1][i] * self.A[i][j])
                self.psi[t][j] = self.max(_plist) * self.b(j, t)
                self.Psi[t][j] = self.argmax(_plist)
        
        # STEP 3
        self.Px = self.max(self.psi[self.n-1])
        self.i[self.n-1] = self.argmax(self.psi[self.n-1])
        self.s_star = self.i[self.n-1]   # w は番号と一致させてる

        # STEP 4
        for t in reversed(range(0, self.n-2)):
            self.i[t] = self.Psi[t+1][self.i[t+1]]
            self.s_star = self.i[t]  # w は番号と一致させてる

if __name__ == '__main__':
    A = np.asarray([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
    B = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    rho = [1 / 3, 1 / 3, 1 / 3]
    x = [0, 1, 0]
    #x = [np.random.randint(0,2) for i in range(10000)]
    vi = ViterbiAlgorithm(A, B, rho, x)
    vi.calc_Px()
    print(vi.Px)