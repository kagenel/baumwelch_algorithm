#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class OutputSymbol():

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.x = []
        self.s = []

    def generate_symbol(self, n):
        self.s.append(self.lottery(0, np.vstack((self.rho, self.rho))))
        for i in range(n):
            self.x += self.lottery(self.s[i], self.B)
            self.s += self.lottery(self.s[i], self.A)

        return self.x, self.s

    def lottery(self, st, array):
        t = 0.0
        r = np.random.rand()
        for i in range(array.shape[0]):
            if r < array[st, i] + t:
                return [i]
            t += array[st, i]
        return [-1]