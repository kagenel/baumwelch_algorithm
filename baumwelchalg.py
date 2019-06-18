#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 続・わかりやすいパターン認識より，
# p.162 ~ 165 までのバウム・ウェルチアルゴリズムの勉強用
# プログラム

import numpy as np
import copy
from fowardalg import FowardAlgorithm
from viterbialg import ViterbiAlgorithm
from backwardalg import BackwardAlgorithm
from outputSymbol import OutputSymbol

class BaumwelchAlgorithm():
    def __init__(self, A, B, rho):
        self.n = 0
        self.c = A.shape[1]
        self.m = B.shape[1]
        self.A = A
        self.B = B
        self.rho = rho
        self.Rrho = [0.0, 0.0, 0.0]
        self.Aa = np.asarray([[0.0, 0.0, 0.0], 
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]])
        self.Bb = np.asarray([[0.0, 0.0], 
                              [0.0, 0.0], 
                              [0.0, 0.0]])
        self.x = []
        self.s = []

    def set_output_symbole(self, A, B, rho, n):
        # 学習データ作成
        dice = OutputSymbol(A, B, rho)
        self.x, self.s = dice.generate_symbol(n)

    def b(self, w, x):
        return self.B[w, self.x[x]]

    def delta(self, t, v):
        if self.x[t] == v:
            return 1
        else:
            return 0

    def estimate_HMM(self, max_epoch):
        # STEP 1
        self.n = len(self.x)
        old_Px = 0.0
        for n in range(max_epoch):
            
            fw = FowardAlgorithm(self.A, self.B, self.rho, self.x)
            bw = BackwardAlgorithm(self.A, self.B, self.rho, self.x)
            fw.calc_Px()
            bw.calc_Px()
            _alpha = fw.alpha
            _beta = bw.beta

            # STEP 2
            for i in range(self.c):
                for j in range(self.c):
                    _xi = 0.0
                    _gamma = 0.0
                    for t in range(self.n - 1):
                        _xi += _alpha[t][i] * self.A[i, j] * self.b(j, t + 1) * _beta[t + 1][j]
                        _gamma += _alpha[t][i] * _beta[t][i]

                    self.Aa[i, j] = _xi / _gamma

            for j in range(self.c):
                for k in range(self.m):
                    _del = 0.0
                    _gamma = 0.0
                    for t in range(self.n):
                        #print("j={},k={},t={}:{}".format(j, k, t, self.delta(t, k) * _alpha[t][j] * _beta[t][j]))
                        _del += self.delta(t, k) * _alpha[t][j] * _beta[t][j]
                        _gamma += _alpha[t][j] * _beta[t][j]
                    self.Bb[j, k] = _del / _gamma

            for i in range(self.c):
                _Px = 0.0
                for j in range(self.c):
                    _Px += _alpha[self.n - 1][j]
                self.Rrho[i] = (_alpha[0][i] * _beta[0][i]) / _Px

            # STEP 3
            print(self.Aa)
            #print(self.Bb[0][0])
            #print(self.Rrho)
            self.A = copy.copy(self.Aa)
            self.B = copy.copy(self.Bb)
            self.rho = copy.copy(self.Rrho)

            print(np.log(fw.Px) - old_Px)
            if np.abs(np.log(fw.Px) - old_Px) < 0.001:
                print(np.log(fw.Px))
                print(self.Aa)
                break
            old_Px = np.log(fw.Px)

    def estimate_HMM_scale(self, max_epoch):
        # STEP 1
        self.n = len(self.x)
        old_Px = 0.0
        for n in range(max_epoch):
            
            fw = FowardAlgorithm(self.A, self.B, self.rho, self.x)
            bw = BackwardAlgorithm(self.A, self.B, self.rho, self.x)
            fw.calc_Px_scale()
            bw.calc_Px_scale(fw.scale)
            _alpha = fw.alpha
            _beta = bw.beta

            # STEP 2
            for i in range(self.c):
                for j in range(self.c):
                    _xi = 0.0
                    _gamma = 0.0
                    for t in range(self.n - 1):
                        _xi += _alpha[t][i] * self.A[i, j] * self.b(j, t + 1) * _beta[t + 1][j] / fw.scale[t+1]
                        _gamma += _alpha[t][i] * _beta[t][i]

                    self.Aa[i, j] = _xi / _gamma

            for j in range(self.c):
                for k in range(self.m):
                    _del = 0.0
                    _gamma = 0.0
                    for t in range(self.n):
                        #print("j={},k={},t={}:{}".format(j, k, t, self.delta(t, k) * _alpha[t][j] * _beta[t][j]))
                        _del += self.delta(t, k) * _alpha[t][j] * _beta[t][j]
                        _gamma += _alpha[t][j] * _beta[t][j]
                    self.Bb[j, k] = _del / _gamma

            for i in range(self.c):
                self.Rrho[i] = (_alpha[0][i] * _beta[0][i])

            # STEP 3
            print("---------")
            print("count: {}".format(n))
            print(self.Aa)
            print(self.Bb)
            #print(self.Rrho)
            self.A = copy.copy(self.Aa)
            self.B = copy.copy(self.Bb)
            self.rho = copy.copy(self.Rrho)

            Px = 0
            for i in range(len(fw.scale)):
                Px += np.log(fw.scale[i])
            self.Px = Px

            print(Px)
            if np.abs(Px - old_Px) < 0.001:
                pass
                #break
            old_Px = Px


if __name__ == '__main__':
    # [3] バウム・ウェルチアルゴリズムの実験
    print("[3] バウム・ウェルチアルゴリズムの実験")
    # 初期値
    A = np.asarray([[0.15, 0.60, 0.25], 
                             [0.25, 0.15, 0.60],
                             [0.60, 0.25, 0.15]])
    B = np.asarray([[0.5, 0.5], 
                             [0.5, 0.5],
                             [0.5, 0.5]])
    rho = [1.0, 0.0, 0.0]
    bm = BaumwelchAlgorithm(A, B, rho)
    # ダイス1(クラス1)の測定
    dice_A = np.array([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
    dice_B = np.array([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    dice_rho = np.array([1.0, 0.0, 0.0])
    bm.set_output_symbole(dice_A, dice_B, dice_rho, 1000)
    bm.estimate_HMM_scale(100)

    # [4] 識別実験
    print("[4] 識別実験")
    A2 = np.asarray([[0.6, 0.25, 0.15], 
                             [0.15, 0.60, 0.25],
                             [0.25, 0.15, 0.60]])
    B2 = np.asarray([[0.5, 0.5], 
                             [0.5, 0.5],
                             [0.5, 0.5]])
    rho2 = [1.0, 0.0, 0.0]
    # ダイス2(クラス2)の測定
    dice2_A = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
    dice2_B = np.array([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    dice2_rho = np.array([1.0, 0.0, 0.0])
    bm2 = BaumwelchAlgorithm(A2, B2, rho2)
    bm2.set_output_symbole(dice2_A, dice2_B, dice2_rho, 1000)
    bm2.estimate_HMM_scale(100)

    # 識別テスト (Foward)
    print("識別テスト (Foward)")
    # W1 なら当たり
    count_fw = 0 # 誤判定数
    for i in range(100):
        dice = OutputSymbol(dice_A, dice_B, dice_rho)
        test_x, test_s = dice.generate_symbol(100)

        fw1 = FowardAlgorithm(bm.A, bm.B, bm.rho, test_x)
        fw1.calc_Px()
        print("W1: {}".format(np.log(fw1.Px)))

        fw2 = FowardAlgorithm(bm2.A, bm2.B, bm2.rho, test_x)
        fw2.calc_Px()
        print("W2: {}".format(np.log(fw2.Px)))
        
        if np.log(fw1.Px) > np.log(fw2.Px):
            print("Judgment: W1")
            
        if np.log(fw1.Px) < np.log(fw2.Px):
            print("Judgment: W2")
            count_fw += 1

    # W2
    for i in range(100):
        dice = OutputSymbol(dice2_A, dice2_B, dice2_rho)
        test_x, test_s = dice.generate_symbol(100)

        fw1 = FowardAlgorithm(bm.A, bm.B, bm.rho, test_x)
        fw1.calc_Px()
        print("W1: {}".format(np.log(fw1.Px)))

        fw2 = FowardAlgorithm(bm2.A, bm2.B, bm2.rho, test_x)
        fw2.calc_Px()
        print("W2: {}".format(np.log(fw2.Px)))
        
        if np.log(fw1.Px) > np.log(fw2.Px):
            print("Judgment: W1")
            count_fw += 1
            
        if np.log(fw1.Px) < np.log(fw2.Px):
            print("Judgment: W2")

    # 識別テスト (Viterbi)
    print("識別テスト (Viterbi)")
    # W1 なら当たり
    count_vi = 0 # 誤判定数
    for i in range(100):
        dice = OutputSymbol(dice_A, dice_B, dice_rho)
        test_x, test_s = dice.generate_symbol(100)

        vi1 = FowardAlgorithm(bm.A, bm.B, bm.rho, test_x)
        vi1.calc_Px()
        print("W1: {}".format(np.log(vi1.Px)))

        vi2 = FowardAlgorithm(bm2.A, bm2.B, bm2.rho, test_x)
        vi2.calc_Px()
        print("W2: {}".format(np.log(vi2.Px)))
        
        if np.log(vi1.Px) > np.log(vi2.Px):
            print("Judgment: W1")
            
        if np.log(vi1.Px) < np.log(vi2.Px):
            print("Judgment: W2")
            count_vi += 1

    # W2
    for i in range(100):
        dice = OutputSymbol(dice2_A, dice2_B, dice2_rho)
        test_x, test_s = dice.generate_symbol(100)

        vi1 = FowardAlgorithm(bm.A, bm.B, bm.rho, test_x)
        vi1.calc_Px()
        print("W1: {}".format(np.log(vi1.Px)))

        vi2 = FowardAlgorithm(bm2.A, bm2.B, bm2.rho, test_x)
        vi2.calc_Px()
        print("W2: {}".format(np.log(vi2.Px)))
        
        if np.log(vi1.Px) > np.log(vi2.Px):
            print("Judgment: W1")
            count_vi += 1
            
        if np.log(vi1.Px) < np.log(vi2.Px):
            print("Judgment: W2")

    # 識別率
    print("================")
    print("Foward")
    print("識別率: {}%".format((200 - count_fw)/200 * 100))
    print("Viterbi")
    print("識別率: {}%".format((200 - count_vi)/200 * 100))
