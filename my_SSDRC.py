#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import librosa
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft

parser = argparse.ArgumentParser(
    prog='SSDRC',
    usage='SSDRC.py -i (filename)',
    epilog='end',
    add_help=True,
)

parser.add_argument('-i', '--infile',
                    help='input wav file (str)',
                    type=str,
                    default='people_people-gaya-classroom1.wav')


def voicing_index(x):
    """
    時変フィルタを制御するパラメータ
    argument
        x: 音声信号
    return
        V: voicing_index(パラメータ)
    """

    # 矩形窓をかける
    win = signal.boxcar(len(x))
    x = x * win
    # 正規化
    x_max = max(x)
    x = x / x_max

    # ゼロ交差率
    p = librosa.feature.zero_crossing_rate(x)
    # RMS
    k = librosa.feature.rmse(x)
    # voicing_indexを導出
    V = x_max * p / k

    return V



if __name__ == "__main__":

    args = parser.parse_args()
    np.set_printoptions(threshold=np.inf)

    # 音声をロード
    x, fs = librosa.load(args.infile)
    # print(x.shape)

    V = voicing_index(x)
    # print(V.shape)
    # 正規化
    x = x / max(x)
    # 離散フーリエ変換を行う
    S = np.abs(librosa.stft(x))
    # print(S.shape)

    # 対数をとる
    log_S = np.log10(S)

    cps = np.zeros((S.shape[0], S.shape[1]))
    spec = np.zeros((S.shape[0], S.shape[1]))

    # ケプストラムを求め、スペクトル包絡を計算する
    for i in range(0, S.shape[1]):
        cps[:, i] = np.real(np.fft.ifft(log_S[:, i]))
        cps_order = 32
        cps[cps_order:len(cps[:, i])-cps_order+1, i] = 0
        spec[:, i] = np.fft.fft(cps[:, i])
    # print(cps.shape)
    # print(spec.shape)

    # スペクトル傾斜を求める
    log_T = np.zeros(spec.shape[1])
    T = np.zeros(spec.shape[1])
    for i in range(0, len(T)):
        log_T[i] =cps[0, i] + 2 * cps[1, i]
        T[i] = np.exp(log_T[i])
    # print(T.shape)

    # H1フィルタの計算
    H1 = np.zeros((S.shape[0], S.shape[1]))
    b = 1.0
    for i in range(0, H1.shape[1]):
        A = spec[:, i] / T[i]
        B = b * V[0, i]
        H1[:, i] = np.power(A, B)
    # print(H1)
    # print(H1.shape)
