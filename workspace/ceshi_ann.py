import scipy.signal as signal
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
ROOT = os.getcwd()
from train import run
import numpy as np
from scipy.io import loadmat
import time

def tiqushuju2(trails):
    data9 = loadmat(
                    r"C:\Users\lenovo\Desktop\二便范式_电刺激\workspace\data\Cent1 Sub002 Exp1 Run02 MI_zaixian" + str(trails)+".mat")
    data9 = data9['data']
    b = data9.shape[-1]
    data9 = data9.reshape(32, b)
    class1 = np.zeros((1, 32, 1500))
    data4 = data9[:, -1500:]
    class1[0] = data4
    return class1

def tiqushuju2_null():
    class1 = np.zeros((1, 32, 1500))
    return class1

# 带通
def preprocess_filt(data, low_cut=0.1, high_cut=40, fs=500, order=4):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    padlen = len(data) // 3
    proced = signal.filtfilt(b, a, data,padlen=padlen)
    return proced

class PreProcessSequential:
    def __init__(self):
        self.fs = 500
        self.refs = 250
        self.bplow = 0.1
        self.bphigh = 50
        self.bslow = 49
        self.bshigh = 51
        self.notch = 50

    def __call__(self, data: np.ndarray):
        return self._sequential(data)

    def _sequential(self, x):
        x = preprocess_filt(x, low_cut=self.bplow, high_cut=self.bphigh, fs=self.fs)
        return x

def standardize_data_new(X_test, channels):
    scaler = StandardScaler()
    for j in range(channels):
        # Reshape the data to 2D for the StandardScaler and then transform
        shape_2d = X_test[:, :, j, :].shape  # Store original shape excluding the channel dimension
        X_temp = X_test[:, :, j, :].reshape(-1, 1)  # Reshape for scaling
        X_test[:, :, j, :] = scaler.fit_transform(X_temp).reshape(shape_2d)  # Scale and reshape back
    return X_test

def get_data(trails):
    X_test = tiqushuju2(trails)
    preprocessor = PreProcessSequential()
    X_test = preprocessor(X_test)
    N_test, N_ch, _ = X_test.shape
    X_test = X_test[:, :, :].reshape(N_test, 1, N_ch, _)
    # X_test = standardize_data_new(X_test, N_ch)
    return X_test

def get_data_null():
    X_test = tiqushuju2_null()
    preprocessor = PreProcessSequential()
    X_test = preprocessor(X_test)
    N_test, N_ch, _ = X_test.shape
    X_test = X_test[:, :, :].reshape(N_test, 1, N_ch, _)
    return X_test

def zaixian_suanfa(trails):
    # 记录开始时间
    start_time = time.time()
    X_test = get_data(trails)
    result = run(X_test)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"程序运行时间为 {runtime} 秒")
    return result

def zaixian_suanfa_null():
    # 记录开始时间
    start_time = time.time()
    X_test = get_data_null()
    result = run(X_test)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"程序运行时间为 {runtime} 秒")
    return result


