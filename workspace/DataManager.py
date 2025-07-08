import numpy as np
# import eCon_iRecorder
import eCon_iRecorder_multiprocess as eCon_iRecorder
import DataSave
from multiprocessing import Queue,Value
import time

TRIGGER_F0 = 20   #trigger触发器

class DataManager:
    def __init__(self, filename):
        self.DATA_LENGTH = 7000 # 数据长度，，原本bp设备这里是100000，eCon的采样率是bp的1/10.这里也对应改了一下
        self.type_idx = 1  # 记录实验类型
        self.nChannels = 32  # 需根据设备修改,通道数
        # self.buff_data = np.zeros((self.nChannels + 1, self.DATA_LENGTH))
        self.buff_data = np.zeros((self.nChannels, self.DATA_LENGTH))
        # 存储各通道原始数据。 +1 是原本代码里有的，因为原bp设备涉及额外的一个参考通道
        self.buff_idx = 0
        self.is_run = 1
        self.Fs = 500  # eCon采样频率为500
        self.raw_data=Queue()
        self.trigger=Value('i',TRIGGER_F0)    # 创建了一个共享内存的 Value 对象，该对象的类型为整数 ('i')，初始值为 TRIGGER_F0 变量的值
        # self.eCon = eCon_iRecorder.eCon_iRecorder()

        self.eCon = eCon_iRecorder.eCon_iRecorder(self.raw_data,self.trigger,)

        self.saver = DataSave.DataSave(data=self, name=filename)

        self.marker = np.zeros((100,), dtype=[('code', 'S100'),
                                              ('latency', 'f'),
                                              ('epoch', 'f')])
        # 这段代码创建了一个包含100个元素的 NumPy 数组，每个元素都是一个包含三个字段的结构化数组。
        # 结构化数组中的每个元素包含 'code'、'latency' (延迟)和 'epoch' (数据集被划分的一个时间段)三个字段，分别指定了不同的数据类型。
        self.marker_idx = 0


    def init_eCon(self):
        self.eCon.startAcquisition()
        time.sleep(1)

    def close_eCon(self):
        self.eCon.stopAcquisition()
        self.saver.saving()

    def saving_trail(self, trails):
        self.saver.saving_trail(trails)

    def getIdx(self, bias=0):
        return (self.buff_idx + bias + self.DATA_LENGTH) % self.DATA_LENGTH    # %取余

    def sendMarker(self, num=1):
        if self.marker_idx == self.marker.shape[0]:
            self.marker = np.hstack((self.marker, self.marker))
        if isinstance(num, str):
            self.marker[self.marker_idx]['code'] = num
        else:
            self.marker[self.marker_idx]['code'] = 'S' + str(int(num))
        # self.marker[self.marker_idx]['latency'] = (self.buff_idx + (80 / 1000) * self.Fs) / self.Fs
        self.marker[self.marker_idx]['latency'] = self.buff_idx / self.Fs    # 使用了 self.buff_idx / self.Fs 来计算 latency 的数值
        self.marker[self.marker_idx]['epoch'] = 1      # 让数据集被划分的一个时间段等于1
        self.marker_idx += 1      # +1
        # 用于记录和标记数据或事件，以便后续分析和处理

    # 将从设备读取的数据存入buffer，并将buffer中为存至save_data数组的部分更新至save_data
    def setData(self):
        # data_temp = self.eCon.getData().astype(np.float64).transpose()[0:self.nChannels + 1, :] * 0.02235174  # μV
        data_temp = self.eCon.getData()
        if data_temp.size > 0:
            # print(data_temp.shape)
            data_temp = data_temp.astype(np.float64).transpose()[0:self.nChannels, :] * 0.02235174  # μV
            for k in range(data_temp.shape[1]):
                idx_temp = self.getIdx()
                self.buff_data[:, idx_temp] = data_temp[:, k]
                self.buff_idx += 1
        self.saver.update()
