import numpy as np
from scipy.io import savemat


class DataSave:

    def __init__(self, data, name='test'):
        self.data = data
        self.nChannels = 32
        # self.save_data = np.zeros([self.nChannels + 1, self.data.DATA_LENGTH], dtype=np.float32)
        self.save_data = np.zeros([self.nChannels, self.data.DATA_LENGTH], dtype=np.float32)     #dtype为数据类型
        self.data_idx = 0
        self.save_idx = 0
        self.name = name

    def getIdx(self, bias=0):
        return (self.data_idx + bias + self.data.DATA_LENGTH) % self.data.DATA_LENGTH
    #计算数据索引

    # 用于将buffer中未存储的数据存储至save_data这个数组
    def update(self):
        while self.data_idx < self.data.buff_idx:
            # if self.data_idx % 5 == 0:
            # 原来这样写推测应该是因为bp采样率为5000，降采样为1000（与后面解码fs = 1000一致）
            # eCon采样率为500，这里就不降采样了，后面解码fs也定为500
            if self.save_idx == self.save_data.shape[1]:
                self.save_data = np.hstack((self.save_data, self.save_data))

            self.save_data[:, self.save_idx] = self.data.buff_data[:, self.getIdx()]
            # self.save_data[self.nChannels, self.save_idx] = 0 原bp设备用于主动将参考通道置零的
            self.save_idx += 1  # 用于索引存储数组中的位置
            self.data_idx += 1  # 用于索引buffer中已存储的数据的位置，一旦落后于buffer中已填充的位置(buff_idx)，则将未存储的部分更新
    # 将缓冲区中的数据更新到save_data中，并且在必要时扩展save_data的大小。
    # 同时，通过更新save_idx和data_idx来跟踪已存储的数据位置和缓冲区中的数据位置。

    def saving(self):
        self.update()
        dt = [('labels', 'S100'),
              ('topo_enabled', 'f'),
              ('theta', 'O'),
              ('radius', 'O'),
              ('sph_theta', 'O'),
              ('sph_phi', 'O'),
              ('sph_theta_besa', 'O'),
              ('sph_phi_besa', 'O'),
              ('X', 'O'),
              ('Y', 'O'),
              ('Z', 'O')]
        ch = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'Cz', 'FC2', 'FC6',
              'F8', 'F4', 'Fz', 'AF4', 'Fp2', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
              'C4', 'T8', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'PO3', 'O1', 'Oz']
        # chanlocs = np.zeros((self.nChannels + 1,), dtype=dt)
        chanlocs = np.zeros((self.nChannels,), dtype=dt)   # 创建了一个包含 nChannels 个通道位置信息的数组 chanlocs，数组中每个元素的数据类型由变量 dt 指定
        # for k in range(self.nChannels + 1):
        for k in range(self.nChannels):     # k 的取值范围是从 0 到 self.nChannels-1
            chanlocs[k]['labels'] = ch[k]
            chanlocs[k]['topo_enabled'] = 0
            chanlocs[k]['theta'] = []
            chanlocs[k]['radius'] = []
            chanlocs[k]['sph_theta'] = []
            chanlocs[k]['sph_phi'] = []
            chanlocs[k]['sph_theta_besa'] = []
            chanlocs[k]['sph_phi_besa'] = []
            chanlocs[k]['X'] = []
            chanlocs[k]['Y'] = []
            chanlocs[k]['Z'] = []

        history = []

        # header = {'filetype': 'time_amplitude',
        #           'name': self.name,
        #           'datasize': [1, self.nChannels + 1, 1, 1, 1, self.save_idx],
        #           'xstart': 0,
        #           'ystart': 0,
        #           'zstart': 0,
        #           'xstep': 1/self.data.Fs,  # 对应采样率
        #           'ystep': 1,
        #           'zstep': 1,
        #           'chanlocs': chanlocs,
        #           'history': history,
        #           'events': self.data.marker[0:self.data.marker_idx],
        #           'index_labels': ['index1'],
        #           'epochdata': []}

        header = {'filetype': 'time_amplitude',
                  'name': self.name,
                  'datasize': [1, self.nChannels, 1, 1, 1, self.save_idx],
                  'xstart': 0,
                  'ystart': 0,
                  'zstart': 0,
                  'xstep': 1/self.data.Fs,  # 对应采样率
                  'ystep': 1,
                  'zstep': 1,
                  'chanlocs': chanlocs,
                  'history': history,
                  'events': self.data.marker[0:self.data.marker_idx],
                  'index_labels': ['index1'],
                  'epochdata': []}

        savemat('data\\'+self.name + '.lw6', {'header': header})
        # data = self.save_data[:, :self.save_idx].reshape(1, self.nChannels + 1, 1, 1, 1, self.save_idx)
        data = self.save_data[:, :self.save_idx].reshape(1, self.nChannels, 1, 1, 1, self.save_idx)
        savemat('data\\'+self.name + '.mat', {'data': data})

    def saving_trail(self, trails):
        self.update()
        dt = [('labels', 'S100'),
              ('topo_enabled', 'f'),
              ('theta', 'O'),
              ('radius', 'O'),
              ('sph_theta', 'O'),
              ('sph_phi', 'O'),
              ('sph_theta_besa', 'O'),
              ('sph_phi_besa', 'O'),
              ('X', 'O'),
              ('Y', 'O'),
              ('Z', 'O')]
        ch = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'Cz', 'FC2', 'FC6',
              'F8', 'F4', 'Fz', 'AF4', 'Fp2', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
              'C4', 'T8', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'PO3', 'O1', 'Oz']
        # chanlocs = np.zeros((self.nChannels + 1,), dtype=dt)
        chanlocs = np.zeros((self.nChannels,), dtype=dt)   # 创建了一个包含 nChannels 个通道位置信息的数组 chanlocs，数组中每个元素的数据类型由变量 dt 指定
        # for k in range(self.nChannels + 1):
        for k in range(self.nChannels):     # k 的取值范围是从 0 到 self.nChannels-1
            chanlocs[k]['labels'] = ch[k]
            chanlocs[k]['topo_enabled'] = 0
            chanlocs[k]['theta'] = []
            chanlocs[k]['radius'] = []
            chanlocs[k]['sph_theta'] = []
            chanlocs[k]['sph_phi'] = []
            chanlocs[k]['sph_theta_besa'] = []
            chanlocs[k]['sph_phi_besa'] = []
            chanlocs[k]['X'] = []
            chanlocs[k]['Y'] = []
            chanlocs[k]['Z'] = []

        # savemat('data\\'+self.name + '.lw6', {'header': header})
        # data = self.save_data[:, :self.save_idx].reshape(1, self.nChannels + 1, 1, 1, 1, self.save_idx)
        data = self.save_data[:, :self.save_idx].reshape(1, self.nChannels, 1, 1, 1, self.save_idx)
        savemat('data\\'+self.name + str(trails) + '.mat', {'data': data})



