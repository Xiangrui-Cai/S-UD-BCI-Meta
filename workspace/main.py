import os
import time
import DataManager
import Decoder_MI_zaixian
import Decoder_REST
from enum import Enum
import pygame
from multiprocessing import Process, Manager


class BCI_Type(Enum):
    REST_EO = 1
    REST_EC = 2
    MI_real_time = 4

def run_proc(share_var):
    share_var[0] = 1


if __name__ == '__main__':
    const_center = 1  #
    const_sub = 2  # 受试者编号
    const_exp = 1  # 受试者实验次数

    totalRun = 9

    for const_run in range(1, 3):

        if const_run == 1:
            const_type_idx = BCI_Type.REST_EO
        elif const_run == 2:
            const_type_idx = BCI_Type.MI_real_time
        elif const_run == 3:
            const_type_idx = BCI_Type.MI_real_time

        t = time.localtime(time.time())   # 获取当前时间，并存储在变量 t 中
        const_date = '%04d%02d%02d_%02d%02d%02d' % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
        filename = 'Cent%01d Sub%03d Exp%01d Run%02d %s' % (const_center, const_sub, const_exp, const_run,
                                                               const_type_idx.name)

        if const_run >= 2 and const_run <= 7:
            datafiles = os.listdir('./data')    # 列出指定目录 ./data 下的所有文件和子目录的名称，并返回一个包含这些名称的列表
            dataname = './data/' + datafiles[-1].split('.')[0] + 'data.mat'
            # 构建出一个特定格式的数据文件路径

        print('filename: ' + filename)

        eeg = DataManager.DataManager(filename)     # 调用datamanager里面的函数
        eeg.eCon.start()  # wifi连接
        data_temp = eeg.eCon.getData()        # 获得数据
        time.sleep(1)   # 休眠 1 秒钟
        eeg.type_idx = const_type_idx.value      # const_type_idx的值
        if const_type_idx == BCI_Type.REST_EO:
            decoder = Decoder_REST.Decoder_REST(eeg)      # 进行Decoder_REST的run
        elif const_type_idx == BCI_Type.REST_EC:
            decoder = Decoder_REST.Decoder_REST(eeg)
        elif const_type_idx == BCI_Type.MI_zaixian:
            decoder = Decoder_MI_zaixian.Decoder_MI(eeg, const_run, train_data='')
        decoder.run()
        eeg.eCon.wifi_disconnect()  # wifi断开
        eeg.eCon.terminate()  # 关掉进程

        share_var = Manager().list()
        share_var.append(0)  # is_run

        p = Process(target=run_proc, args=(share_var, ))
        p.start()
        while share_var[0] == 0:
            pass
        p.join()
