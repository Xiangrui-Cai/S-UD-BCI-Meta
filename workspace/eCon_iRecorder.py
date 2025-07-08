import binascii
import numpy as np
import datetime
import socket
import re
from scipy.io import savemat

def negetive_trans(width,data):

    dec_data = int(data, 16)
    if dec_data > 2 ** (width - 1) - 1:
        dec_data = 2 ** width - dec_data
        dec_data = 0 - dec_data
    return dec_data
    #将十六进制数据转换为负数的功能，根据数据宽度和具体数值进行判断和计算，得到转换后的负数结果

def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]
    #用于将一个对象按照指定长度进行切割，并返回切割后的子序列组成的列表

class eCon_iRecorder:
    def __init__(self):
        self.sock = None
        self.host = "192.168.4.1"  # IPv4地址
        self.port = 4321  # 端口号
        self.bf = ""
        self.next = 0
        self.last = 0

    def wifi_connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print("Connected to Wi-Fi")
        #用于在设备上建立 WiFi 连接

    def wifi_disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Disconnected from Wi-Fi")
        # 用于在设备上取消 WiFi 连接

    def startAcquisition(self):
        if self.sock:
            # self.sock.send("T".encode('ascii'))
            self.sock.send("W".encode('ascii'))  # 测试时可以用“T”，生成方波   采集真实脑电信号时记得改为“W”
            print("Acquisition started")
            #开始采集脑电信号

    def saver(self):
        if self.sock:
            self.sock.send("R".encode('ascii'))
            print("Acquisition stopped")
            #停止采集脑电信号

    def getData(self):
        if self.sock:
            buffer = self.sock.recv(4096)  # 根据实际情况调整缓冲区大小 256
            parsed_data = self.parse(buffer)
            parsed_data_array = np.array(parsed_data)[:, :]  # 每列为1个通道
            return parsed_data_array
        else:
            return []
        #获取通过网络传输的脑电信号数据的一部分

    def parse(self, buffer):
        bf = str(binascii.b2a_hex(buffer))[2:-1]
        self.bf += bf  # 将接收到的数据追加到self.bf中

        pattern = re.compile(r'bbaa[0-9A-Za-z]{200}')

        frame_list = pattern.findall(self.bf)
        self.bf = re.sub(pattern, '', self.bf)

        parsed_data = []
        for frame in frame_list:
            dd = []
            d = cut(frame[4:-2], 6)
            self.next = int(frame[-2:], 16)
            if (self.next == self.last + 1) or (self.next == 0 and self.last == 255):
                qwe = 1
            else:
                print('丢包')
                time1 = datetime.datetime.now()
                print(time1)
                print("上一个包序号：{}，当前包序号：{}".format(self.last, self.next))
            self.last = self.next
            check_sum_str = hex((~sum([int(i, 16) for i in cut(frame[4:-8], 2)])) & 0xFFFFFFFF)[-2:]
            vaild = frame[-8:-6]
            if (vaild != check_sum_str):
                print('校验位未通过')
                time1 = datetime.datetime.now()
                print(time1)
                continue
            for i in range(32):  # 32通道
                x = negetive_trans(24, d[i])
                dd.append(x)

            dd.append(int(frame[-8:-6], 16))  # 校验位
            dd.append(int(frame[-6:-4], 16))  # Trigger位
            # dd.append(2)
            dd.append(int(frame[-4:-2], 16))  # 电量
            dd.append(int(frame[-2:], 16))  # 包序号
            parsed_data.append(dd)
        return parsed_data
    #实现了对接收到的特定格式数据帧的解析和处理

if __name__ == "__main__":
    # 用于测试本模块的各项功能，与实验本身无关

    device = eCon_iRecorder()  # 替换为你的设备的 MAC 地址

    device.wifi_connect()
    device.startAcquisition()

    all_data = np.empty((36,))
    end_time = datetime.datetime.now() + datetime.timedelta(seconds=10)

    while datetime.datetime.now() < end_time:
        data = device.getData()  # data的大小与buffer长度有关
        # data = data.T
        all_data = np.vstack((all_data, data))

    savemat('all_data.mat', {'all_data': all_data})

    device.stopAcquisition()
    device.wifi_disconnect()
