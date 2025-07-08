import binascii
import threading
import queue,traceback
import time
import numpy as np
from multiprocessing import Process,Queue,Value,Lock
import datetime
import socket
import re

TRIGGER_SIGNAL = 10
TRIGGER_SIGNAL_START=11
TRIGGER_IDLE = 30
TRIGGER_IDLE_START = 31
TRIGGER_END = 101
TRIGGER_TERMINATED = 102
TRIGGER_F0 = 20


def negetive_trans(width, data):
    dec_data = int(data, 16)
    if dec_data >= (1 << (width - 1)):
        dec_data -= (1 << width)
    return dec_data
    #将给定的十六进制数据正确地转换为有符号整数

def cut(obj, sec):
    return [obj[i:i + sec] for i in range(0, len(obj), sec)]


class eCon_iRecorder(Process):
    def __init__(self,raw_data:Queue,trigger:Value):
        Process.__init__(self,daemon=True)          #当 daemon 参数被设置为 True 时，表示将该进程设置为守护进程，守护进程会随着主进程的结束而结束。
        self.raw_data=raw_data
        self.cap_status=Value('i',TRIGGER_IDLE_START)   #TRIGGER_IDLE_START = 11
        self.trigger=trigger
        self.data_lock=Lock()
        self.sock = None
        self.host = "192.168.4.1"  # IPv4地址
        self.port = 4321  # 端口号
        self.bf = ""
        self.next = 0
        self.last = 0
        self.colflag = True
        self.runflag = True

    def startAcquisition(self):
        self.cap_status.value=TRIGGER_SIGNAL_START
        print("Acquisition started")

    def stopAcquisition(self):
        self.cap_status.value=TRIGGER_IDLE_START
        print("Acquisition stopped")

    def wifi_disconnect(self):
        self.cap_status.value=TRIGGER_END
        while(self.cap_status.value!=TRIGGER_TERMINATED): # 确保socket断开
            pass
        print("Disconnected from Wi-Fi")

    def getData(self,):
        data = []
        self.data_lock.acquire()
        try:
            while (not self.raw_data.empty()):
                tep=self.raw_data.get(timeout=0.01)
                data.append(tep)
        except queue.Empty:
            pass
        except Exception:
            traceback.print_exc()
        self.data_lock.release()
        data= np.array(data)  # 每列为一个时刻的数据
        return data
    #从一个队列中获取数据并将其存储为 NumPy 数组的功能，同时使用锁确保在多线程环境下对共享数据的安全访问


    def recv_thread(self):
        while self.runflag:
            
            if self.colflag == False:
                time.sleep(0.5)
            try:
                data = self.sock.recv(4096) # block issue
                if data is not None:
                    self.qdt.put(data)
            except Exception:
                # return
                pass
            #实现了一个后台线程，用于持续接收数据并放入队列中，以供后续处理

    def run(self):

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print("Connected to Wi-Fi")
        if not self.sock:
            return 
        self.qdt=queue.Queue()
        recv_data = threading.Thread(target=self.recv_thread,daemon=True)
        recv_data.start()
        self.cap_status.value = TRIGGER_IDLE_START # 默认进入空闲状态
        while True:
            if self.cap_status.value == TRIGGER_SIGNAL_START:
                self.bf = ""
                while not self.raw_data.empty():
                    try:self.raw_data.get(block=False)
                    except queue.Empty:pass
                while not self.qdt.empty():
                    try: self.qdt.get(block=False)
                    except queue.Empty:pass
                self.colflag = True
                self.sock.send("W".encode())
                self.cap_status.value = TRIGGER_SIGNAL
                self.timestamp=time.time()
                print('started gathering data')

            elif self.cap_status.value == TRIGGER_SIGNAL:
                try:
                    q = self.qdt.get(timeout=0.01)
                    data_list = self.parse(q)
                    if data_list is not None:
                        self.timestamp=time.time()
                        for data in data_list:
                            if data is not None:
                                if self.trigger.value != TRIGGER_F0:
                                    data[-3] = self.trigger.value - TRIGGER_F0
                                    self.trigger.value = TRIGGER_F0
                                self.data_lock.acquire()
                                self.raw_data.put(data, block=False)
                                self.data_lock.release()
                except queue.Empty:
                    if(time.time()-self.timestamp)>5:
                        print('transmission timeout')
                        self.sock.close()
                        self.runflag=False
                        return
                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    self.sock.close()
                    self.runflag=False
                    return

            elif self.cap_status.value == TRIGGER_IDLE_START:
                print("idle")
                self.sock.send("R".encode('ascii'))
                self.colflag = False
                self.cap_status.value = TRIGGER_IDLE
                self.timestamp=time.time()

            elif self.cap_status.value == TRIGGER_IDLE:
                time.sleep(0.5)
                if (time.time()-self.timestamp)>5:
                    try:
                        self.sock.send('B'.encode())
                        # self.sock.send(b'\xFF')    # heartbeat to keep socket alive
                        self.timestamp=time.time()
                        print('Staying alive...')
                    except:
                        print('heartbeat lost')
                        self.sock.close()
                        self.runflag=False
                        return

            elif self.cap_status.value == TRIGGER_END:
                try:
                    self.sock.send("R".encode())
                    self.runflag = False
                    self.sock.close()
                    self.sock = None
                    recv_data.join(timeout=1)
                except Exception as e: print(e)
                self.cap_status.value=TRIGGER_TERMINATED
            else:
                pass


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
    # 涉及到数据帧的处理、校验和校验、数据解析等功能

if __name__ == "__main__":
    # 用于测试本模块的各项功能，与实验本身无关
    raw_data_queue=Queue()
    cap_status=Value('i',TRIGGER_IDLE_START)
    trigger=Value('i',TRIGGER_F0)

    # device = eCon_iRecorder(raw_data_queue,cap_status,trigger)    #报错，传参多了一个cap_status
    device = eCon_iRecorder(raw_data_queue,trigger)
    device.start() # 调用run方法 连接Wi-Fi并进入idle状态
    time.sleep(1)

    # device.startAcquisition() deprecated, use following shared value
    # cap_status.value=TRIGGER_SIGNAL_START
    device.startAcquisition()
    time.sleep(1)

    # # 
    start=time.time()
    while((time.time()-start)<2):
        device.getData()
    device.stopAcquisition()

    device.wifi_disconnect()

    device.terminate()
    time.sleep(1)
