import time
import sys
from multiprocessing import Process, Manager
import pylsl
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import os
from scipy.io import savemat, loadmat
import pygame
import ceshi_ann as suanfa
import cv2
import serial
import time
import winsound
import random
import main
import cnn_model
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

Trial_NUM = 40

pygame.mixer.init()
FPS = 30
FramePerSec = pygame.time.Clock()

def send_hex_command(ser, hex_command):
    """
    在已打开的串口上发送十六进制命令并接收响应

    参数:
        ser (Serial): 已打开的串口对象
        hex_command (str): 十六进制命令字符串
    """
    try:
        # 将十六进制字符串转换为字节数据
        byte_command = bytes.fromhex(hex_command.replace(" ", ""))
        ser.write(byte_command)
        print(f"已发送命令（十六进制）: {hex_command}")

        # 等待并读取响应
        time.sleep(0.1)  # 根据设备响应时间调整
        response = ser.read_all()

        if response:
            hex_response = " ".join([f"{byte:02X}" for byte in response])
            print(f"收到响应（十六进制）: {hex_response}")
        else:
            print("未收到响应")

    except Exception as e:
        print(f"发生错误: {e}")

def run_proc(share_var, runNum):
    pygame.init()
    infoObject = pygame.display.Info()
    screen_width, screen_height = infoObject.current_w, infoObject.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN, pygame.NOFRAME, display=0)

    centerX = screen.get_width() // 2
    centerY = screen.get_height() // 2
    lx, ly = (screen_width/2.5, screen_height/2.5)
    yBias = ly/8
    xBias = lx/4

    screen.fill([0, 0, 0])

    pygame.mouse.set_visible(True)
    font1 = pygame.font.Font("static/NotoSansSC-Light.ttf", 60)
    text1 = font1.render("MI", True, (255, 255, 255))
    startmsg_rect = text1.get_rect(center=(centerX, centerY - 200))
    screen.blit(text1, startmsg_rect)
    font2 = pygame.font.Font("static/NotoSansSC-Light.ttf", 30)
    text2 = font2.render("Press Space to start", True, (255, 255, 255))
    startmsg_rect = text2.get_rect(center=(centerX, centerY + 200))
    screen.blit(text2, startmsg_rect)
    font3 = pygame.font.SysFont('Times New Roman', 20, bold=True)

    share_var[0] = 0
    pygame.display.flip()
    while share_var[0] == 0:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    share_var[0] = 1
                    screen.fill([0, 0, 0])

    share_var[0] = 1

    while share_var[0]:
        screen.fill((0, 0, 0))
        text3 = font3.render(f"ALL: {share_var[8]}", True, (0, 255, 0))
        text4 = font3.render(f"RUN: {runNum}", True, (0, 255, 0))
        screen.blit(text3, (centerX + centerX*5/8, centerY - centerY*3/4 - 40))
        screen.blit(text4, (centerX + centerX*5/8, centerY - centerY*3/4))

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    share_var[0] = 0
                    pygame.quit()
                    sys.exit()

        if share_var[2] > 0:
            if share_var[2] == 1:  # 触发双手同时震动
                if share_var[6] == 1:
                    share_var[6] = 0

            elif share_var[2] == 2:
                # 绘制cue
                pygame.draw.rect(screen, [255, 0, 0], [centerX - xBias, centerY - yBias, centerX - startX, 2 * yBias])
                # 中间  下面的代码是在左边
                # pygame.draw.rect(screen, [255, 0, 0], [startX, centerY - yBias, centerX - startX, 2 * yBias])
                # 绘制一个红色矩形
                # 如果没有开刺激，打开
                if share_var[7] == 1:
                    share_var[7] = 0  # 确保仅触发一次

            elif share_var[2] == 3:
                pass

            elif share_var[2] == 4:  # 刺激结束，需要清除屏幕
                pass

            elif share_var[2] == 5:  # 反馈左手，正确
                if share_var[9] == 1:  # 确保仅触发一次
                    share_var[9] = 0

            elif share_var[2] == 6:  # 反馈右手，错误
                if share_var[9] == 1:
                    share_var[9] = 0

        if share_var[3] > 0:
            if share_var[3] == 1:  # 触发双手同时震动
                if share_var[6] == 1:
                    share_var[6] = 0

            elif share_var[3] == 2:
                # 绘制cue
                pygame.draw.rect(screen, [255, 0, 0], [centerX - xBias, centerY - yBias, endX - centerX, 2*yBias])

                if share_var[7] == 1:
                    share_var[7] = 0  # 确保仅触发一次

            elif share_var[3] == 3:
                pass

            elif share_var[3] == 4:  # 刺激结束，需要清除屏幕
                pass

            elif share_var[3] == 5:  # 反馈右手，正确
                if share_var[9] == 1:  # 确保仅触发一次
                    share_var[9] = 0

            elif share_var[3] == 6:  # 反馈右手，错误
                if share_var[9] == 1:
                    share_var[9] = 0

        if share_var[1] == 1:  # 展示十字架
            startX, endX = -(lx // 2) + centerX, lx // 2 + centerX
            pygame.draw.line(screen, (255, 255, 255), (startX, centerY), (endX, centerY), width=2)
            startY, endY = -(ly // 2) + centerY, ly // 2 + centerY
            pygame.draw.line(screen, (255, 255, 255), (centerX, startY), (centerX, endY), width=2)

        FramePerSec.tick(FPS)
        pygame.display.update()
    pygame.quit()
    sys.exit()

### # Decoder_MI
class Decoder_MI:
    def __init__(self, data, runNum, train_data=''):
        self.data = data
        self.marker_idx = 0
        if train_data == '':
            self.is_train = 1
            self.v = 0
            self.clf = 0
        else:
            self.is_train = 0
            temp = loadmat(train_data)
            C = temp['C']    #从 .mat 文件中获取名为 'C' 的变量
            label = np.squeeze(temp['label'])
            # self.train(C, label)

        self.trial_idx = 0
        self.C = np.zeros((32, 1500, Trial_NUM * 2))              # 按需修改，需与后面get_cov选择的通道总数一致
        self.label = []  # 各个trial的任务类型标签，结果生成一个【1.2.1.2.1……】的列表
        # self.result = 0
        self.runNum = runNum
        # random.shuffle(self.label)    # 对lable重新打乱重排

    # 从 EEG 数据中提取 第 1 秒到第 4 秒（共 3 秒） 的所有通道的数据，并返回
    def get_cov1(self):
        fs = self.data.Fs  # 采样率
        time_idx = [int((1 * fs + x + self.data.DATA_LENGTH) % self.data.DATA_LENGTH)
                    for x in range(int(3 * fs))]  # 取2-5s内数据进行处理
        eeg = self.data.buff_data[:, time_idx]
        return eeg

    def get_cov2(self):
        fs = self.data.Fs  # 采样率
        time_idx = [int((2500 + 1 * fs + x + self.data.DATA_LENGTH) % self.data.DATA_LENGTH)
                    for x in range(int(3 * fs))]  # 取5-8s内数据进行处理
        eeg = self.data.buff_data[:, time_idx]
        return eeg

    def run(self):
        port_name = 'COM3'  # 修改为你的串口号
        # 一次性打开串口
        ser = serial.Serial(
            port=port_name,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print(f"成功连接到串口 {port_name}")
        share_var = Manager().list()

        send_hex_command(ser, "55 bb 06 02  00 02 00 0a")  # 设备工作状态到空闲

        send_hex_command(ser, "55 bb 06 02  00 02 01 0b")  # 循环刺激

        send_hex_command(ser, "55 BB 0D 02 01 04 00 14 14 00 00 04 00 02 42")  # 设置通道参数

        send_hex_command(ser, "55 bb 06 02  01 08 02 13")  # 切换到电流调节

        # send_hex_command(ser, "55 BB 07 02 01 06 00 05 15")  # 调节电流
        #
        # send_hex_command(ser, "55 BB 07 02 01 06 00 05 15")  # 调节电流

        send_hex_command(ser, "55 BB 07 02 01 06 00 05 15")  # 调节电流

        send_hex_command(ser, " 55 bb 06 02  01 08 03 14")  # 正常工作
        time.sleep(0.5)
        send_hex_command(ser, "55 BB 06 02 01 08 01 12")  # 暂停

        share_var.append(0)  # is_run      0
        share_var.append(0)  # cross display  十字显示   1
        share_var.append(0)  # left bar     2
        share_var.append(0)  # right bar     3
        share_var.append(0)  # down bar     4
        share_var.append(self.data.type_idx)  # MI or ME     5
        share_var.append(1)  # burst on(0) or off(1)      6
        share_var.append(1)  # stimulation on(0) or off(1)      7
        share_var.append(self.trial_idx)  # Trial Num     8
        share_var.append(1)  # feed back stimulation on(0) or off(1)     9

        state = 1

        p = Process(target=run_proc, args=(share_var, self.runNum))

        null_data = suanfa.zaixian_suanfa_null()
        # 在新的进程中并行执行 run_proc 函数
        p.start()
        while share_var[0] == 0:
            pass

        aaa = []

        self.data.init_eCon()
        # t_eeg = time.perf_counter()
        t_display = time.perf_counter()
        while share_var[0] == 1:
            self.data.setData()
            share_var[8] = self.trial_idx + 1
            if state == 1:  # 准备时间
                if time.perf_counter() - t_display >= 0:
                    self.label.append(0)
                    self.data.sendMarker(0)
                    t_display = time.perf_counter()
                    state = 2

            if state == 2:
                # 绘制十字架
                share_var[1] = 1
                if time.perf_counter() - t_display >= 2:    # 2s后（无用）
                    share_var[1 + 1] = 1

                if time.perf_counter() - t_display >= 5:   # 5s cue
                    self.label.append(1)
                    self.C[:, :, self.trial_idx * 2 - 1] = self.get_cov1()
                    self.data.sendMarker(1)  # 打标
                    t_display = time.perf_counter()
                    state = 3

            elif state == 3:
                if share_var[1 + 1] != 3:    # 如果没有要求停止绘制cue
                    share_var[1 + 1] = 2
                    # pass

                if time.perf_counter() - t_display >= 1.5:   # 1.5s后，结束cue
                    share_var[1 + 1] = 3

                if time.perf_counter() - t_display >= 3:   # 3s后，保存数据, 需要注意的是， 这个3s和上面的1.5s是同步进行的
                    t_display = time.perf_counter()  # 记录时间, 这里的t_display代表着rest时间的开始
                    self.data.saving_trail(self.trial_idx + 1)
                    state = 4

            elif state == 4:
                # 添加一个标志变量，比如 'executed'，确保只执行一次
                if not hasattr(self, 'executed'):
                    self.executed = False
                if not self.executed:
                    result = suanfa.zaixian_suanfa(self.trial_idx + 1)
                    if result == 1:
                        aaa.append(1)
                        send_hex_command(ser, "55 bb 06 02  01 08 03 14")  # 正常工作
                        time.sleep(6)
                        send_hex_command(ser, "55 BB 06 02 01 08 01 12")  # 暂停
                    else:
                        aaa.append(0)
                    print(aaa)
                    self.executed = True  # 标记为已执行

                if time.perf_counter() - t_display >= 7:
                    print("清除十字架")
                    share_var[1] = 0  # 清除十字架
                    t_display = time.perf_counter()  # 记录时间
                    self.executed = False
                    state = 5

            elif state == 5:

                if time.perf_counter() - t_display > 5:   # 休息时间
                    print("开始休息")

                    self.C[:, :, self.trial_idx * 2] = self.get_cov2()
                    self.trial_idx += 1
                    if self.trial_idx == Trial_NUM:
                        share_var[0] = 0
                        filename = self.data.saver.name
                        savemat('data\\' + filename + " data.mat", {"C": self.C, "label": self.label})
                        self.data.close_eCon()
                        # 保存数据保存数据关闭设备
                    else:
                        share_var[1] = 0
                        share_var[2] = 0
                        share_var[3] = 0
                        share_var[4] = 0
                        share_var[6] = 1
                        share_var[7] = 1
                        share_var[9] = 1
                        t_display = time.perf_counter()
                        state = 1
                        print("再来一次")
                        # 再来一次
        p.join()