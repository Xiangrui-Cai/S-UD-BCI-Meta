# import random
import time
import sys
import pygame
# from OpenGL.GL import *
# from OpenGL.GLU import *
from multiprocessing import Process, Manager
# from scipy import signal
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
TimeInterval1 = 0
TimeInterval2 = 5
TimeInterval3 = 0

pygame.mixer.init()

def run_proc(share_var):
    pygame.init()
    infoObject = pygame.display.Info()
    screen_width, screen_height = infoObject.current_w, infoObject.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    screen.fill([0, 0, 0])
    pygame.mouse.set_visible(False)

    centerX = screen.get_width() // 2
    centerY = screen.get_height() // 2

    font1 = pygame.font.Font("static/NotoSansSC-Light.ttf", 60)
    text1 = font1.render("请注视屏幕并保持平静", True, (255, 255, 255))
    startmsg_rect = text1.get_rect(center=(centerX, centerY - 200))
    screen.blit(text1, startmsg_rect)

    font2 = pygame.font.Font("static/NotoSansSC-Light.ttf", 30)
    text2 = font2.render("Press Space to start", True, (255, 255, 255))
    startmsg_rect = text2.get_rect(center=(centerX, centerY + 200))
    screen.blit(text2, startmsg_rect)

    pygame.display.flip()
    share_var[0] = 0
    while share_var[0] == 0:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    share_var[0] = 1
                    screen.fill([0, 0, 0])

    w, h = 300/1.5, 60/1.5
    if share_var[1] == 1:  # BCI_Type.REST_EO 白色矩形
        pygame.draw.rect(screen, [255, 255, 255],
                         np.array([screen_width / 2 - w / 2, screen_height / 2 - h / 2, w, h]), 0)
        pygame.draw.rect(screen, [255, 255, 255],
                         np.array([screen_width / 2 - h / 2, screen_height / 2 - w / 2, h, w]), 0)
    else:                  # BCI_Type.REST_EC 灰色矩形
        pygame.draw.rect(screen, [128, 128, 128],
                         np.array([screen_width / 2 - w / 2, screen_height / 2 - h / 2, w, h]), 0)
        pygame.draw.rect(screen, [128, 128, 128],
                         np.array([screen_width / 2 - h / 2, screen_height / 2 - w / 2, h, w]), 0)

    pygame.display.flip()

    share_var[0] = 1
    while share_var[0]:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    share_var[0] = 0
                    pygame.quit()
                    sys.exit()

class Decoder_REST:
    def __init__(self, data):
        self.data = data

    def run(self):
        share_var = Manager().list()
        share_var.append(0)  # is_run
        share_var.append(self.data.type_idx)
        state = 1
        p = Process(target=run_proc, args=(share_var,))
        p.start()

        while share_var[0] == 0:
            pass

        self.data.init_eCon()

        t_display = time.perf_counter()
        if share_var[1] == 1:
            pass
        else:
            pass

        while share_var[0] == 1:
            self.data.setData()     # datamanager之中的函数

            if state == 1:
                if time.perf_counter() - t_display > TimeInterval1:
                    t_display = time.perf_counter()
                    self.data.sendMarker(1)
                    state = 2
            elif state == 2:
                if time.perf_counter() - t_display > TimeInterval2:
                    t_display = time.perf_counter()
                    self.data.sendMarker(2)
                    state = 3
            elif state == 3:
                if time.perf_counter() - t_display > TimeInterval3:
                    self.data.close_eCon()
                    share_var[0] = 0
        p.join()
