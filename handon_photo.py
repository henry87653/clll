from driver import *
import cv2
import numpy as np
from math import atan as arctan
from math import asin as arcsin
from math import tan
import time
import os
import collections
from datetime import datetime

def visualization(doshow, dosave, dovideo1,dovideo2):
    
    global img1,img2
    
    def show():
        cv2.imshow('image1', img1)
        cv2.imshow('image2', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save():
        _, img = cv2.VideoCapture(0).read()
        OUTPUT_DIR = 'images'
        if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
        cv2.imwrite(OUTPUT_DIR + '/' + datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')[:-4]
 + '.jpg', img)
    
    def video1():
        while 1:
            _, img1 = cv2.VideoCapture(0).read()
            cv2.imshow('image1', img1)
            if cv2.waitKey(1) & 0xFF == ord('s'): save()
            if cv2.waitKey(1) & 0xFF == 27: break
        cv2.destroyAllWindows()
        
    def video2():
        while 1:
            _, img2=cv2.VideoCapture(1).read()
            cv2.imshow('image2', img2)
            if cv2.waitKey(1) & 0xFF == ord('p'): save()
            if cv2.waitKey(1) & 0xFF == 27: break
        cv2.destroyAllWindows()

    if not doshow and not dosave and not dovideo1 and not devideo2: return

    if doshow: show()
    if dosave: save()
    if dovideo1:video1()
    if dovideo2:video2()

def cruise():
    d = driver()
    d.setStatus(mode="speed")
    isfirst = True

    while 1:
        try:
            # _, img = get_img(camera, closed_size)
            # alpha, dist, black_depth, keep = get_error(img, up_step, black_step_max, dist_diff_max, black_depth_min, black_depth_max)
            # if isfirst or not keep: motor, steer, text_dict = get_stanley_control(img, alpha, dist, motorAlphaPara, motorDistPara, motorMax, motorMin, steerAlphaPara, steerDistPara, steerMax)
            # isfirst = False
            
            # visualization(img, text_dict, closed_size, black_depth, doshow=False, dosave=False, dovideo=False)
            visualization(doshow=False, dosave=True, dovideo1=False, dovideo2=False)
            motor = -0.1
            steer = 0.0
            control(d, motor, steer)
        except KeyboardInterrupt: break
    
    d.setStatus(motor=0.0, servo=0.0, dist=0x00, mode="stop")
    d.close()
    del d

def control(d, motor, steer):
    # global time_before
    d.setStatus(motor=motor, servo=steer)
    # current = time.time()
    print('[ Motor', motor, '] [ Steer', steer, ']\n')#  [ Time', cut(current - time_before), ']
    # time_before = current

if __name__ == '__main__':
    cruise()