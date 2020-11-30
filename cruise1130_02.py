# -*- coding: UTF-8 -*-
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

# 常量定义
DRIFT = 280  # 裁剪图像上部
KERNEL = 20  # 开运算的核大小，越大噪声越小，但容易丢失黑线

WIDTH = 640
HEIGHT = 480
ONE_SIDE_OFFSET = 0
MOTOR_MIN = 0.01
MOTOR_MAX = 0.4
STEER_MAX = 1

KP_DISTANCE = 0.015
KP_ANGLE = 0.65


def visualization(img_, text, doshow, dosave, dosavetext, dovideo1, dovideo2):
    global img1, img2

    def show():
        cv2.imshow('image1', img1)
        cv2.imshow('image2', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save():
        # _, img = cv2.VideoCapture(cam).read()
        OUTPUT_DIR = 'images'
        if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR) 
        cv2.imwrite(OUTPUT_DIR + '/' + datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')[:-4]
                    + '.jpg', img_)
    def savetext():
        # _, img = cv2.VideoCapture(cam).read()
        OUTPUT_DIR = 'imagestext'
        imgtext = img_
        position = 30
        for key, value in text.items():
            cv2.putText(imgtext, key + ' = ' + str(value), (20, position), cv2.FONT_HERSHEY_PLAIN, 1.3, (128, 128, 128), 2)
            position += 30
        if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR) 
        cv2.imwrite(OUTPUT_DIR + '/' + datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')[:-4]
                    + 'text.jpg', imgtext)

    def video1():
        _, img1 = cv2.VideoCapture(0).read()
        cv2.imshow('image1', img1)
        cv2.destroyAllWindows()

    def video2():
        _, img2 = cv2.VideoCapture(1).read()
        cv2.imshow('image2', img2)
        cv2.destroyAllWindows()

    if not doshow and not dosave and not dovideo1 and not devideo2: return
    
    
    
    if doshow: show()
    if dosave: save()
    if savetext: savetext()
    if dovideo1: video1()
    if dovideo2: video2()


def process(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 二值化
    image = image[DRIFT:, :]  # 裁剪出图片下部

    image = cv2.GaussianBlur(image, (9, 9), 15)  # 高斯滤波
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 阈值分割
    image = cv2.dilate(cv2.erode(image, np.ones((KERNEL, KERNEL))),
                       np.ones((KERNEL, KERNEL)))  # 开运算
    return image


# get_points输入处理后的图像，返回(left, right)
# 其中left和right均为二维list，分别代表左边和右边从下往上数的四个点坐标

def get_points(image):
    # 四个点的坐标比例
    x1 = 0.9
    x2 = 0.85
    x3 = 0.8
    x4 = 0.75

    left = [[0, 0], [0, 0], [0, 0], [0, 0]]
    right = [[0, 0], [0, 0], [0, 0], [0, 0]]

    # 如果检测不到，left和right中x坐标的初始值
    ln = 0
    rn = image.shape[1]

    for i, xn in enumerate([x1, x2, x3, x4]):
        xn_pos = int(image.shape[0] * xn)

        # bool变量表示是否检测到左边/右边的点
        is_left = bool(np.where(image[xn_pos, :320] == 0)[0].shape[0])
        is_right = bool(np.where(image[xn_pos, 320:] == 0)[0].shape[0])

        # 如果检测到，重新划定x坐标的值
        if is_left:
            ln = np.max(np.where(image[xn_pos, :320] == 0))
        if is_right:
            rn = np.min(np.where(image[xn_pos, 320:] == 0)) + 320

        left[i] = [ln, xn_pos]
        right[i] = [rn, xn_pos]

    return left, right


# draw_points给图像画点
# 可以调节模式，在默认模式下输入处理前的原图像、left和right坐标即可画点

def draw_points(image, left_, right_, mode='origin'):
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8

    for direc in [left_, right_]:
        for point in direc:
            if mode == 'origin':
                point[1] += DRIFT
            cv2.circle(image, (point[0], point[1]), point_size, point_color, thickness)

    return image


def cut(value, bit=3): return round(value, bit)


def get_control(left_, right_):
    def least_squares(x, y):
        x_ = x.mean()
        y_ = y.mean()
        m = np.zeros(1)
        n = np.zeros(1)
        k = np.zeros(1)
        p = np.zeros(1)
        for i in np.arange(x.shape[0]):
            k = (x[i] - x_) * (y[i] - y_)
            m += k
            p = np.square(x[i] - x_)
            n = n + p
        a = m / n
        b = y_ - a * x_
        return a, b

    # def constrain(value, threshold_min, threshold_max):
    #     return min(max(value, threshold_min), threshold_max)

    left_array = np.array(left_)
    right_array = np.array(right_)

    # 判断两组线/一组线
    is_left_good = (left_array[:, 0] != 0).all()
    is_right_good = (right_array[:, 0] != WIDTH).all()
    if is_left_good and is_right_good:
        # print('left and right')  # 两组线
        if (left_array[:, 1] == right_array[:, 1]).all():
            mid_array_x1 = left_array[:, 0] * 0.5 + right_array[:,
                                                    0] * 0.5  # np.concatenate(, left_array[:, 1]).reshape([4, 2])
            mid_array_x2 = left_array[:, 1]
            a, b = least_squares(mid_array_x2, mid_array_x1)  # 求直线
            distance_error = 0.5 * WIDTH - (a * HEIGHT + b)  # 下方交点，横向误差
            angle_error = np.arctan(a)  # in radian
        else:
            print("ERROR! (left_array[:, 1] == right_array[:, 1]).all() is FALSE")
    elif is_left_good:
        print('left')  # 只有左边线
        a, b = least_squares(left_array[:, 1], left_array[:, 0])  # 求直线
        distance_error = - ONE_SIDE_OFFSET
        angle_error = np.arctan(a)  # in radian
    elif is_right_good:
        print('right')  # 只有右边线
        a, b = least_squares(right_array[:, 1], right_array[:, 0])  # 求直线
        distance_error = ONE_SIDE_OFFSET
        angle_error = np.arctan(a)  # in radian
    else:
        print('ERROR! no line!')  # 两边都没有
        distance_error = 0
        angle_error = 0
        # TODO 维持之前的动作，设置一个keep的bool变量

    print('distance_error:', distance_error)
    print('angle_error:', angle_error)

    motor = MOTOR_MIN  # TODO motor 如何变化
    steer = KP_DISTANCE * distance_error + KP_ANGLE * angle_error
    # steer = constrain(steer, -STEER_MAX, STEER_MAX)
    steer = np.clip(steer, -STEER_MAX, STEER_MAX)

    text_dict = collections.OrderedDict()
    text_dict['Time'] = time.strftime("%Y-%m-%d %H-%M-%S")
    text_dict['Left'] = left_
    text_dict['Right'] = right_
    text_dict['Dist_err'] = int(distance_error)
    text_dict['Ang_err'] = cut(angle_error)
    text_dict['KP_DISTANCE'] = cut(KP_DISTANCE)
    text_dict['KP_ANGLE'] = cut(KP_ANGLE)
    text_dict['Motor'] = cut(motor)
    text_dict['Steer'] = cut(steer)
    return motor, steer, text_dict

def detect_angle_error(left, right):
    basepoint=[377, 480]

    delta_right=np.arctan((basepoint[1]-right[-1][1])/(right[-1][0]-basepoint[0]))
    delta_left=np.arctan((basepoint[1]-left[-1][1])/(basepoint[0]-left[-1][0]))
    return delta_right-delta_left  #当车辆偏左行驶时，返回误差角为负

def get_control_new(left_, right_):  #只修改了两组线时的angle_erro获取方式，
    def least_squares(x, y):
        x_ = x.mean()
        y_ = y.mean()
        m = np.zeros(1)
        n = np.zeros(1)
        k = np.zeros(1)
        p = np.zeros(1)
        for i in np.arange(x.shape[0]):
            k = (x[i] - x_) * (y[i] - y_)
            m += k
            p = np.square(x[i] - x_)
            n = n + p
        a = m / n
        b = y_ - a * x_
        return a, b

    # def constrain(value, threshold_min, threshold_max):
    #     return min(max(value, threshold_min), threshold_max)

    left_array = np.array(left_)
    right_array = np.array(right_)

    # 判断两组线/一组线
    is_left_good = (left_array[:, 0] != 0).all()
    is_right_good = (right_array[:, 0] != WIDTH).all()
    if is_left_good and is_right_good:
        # print('left and right')  # 两组线
        if (left_array[:, 1] == right_array[:, 1]).all():
            mid_array_x1 = left_array[:, 0] * 0.5 + right_array[:,
                                                    0] * 0.5  # np.concatenate(, left_array[:, 1]).reshape([4, 2])
            mid_array_x2 = left_array[:, 1]
            a, b = least_squares(mid_array_x2, mid_array_x1)  # 求直线
            distance_error = 0.5 * WIDTH - (a * HEIGHT + b)  # 下方交点，横向误差

            angle_error = detect_angle_error(left_, right_)  # in radian

        else:
            print("ERROR! (left_array[:, 1] == right_array[:, 1]).all() is FALSE")
    elif is_left_good:
        print('left')  # 只有左边线
        a, b = least_squares(left_array[:, 1], left_array[:, 0])  # 求直线
        distance_error = - ONE_SIDE_OFFSET
        angle_error = np.arctan(a)  # in radian
    elif is_right_good:
        print('right')  # 只有右边线
        a, b = least_squares(right_array[:, 1], right_array[:, 0])  # 求直线
        distance_error = ONE_SIDE_OFFSET
        angle_error = np.arctan(a)  # in radian
    else:
        print('ERROR! no line!')  # 两边都没有
        distance_error = 0
        angle_error = 0
        # TODO 维持之前的动作，设置一个keep的bool变量

    print('distance_error:', distance_error)
    print('angle_error:', angle_error)

    motor = MOTOR_MIN  # TODO motor 如何变化
    steer = KP_DISTANCE * distance_error + KP_ANGLE * angle_error
    # steer = constrain(steer, -STEER_MAX, STEER_MAX)
    steer = np.clip(steer, -STEER_MAX, STEER_MAX)

    text_dict = collections.OrderedDict()
    text_dict['Time'] = time.strftime("%Y-%m-%d %H-%M-%S")
    text_dict['Left'] = left_
    text_dict['Right'] = right_
    text_dict['Dist_err'] = int(distance_error)
    text_dict['Ang_err'] = cut(angle_error)
    text_dict['KP_DISTANCE'] = cut(KP_DISTANCE)
    text_dict['KP_ANGLE'] = cut(KP_ANGLE)
    text_dict['Motor'] = cut(motor)
    text_dict['Steer'] = cut(steer)
    return motor, steer, text_dict

def get_img(camtype='front'):
    if camtype == 'front':
        _, img = cv2.VideoCapture(1).read()
    elif camtype == 'back':
        _, img = cv2.VideoCapture(0).read()
    return img


def control(d, motor, steer):
    # global time_before
    d.setStatus(motor=motor, servo=steer)
    # current = time.time()
    print('Time:', datetime.now().strftime('%H:%M:%S.%f')[:-4],'Motor:', motor, ',Steer: ', steer)
    # time_before = current

def cruise():
    d = driver()
    d.setStatus(mode="speed")
    isfirst = True

    while 1:
        try:
            img = get_img(camtype='front')
            black_line_img = process(img)  # 处理后的图像
            left, right = get_points(black_line_img)
            # print('left:', left, ', right:', right)
            # draw_point_img = draw_points(img, left, right)  # 对原图像画点
            # cv2.imwrite('./output/' + str(idx) + '.jpg', black_line_img)   # 保存处理后的未画点图像
            # cv2.imwrite('./process/' + str(idx) + '.jpg', draw_point_img)  # 保存画点后的原图像
            motor, steer, text_dict = get_control(left, right)
            # motor, steer, text_dict = get_control_new(left, right)
            


            # visualization(img_=img, text=text_dict, doshow=False, dosave=True, dosavetext=True, dovideo1=False, dovideo2=False)
            control(d, motor, steer)
        except KeyboardInterrupt:
            break

    d.setStatus(motor=0.0, servo=0.0, dist=0x00, mode="stop")
    d.close()
    del d



if __name__ == '__main__':
    cruise()
