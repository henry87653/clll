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
MOTOR_MIN = 0.01  # 0.1
MOTOR_MAX = 0.01  # 0.15
STEER_MAX = 1

KP_DISTANCE = 0.00  # 0.015
KP_ANGLE = 0.65
SLEEP_TIME = 0.3
MID_POS = 377

INTUITION_WIDTH = 100
STEER_SHARP = 0.6
STEER_MILD = 0.3

K_BAD = 0.5

camera = cv2.VideoCapture(1)  # front
# camera = cv2.VideoCapture(0) # back


PARK_SLEEP_TIME = 0.3
PARK_POS = 1

PARK_TIME0_1 = 2
PARK_TIME1_1 = 38 # 37
PARK_TIME2_1 = 40 # 39
PARK_TIME3_1 = 30
PARK_SPEED_1 = -0.01
PARK_STEER_1 = 1

PARK_TIME0_2 = 2
PARK_TIME1_2 = 25
PARK_TIME2_2 = 26    # 27.5
PARK_TIME3_2 = 33
PARK_SPEED_2 = -0.01
PARK_STEER_2 = 0.6

PARK_TIME0_3 = 2
PARK_TIME1_3 = 29 # 25
PARK_TIME2_3 = 23.5 # 25.5
PARK_TIME3_3 = 31 # 33
PARK_SPEED_3 = -0.01
PARK_STEER_3 = 0.6

PARK_TIME0_4 = 2
PARK_TIME1_4 = 39
PARK_TIME2_4 = 36
PARK_TIME3_4 = 31
PARK_SPEED_4 = -0.01
PARK_STEER_4 = 1


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
                    + '.jpg', draw_points(img_, text['Left'], text['Right']))

    def savetext():
        # _, img = cv2.VideoCapture(cam).read()
        OUTPUT_DIR = 'imagestext'
        imgtext = img_
        position = 30
        for key, value in text.items():
            cv2.putText(imgtext, key + ' = ' + str(value), (20, position), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)
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

def get_points(image, n=4, ini_cut=0.9, cut_gap=0.05):
    # 点的坐标比例
    cut_list = []
    left = []
    right = []

    # 如果检测不到，left和right中x坐标的初始值
    ln = 0
    rn = image.shape[1]

    for i in range(n):
        cut_list.append(ini_cut - cut_gap * i)
        left.append([0, 0])
        right.append([0, 0])

    for i, xn in enumerate(cut_list):
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
    left_good_bool_array = left_array[:, 0] != 0
    right_good_bool_array = right_array[:, 0] != WIDTH
    both_good_bool_array = left_good_bool_array * right_good_bool_array
    # left_array = left_array[both_good_bool_array, :]
    # right_array = right_array[both_good_bool_array, :]
    is_left_good = (left_array[:, 0] != 0).all()
    is_right_good = (right_array[:, 0] != WIDTH).all()
    print(both_good_bool_array.sum())
    if both_good_bool_array.sum() > 1:
        print('left and right')  # 两组线
        if (left_array[:, 1] == right_array[:, 1]).all():
            mid_array_x1 = left_array[both_good_bool_array, 0] * 0.5 + right_array[both_good_bool_array,
                                                                                   0] * 0.5  # np.concatenate(, left_array[:, 1]).reshape([4, 2])
            mid_array_x2 = left_array[both_good_bool_array, 1]
            a, b = least_squares(mid_array_x2, mid_array_x1)  # 求直线
            print('a, b:', cut(a), cut(b))
            distance_error = MID_POS - (a * HEIGHT + b)  # 下方交点，横向误差
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

    # print('distance_error:', distance_error)
    # print('angle_error:', angle_error)

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
    text_dict['Steer_Dist'] = cut(KP_DISTANCE * distance_error)
    text_dict['Steer_Ang'] = cut(KP_ANGLE * angle_error)
    return motor, steer, text_dict


def get_control_intuition(left_, right_):
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
    left_max = left_array[:, 0].max()  # 0~WIDTH/2
    right_min = right_array[:, 0].min()  # WIDTH/2~WIDTH

    # 判断两组线/一组线
    left_good_bool_array = left_array[:, 0] != 0
    right_good_bool_array = right_array[:, 0] != WIDTH
    if left_good_bool_array.any():
        left_min = left_array[left_good_bool_array, 0].min()  # 0~WIDTH/2
    else:
        left_min = WIDTH / 2
    if right_good_bool_array.any():
        right_max = right_array[right_good_bool_array, 0].max()  # WIDTH/2~WIDTH
    else:
        right_max = WIDTH / 2 + 1
    both_good_bool_array = left_good_bool_array * right_good_bool_array
    # left_array = left_array[both_good_bool_array, :]
    # right_array = right_array[both_good_bool_array, :]
    is_left_good = (left_array[:, 0] != 0).any()  # all
    is_right_good = (right_array[:, 0] != WIDTH).any()  # all
    left_bad_rate = 1 - left_good_bool_array.sum() / float(left_good_bool_array.shape[0])
    right_bad_rate = 1 - right_good_bool_array.sum() / float(right_good_bool_array.shape[0])
    print(left_good_bool_array)
    print(left_good_bool_array.shape)
    print('left_bad_rate, right_bad_rate', left_bad_rate, right_bad_rate)
    print(both_good_bool_array.sum())
    if right_min - left_max <= 5: # 线条穿过中间竖线
        print("线条穿过中间竖线")
        if left_min < WIDTH - right_max: # 左边线
            motor = MOTOR_MIN
            steer = -STEER_MAX
        else:
            motor = MOTOR_MIN
            steer = STEER_MAX
    elif is_left_good and is_right_good:  # both_good_bool_array.sum() > 1:
        print('left and right')  # 两组线

        motor = MOTOR_MAX  # 两侧车道线都能检测到，则直行
        steer = K_BAD * (left_bad_rate - right_bad_rate)
    elif is_left_good:
        print('left')  # 只有左边线
        # a, b = least_squares(left_array[:, 1], left_array[:, 0])  # 求直线
        # distance_error = - ONE_SIDE_OFFSET
        # angle_error = np.arctan(a)  # in radian
        if left_max < INTUITION_WIDTH:  # 靠近图像边缘，则还可以继续直行
            motor = MOTOR_MIN
            steer = 0
        elif left_max > 0.5 * WIDTH - INTUITION_WIDTH:  # 靠近图像中央，则需急右转弯
            motor = MOTOR_MIN
            steer = -STEER_SHARP
        else:  # 位置适中，则缓慢右转弯
            motor = MOTOR_MIN
            steer = -STEER_MILD
    elif is_right_good:
        print('right')  # 只有右边线
        # a, b = least_squares(right_array[:, 1], right_array[:, 0])  # 求直线
        # distance_error = ONE_SIDE_OFFSET
        # angle_error = np.arctan(a)  # in radian
        if right_min > WIDTH - INTUITION_WIDTH:  # 靠近图像边缘，则还可以继续直行
            motor = MOTOR_MIN
            steer = 0
        elif right_min < 0.5 * WIDTH + INTUITION_WIDTH:  # 靠近图像中央，则需急左转弯
            motor = MOTOR_MIN
            steer = STEER_SHARP
        else:  # 位置适中，则缓慢左转弯
            motor = MOTOR_MIN
            steer = STEER_MILD
    else:
        print('ERROR! no line!')  # 两边都没有
        motor = MOTOR_MIN
        steer = 0
        # distance_error = 0
        # angle_error = 0
        # TODO 维持之前的动作，设置一个keep的bool变量

    # print('distance_error:', distance_error)
    # print('angle_error:', angle_error)

    # motor = MOTOR_MIN  # TODO motor 如何变化
    # steer = KP_DISTANCE * distance_error + KP_ANGLE * angle_error
    # steer = constrain(steer, -STEER_MAX, STEER_MAX)
    steer = np.clip(steer, -STEER_MAX, STEER_MAX)

    text_dict = collections.OrderedDict()
    text_dict['Time'] = time.strftime("%Y-%m-%d %H-%M-%S")
    text_dict['Left'] = left_
    text_dict['Right'] = right_
    # text_dict['Dist_err'] = int(distance_error)
    # text_dict['Ang_err'] = cut(angle_error)
    # text_dict['KP_DISTANCE'] = cut(KP_DISTANCE)
    # text_dict['KP_ANGLE'] = cut(KP_ANGLE)
    text_dict['Motor'] = cut(motor)
    text_dict['Steer'] = cut(steer)
    # text_dict['Steer_Dist'] = cut(KP_DISTANCE * distance_error)
    # text_dict['Steer_Ang'] = cut(KP_ANGLE * angle_error)
    return motor, steer, text_dict


def get_img(camera):
    _, img = camera.read()
    return img


def control(d, motor, steer):
    # global time_before
    d.setStatus(motor=motor, servo=steer)
    # current = time.time()
    print('Time:', datetime.now().strftime('%H:%M:%S.%f')[:-4], 'Motor:', motor, ',Steer: ', steer)
    # time_before = current


def cruise():
    d = driver()
    d.setStatus(mode="speed")
    isfirst = True

    while 1:
        try:
            img = get_img(camera)
            black_line_img = process(img)  # 处理后的图像
            left, right = get_points(black_line_img, n=7, ini_cut=0.95, cut_gap=0.05)  # 0.8
            # print('left:', left, ', right:', right)
            # draw_point_img = draw_points(img, left, right)  # 对原图像画点
            # cv2.imwrite('./output/' + str(idx) + '.jpg', black_line_img)   # 保存处理后的未画点图像
            # cv2.imwrite('./process/' + str(idx) + '.jpg', draw_point_img)  # 保存画点后的原图像
            motor, steer, text_dict = get_control_intuition(left, right)
            # motor=0
            # steer=0
            # text_dict={}
            visualization(img_=img, text=text_dict, doshow=False, dosave=True, dosavetext=True, dovideo1=False,
                          dovideo2=False)
            time.sleep(SLEEP_TIME)  # wait for server to response ctrl signal
            control(d, motor, steer)

        except KeyboardInterrupt:
            break

    d.setStatus(motor=0.0, servo=0.0, dist=0x00, mode="stop")
    d.close()
    del d


def park_control(start_time, pos):
    if pos == 2:
        time0 = PARK_TIME0_2
        time1 = PARK_TIME1_2
        time2 = PARK_TIME2_2
        time3 = PARK_TIME3_2

        park_speed = PARK_SPEED_2
        park_steer = PARK_STEER_2
    elif pos == 3:
        time0 = PARK_TIME0_3
        time1 = PARK_TIME1_3
        time2 = PARK_TIME2_3
        time3 = PARK_TIME3_3

        park_speed = PARK_SPEED_3
        park_steer = PARK_STEER_3
    elif pos == 1:
        time0 = PARK_TIME0_1
        time1 = PARK_TIME1_1
        time2 = PARK_TIME2_1
        time3 = PARK_TIME3_1

        park_speed = PARK_SPEED_1
        park_steer = PARK_STEER_1
    elif pos == 4:
        time0 = PARK_TIME0_4
        time1 = PARK_TIME1_4
        time2 = PARK_TIME2_4
        time3 = PARK_TIME3_4

        park_speed = PARK_SPEED_4
        park_steer = PARK_STEER_4
    state = 'init'
    motor = 0
    steer = 0
    
    current_time = time.time()
    delta_time = current_time - start_time
    if delta_time < time0:
        state = 'init'
    elif delta_time < time0 + time1:
        if pos == 1 or pos == 2:
            state = 'left'
        else:
            state = 'right'
    elif delta_time < time0 + time1 + time2:
        if pos == 1 or pos == 2:
            state = 'right'
        else:
            state = 'left'
    elif delta_time < time0 + time1 + time2 + time3:
        state = 'back'
    else:
        state = 'stop'

    if state == 'init':
        motor = park_speed
        steer = 0
    elif state == 'left':
        motor = park_speed
        steer = park_steer
    elif state == 'right':
        motor = park_speed
        steer = - park_steer
    elif state == 'back':
        motor = park_speed
        steer = 0
    elif state == 'stop':
        motor = 0
        steer = 0

    text_dict = collections.OrderedDict()
    text_dict['Time'] = time.strftime("%Y-%m-%d %H-%M-%S")
    text_dict['Delta_Time'] = delta_time
    text_dict['State'] = state
    text_dict['Motor'] = cut(motor)
    text_dict['Steer'] = cut(steer)

    return motor, steer, text_dict


def park(park_pos):
    d = driver()
    d.setStatus(mode="speed")
    # isfirst = True
    start_time = time.time()

    while 1:
        try:
            img = get_img(camera)
            # black_line_img = process(img)  # 处理后的图像
            # left, right = get_points(black_line_img, n=7, ini_cut=0.8, cut_gap=0.05)
            # print('left:', left, ', right:', right)
            # draw_point_img = draw_points(img, left, right)  # 对原图像画点
            # cv2.imwrite('./output/' + str(idx) + '.jpg', black_line_img)   # 保存处理后的未画点图像
            # cv2.imwrite('./process/' + str(idx) + '.jpg', draw_point_img)  # 保存画点后的原图像
            # motor, steer, text_dict = get_control(left, right)
            motor, steer, text_dict = park_control(start_time=start_time, pos=park_pos)
            # motor=0
            # steer=0
            # text_dict={}
            # visualization(img_=img, text=text_dict, doshow=False, dosave=True, dosavetext=True, dovideo1=False,
            #               dovideo2=False)
            time.sleep(PARK_SLEEP_TIME)  # wait for server to response ctrl signal
            control(d, motor, steer)

        except KeyboardInterrupt:
            break

    d.setStatus(motor=0.0, servo=0.0, dist=0x00, mode="stop")
    d.close()
    del d


if __name__ == '__main__':
    #cruise()
    park(park_pos=PARK_POS)
