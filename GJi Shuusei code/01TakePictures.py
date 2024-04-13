# キャリブレーション用の写真と三次元復元用の写真を撮ります
# マウスの右クリックで写真撮影．

# 1回目の写真撮影→Escボタン押→2回目の写真撮影→Escボタン押→終了

# 1回目はキャリブレーション用の写真を40枚くらい撮るといいよ
# Escを押したら2回目の三次元復元用の写真撮影をしてね

import pyrealsense2 as rs
import numpy as np
import os
import datetime
import cv2

# ストリーム(Color/Depth)の設定
config1 = rs.config()
config2 = rs.config()

# realsenseのシリアルナンバーを読み込む
with open("Serial/1.txt") as f:
    serial1 = f.read()

with open("Serial/2.txt") as f:
    serial2 = f.read()

config1.enable_device(serial1)
config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

config2.enable_device(serial2)
config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ストリーミング開始
pipeline1 = rs.pipeline()
pipeline2 = rs.pipeline()

profile1 = pipeline1.start(config1)
profile2 = pipeline2.start(config2)

# 時刻でディレクトリを作成
dt_now = datetime.datetime.now()

DATE = [str(dt_now.month).zfill(2) + "_"
        + str(dt_now.day).zfill(2) + "_"
        + str(dt_now.hour).zfill(2) + "_"
        + str(dt_now.minute).zfill(2)][0]

path = f"RGB"
if not os.path.isdir(path):
    os.makedirs(path)

path = f"RGB/{DATE}"
if not os.path.isdir(path):
    os.makedirs(path)

path = f"RGBTg"
if not os.path.isdir(path):
    os.makedirs(path)

path = f"RGBTg/{DATE}"
if not os.path.isdir(path):
    os.makedirs(path)

# 写真を入れておく箱
RGB_image1_s = np.array([])
RGB_image2_s = np.array([])

#写真保存カウント
counts = 0

def mouse_event(event, x, y, flags, param):
    global counts

    try:
        if event == cv2.EVENT_RBUTTONDOWN:
            path1 = f"RGB/{DATE}/{DATE}_{counts}_1.jpg"
            path2 = f"RGB/{DATE}/{DATE}_{counts}_2.jpg"

            cv2.imwrite(path1, RGB_image1_s)
            cv2.imwrite(path2, RGB_image2_s)

            print("The 1 RGB image was created as " + path1)
            print("The 2 RGB image was created as " + path2)
            
            counts += 1

    except Exception as e:
        print(e)

def mouse_event2(event, x, y, flags, param):
    global counts

    try:
        if event == cv2.EVENT_RBUTTONDOWN:
            path1 = f"RGBTg/{DATE}/{DATE}_tg_{counts}_1.jpg"
            path2 = f"RGBTg/{DATE}/{DATE}_tg_{counts}_2.jpg"

            cv2.imwrite(path1, RGB_image1_s)
            cv2.imwrite(path2, RGB_image2_s)

            print("The 1 RGB image was created as " + path1)
            print("The 2 RGB image was created as " + path2)

            counts += 1

    except Exception as e:
        print(e)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    # キャリブレーション用写真撮影
    while True:
        # フレーム待ち
        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()

        #RGB
        RGB_frame1 = frames1.get_color_frame()
        RGB_image1 = np.asanyarray(RGB_frame1.get_data())

        RGB_frame2 = frames2.get_color_frame()
        RGB_image2 = np.asanyarray(RGB_frame2.get_data())

        # 表示
        RGB_image1_s = cv2.resize(RGB_image1, (640, 480))
        RGB_image2_s = cv2.resize(RGB_image2, (640, 480))
            
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('RealSense', mouse_event)

        flipRGBImg1 = cv2.flip(cv2.resize(RGB_image1_s, None, None, 0.5, 0.5), 1)
        flipRGBImg2 = cv2.flip(cv2.resize(RGB_image2_s, None, None, 0.5, 0.5), 1)

        conImg1 = np.hstack((flipRGBImg2, flipRGBImg1))

        cv2.imshow('RealSense', conImg1)
        if cv2.waitKey(1) & 0xff == 27:#ESCで終了
            cv2.destroyAllWindows()
            counts=0
            break

    # 三次元復元用写真撮影
    while True:
        # フレーム待ち
        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()

        #RGB
        RGB_frame1 = frames1.get_color_frame()
        RGB_image1 = np.asanyarray(RGB_frame1.get_data())

        RGB_frame2 = frames2.get_color_frame()
        RGB_image2 = np.asanyarray(RGB_frame2.get_data())

        # 表示
        RGB_image1_s = cv2.resize(RGB_image1, (640, 480))
        RGB_image2_s = cv2.resize(RGB_image2, (640, 480))
            
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('RealSense', mouse_event2)
   
        flipRGBImg1 = cv2.flip(cv2.resize(RGB_image1_s, None, None, 0.5, 0.5), 1)
        flipRGBImg2 = cv2.flip(cv2.resize(RGB_image2_s, None, None, 0.5, 0.5), 1)

        conImg1 = np.hstack((flipRGBImg2, flipRGBImg1))
    
        cv2.imshow('RealSense', conImg1)
        if cv2.waitKey(1) & 0xff == 27:#ESCで終了
            cv2.destroyAllWindows()
            break

finally:
    # ストリーミング停止
    pipeline1.stop()
    pipeline2.stop()
    print("Finish")