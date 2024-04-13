# REALSENSEの内部パラメータとシリアル番号を取得します
# 1つずつ挿して確認してね

import pyrealsense2 as rs
import numpy as np
import os
import datetime

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

dt_now = datetime.datetime.now()

DATE = [str(dt_now.month).zfill(2) + "_"
        + str(dt_now.day).zfill(2) + "_"
        + str(dt_now.hour).zfill(2) + "_"
        + str(dt_now.minute).zfill(2)][0]

# カメラの情報を取得
pipeline_profile = pipeline.get_active_profile()
device = pipeline_profile.get_device()
serial_number = device.get_info(rs.camera_info.serial_number)

print(f"RealSenseカメラのシリアル番号: {serial_number}")

print("これは左から(カメラの撮る方向を基準に)何番目のカメラですか？")
n = input()
print(f"your input: {n}")

path = f'Serial'
if not os.path.isdir(path):
    os.makedirs(path)

with open(f'Serial/{n}.txt', 'w') as f:
    f.write(serial_number)

print("内部パラメータを取得します")

# ストリーム(Color/Depth)の設定
config = rs.config()

camera_profile = rs.video_stream_profile(pipeline_profile.get_stream(rs.stream.color))

intrinsics = camera_profile.get_intrinsics()

camera_matrix = np.array([[intrinsics.fx,               0,      intrinsics.ppx],
                          [0,               intrinsics.fy,      intrinsics.ppy],
                          [0,                           0,                   1]])

print(camera_matrix)

path = f'Inter'
if not os.path.isdir(path):
    os.makedirs(path)

print(f"保存ファイル名 : matrix{n}_{DATE[:8]})")
np.save(f"Inter/matrix{n}_{DATE[:8]}.npy", camera_matrix)

# RealSenseカメラをシャットダウン
pipeline.stop()