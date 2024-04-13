# ２つのカメラで外部パラメータの取得＆視差画像の作成＆点群を作成します

# importたち
import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
from sklearn.preprocessing import MinMaxScaler

# openCVのバージョン確認
print(cv2.__version__)

# チェスボードのマス目を指定する（縦x横）
CHECKERBOARD = (7,10)

# 僕は日付管理してるので，ここは自由に
print("欲しい写真の日付と時刻を入力してください。 ex)10_28_12_12")
DATE = input()
print(f"your input: {DATE}")

print("三次元復元したい写真は何番目の写真ですか ex)0")
tg = input()
print(f"your input: {tg}")

# 座標をしまっておく箱たち
obj_points = []
img_points1 = []
img_points2 = []

# objpはチェスボードのワールド座標を入れておくやつ．
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# ちゃんと実際のメートル単位にしましょう(これは黒の幅が2.2cmの場合)
objp = objp * 0.022

# count_sumImg : チェスボードを認識できた画像の枚数を保存しておきたい
count_sumImg = 0

# 撮った画像の枚数だけループ
# 一枚一枚撮影したチェスボードの画像座標を取得するためのループ
p1_c_paths = sorted(glob.glob(f"RGB/{DATE}/{DATE}_*_1.jpg"))
p2_c_paths = sorted(glob.glob(f"RGB/{DATE}/{DATE}_*_2.jpg"))
for p1_c_path, p2_c_path in zip(p1_c_paths, p2_c_paths):
    p1_c = cv2.imread(p1_c_path)
    p2_c = cv2.imread(p2_c_path)
    # グレースケールにします
    p1 = cv2.cvtColor(p1_c,cv2.COLOR_BGR2GRAY)
    p2 = cv2.cvtColor(p2_c,cv2.COLOR_BGR2GRAY)

    # チェスボードの画像座標を取得
    ret_1, corners_1 = cv2.findChessboardCorners(p1, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_2, corners_2 = cv2.findChessboardCorners(p2, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    # もし2枚ともチェスボードを認識できていたら
    if ret_1 and ret_2:
        count_sumImg += 1

        # なんだろうこれ
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 取得した画像座標を高精度化します
        corners2_1 = cv2.cornerSubPix(p1,corners_1,(11,11),(-1,-1),criteria)
        corners2_2 = cv2.cornerSubPix(p2,corners_2,(11,11),(-1,-1),criteria)

        # 画像にコーナーを描写します
        leftCheck = cv2.drawChessboardCorners(p1_c, CHECKERBOARD, corners2_1, ret_1)
        righCheck = cv2.drawChessboardCorners(p2_c, CHECKERBOARD, corners2_2, ret_2)

        # plt.subplot(1,2,1)
        # plt.imshow(cv2.cvtColor(leftCheck, cv2.COLOR_BGR2RGB))
        # plt.subplot(1,2,2)
        # plt.imshow(cv2.cvtColor(righCheck, cv2.COLOR_BGR2RGB))
        # plt.show()

        # チェスボードのワールド座標とカメラ座標をリストに格納しておこう
        obj_points.append(objp)
        img_points1.append(corners2_1)
        img_points2.append(corners2_2)

print(f"有効なキャリブレーション画像枚数 : {count_sumImg}")

# 画像の縦横ピクセル数
h,w = p1.shape

# それぞれのカメラの内部パラメータを読み込み
cameraMatrix1 = np.load(f"Inter/matrix1_{DATE[:8]}.npy")
cameraMatrix2 = np.load(f"Inter/matrix2_{DATE[:8]}.npy")

# 歪み係数は0とします．
distCoeffs = np.array([[0., 0., 0., 0., 0.]])

distCoeffs1 = distCoeffs.copy()
distCoeffs2 = distCoeffs.copy()

imageSize = (w,h)

# 外部パラメータゲット！(RとT)
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points1, img_points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize)

flags = 0
alpha = 1
newimageSize = (w,h)

# 説明するのだるくなってきた．．．．．平行化するためのものです
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, flags, alpha, newimageSize)

m1type = cv2.CV_32FC1
map1_l, map2_l = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, newimageSize, m1type) #m1type省略不可
map1_r, map2_r = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, newimageSize, m1type)

interpolation = cv2.INTER_NEAREST # INTER_RINEARはなぜか使えない


# ここでは，三次元復元したい画像を入れてください
tgtImg_l = cv2.imread(f"RGBTg/{DATE}/{DATE}_tg_{tg}_1.jpg").astype(np.uint8)
tgtImg_r = cv2.imread(f"RGBTg/{DATE}/{DATE}_tg_{tg}_2.jpg").astype(np.uint8)

# 画像を平行化してます
Re_TgtImg_l_0 = cv2.remap(tgtImg_l, map1_l, map2_l, interpolation) #interpolation省略不可
Re_TgtImg_r_0 = cv2.remap(tgtImg_r, map1_r, map2_r, interpolation)

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(Re_TgtImg_l_0, cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(Re_TgtImg_r_0, cv2.COLOR_BGR2RGB))
plt.show()

# ノイズを除去してるけど，やらなくてもいいよ
Re_TgtImg_l=cv2.GaussianBlur(Re_TgtImg_l_0, (3, 3), -1)
Re_TgtImg_r=cv2.GaussianBlur(Re_TgtImg_r_0, (3, 3), -1)

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(Re_TgtImg_l, cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(Re_TgtImg_r, cv2.COLOR_BGR2RGB))
plt.show()

# こっからは視差画像作成------------
window_size = 3
min_disp = 0
num_disp = 16*8 - min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,  # 視差の下限
    numDisparities=num_disp,  # 最大の上限
    blockSize=window_size,  # SADの窓サイズ
    uniquenessRatio=10,  # パーセント単位で表現されるマージン
    speckleWindowSize=50,  # 視差領域の最大サイズ
    speckleRange=1,  # それぞれの連結成分における最大視差値
    disp12MaxDiff=32,  # left-right 視差チェックにおけて許容される最大の差
    P1=8 * 3 * window_size ** 2,  # 視差のなめらかさを制御するパラメータ1
    P2=32 * 3 * window_size ** 2,  # 視差のなめらかさを制御するパラメータ2
    mode=cv2.StereoSGBM_MODE_HH)

disp2 = np.array([])
disp = np.array([])

def setCanvas(tk_image):
     canvas.create_image(1920 / 4, 1080 / 4, image=tk_image)

def stereoCompute():
     global tk_image
     global disp, disp2
     disp = stereo.compute(Re_TgtImg_l, Re_TgtImg_r).astype(np.float32) / 16.0
     disp2 = (disp-min_disp)/num_disp
     disp2 = np.clip(disp2 * 255, a_min = 0, a_max = 255).astype(np.uint8)

     scaler = MinMaxScaler((0,255))
     disp2 = scaler.fit_transform(disp2).astype(np.uint8)

     pil_image = Image.fromarray(disp2)
     tk_image = ImageTk.PhotoImage(image=pil_image)
     setCanvas(tk_image)

def setMinDisparity(str):
     value = int(str)
     stereo.setMinDisparity(value)
     stereoCompute()

def setNumDisparities(str):
     value = int(str)
     stereo.setNumDisparities(value)
     stereoCompute()

def setBlockSize(str):
     value = int(str)
     stereo.setBlockSize(value)
     stereoCompute()

def setUniquenessRatio(str):
     value = int(str)
     stereo.setUniquenessRatio(value)
     stereoCompute()

def setSpeckleWindowSize(str):
     value = int(str)
     stereo.setSpeckleWindowSize(value)
     stereoCompute()

def setSpeckleRange(str):
     value = int(str)
     stereo.setSpeckleRange(value)
     stereoCompute()

def setDisp12MaxDiff(str):
     value = int(str)
     stereo.setDisp12MaxDiff(value)
     stereoCompute()

def setP1(str):
     value = int(str)
     winSize = blockSize_sc.get()
     stereo.setP1(value * 3 * winSize ** 2)
     stereoCompute()

def setP2(str):
     value = int(str)
     winSize = blockSize_sc.get()
     stereo.setP2(value * 3 * winSize ** 2)
     stereoCompute()


root = tk.Tk()
root.title("DisparityMap")
root.geometry("1920x1080")

frame_code = tk.Frame(root)
frame_code.pack(side=tk.LEFT)

frame_canvas = tk.Frame(root)
frame_canvas.pack(side=tk.RIGHT)

#--------------------------------------------#
canvas = tk.Canvas(frame_canvas, width=1920 / 2, height=1080 / 2)
canvas.pack()

minDisparity_sc = tk.Scale(
     frame_code,
     from_=0,
     resolution=16,
     to_=16,
     label="minDisparity",
     command=setMinDisparity,
     length=1000,
     orient=tk.HORIZONTAL
)
minDisparity_sc.pack()
minDisparity_sc.set(min_disp)

numDisparities_sc = tk.Scale(
     frame_code,
     from_=16,
     resolution=16,
     to_=16*20,
     label="numDisparities",
     command=setNumDisparities,
     length=1000,
     orient=tk.HORIZONTAL
)
numDisparities_sc.pack()
numDisparities_sc.set(num_disp)

blockSize_sc = tk.Scale(
     frame_code,
     from_=3,
     resolution=1,
     to_=11,
     label="blockSize",
     command=setBlockSize,
     length=1000,
     orient=tk.HORIZONTAL
)
blockSize_sc.pack()
blockSize_sc.set(window_size)

uniquenessRatio_sc = tk.Scale(
     frame_code,
     from_=5,
     to_=15,
     label="uniquenessRatio",
     command=setUniquenessRatio,
     length=1000,
     orient=tk.HORIZONTAL
)
uniquenessRatio_sc.pack()
uniquenessRatio_sc.set(10)

speckleWindowSize_sc = tk.Scale(
     frame_code,
     from_=50,
     to_=200,
     label="speckleWindowSize",
     command=setSpeckleWindowSize,
     length=1000,
     orient=tk.HORIZONTAL
)
speckleWindowSize_sc.pack()
speckleWindowSize_sc.set(50)

speckleRange_sc = tk.Scale(
     frame_code,
     from_=1,
     to_=2,
     label="speckleRange",
     command=setSpeckleRange,
     length=1000,
     orient=tk.HORIZONTAL
)
speckleRange_sc.pack()
speckleRange_sc.set(1)

disp12MaxDiff_sc = tk.Scale(
     frame_code,
     from_=0,
     to_=32,
     label="disp12MaxDiff",
     command=setDisp12MaxDiff,
     length=1000,
     orient=tk.HORIZONTAL
)
disp12MaxDiff_sc.pack()
disp12MaxDiff_sc.set(32)

P1_sc = tk.Scale(
     frame_code,
     from_=8,
     to_=128,
     label="P1",
     command=setP1,
     length=1000,
     orient=tk.HORIZONTAL
)
P1_sc.pack()
P1_sc.set(8 * 3 * window_size ** 2)

P2_sc = tk.Scale(
     frame_code,
     from_=8,
     to_=128,
     label="P2",
     command=setP2,
     length=1000,
     orient=tk.HORIZONTAL
)
P2_sc.pack()
P2_sc.set(32 * 3 * window_size ** 2)

stereoCompute()

#---------------------------------------------#
root.mainloop()







# 視差画像滑らかにしてる
# 平滑化
disp3 = cv2.medianBlur(disp, ksize=5)
disp4 = (disp3-min_disp)/num_disp
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(disp2, cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(disp4, cv2.COLOR_BGR2RGB))
plt.show()

disp = disp3.copy()

points = cv2.reprojectImageTo3D(disp, Q)
colors = cv2.cvtColor(Re_TgtImg_l_0,cv2.COLOR_BGR2RGB)

mask = disp > min_disp
out_points = points[mask]
out_colors = colors[mask]

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

# ゲットした点群を保存
verts = out_points.reshape(-1, 3)
colors = out_colors.reshape(-1, 3)
verts = np.hstack([verts, colors])

path = f"ply"
if not os.path.isdir(path):
    os.makedirs(path)

with open(f"ply/{DATE}_{tg}.ply", 'w') as f:
	f.write(ply_header % dict(vert_num=len(verts)))
	np.savetxt(f, verts,"%f %f %f %d %d %d")