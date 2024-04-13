import cv2
import pyrealsense2 as rs
import numpy as np
from datetime import datetime
import open3d as o3d


WIDTH = 640
HEIGHT = 320

ctx = rs.context()
device_serial = [device.get_info(rs.camera_info.serial_number) for device in ctx.devices if device.get_info(rs.camera_info.name).lower() != "platform camera"][0]

conf = rs.confing()
conf.enable_device(device_serial)
conf.enable_stream(rs.stream.color,WIDTH,HEIGHT,rs.format.rgb8,30)
conf.enable_stream(rs.stream.depth,WIDTH,HEIGHT,rs.format.z16,30)

pipe = rs.pipeline()
pipe.start(conf)
align = rs.align(rs.stream_color)
frames = pipe.wait_for_frames()
frames = align.process(frames)
intrinsics = frames.get_profile().as_video_stream_profile().get_intrinsics()
color_image = np.asanyarray(frames.get_color_frame().get_data())
depth_image = np.asanyarray(frames.get_depth_frame().get_data())
#camera_matrix = np.array(([intrinsics.fx,0,intrinsics.ppx],
#                          [0,intrinsics.fy,intrinsics.ppy],
#                          [0,0,1]))
#dist_coeffs = intrinsics.coeffs
depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH,HEIGHT,intrinsics.fx,intrinsics.fy,intrinsics.ppx,intrinsics.ppy)
depth_color_image = o3d.geometry.Image(color_image)
depth_depth_image = o3d.geometry.Image(depth_image)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(depth_color_image,depth_depth_image,convert_rgb_to_intensity = False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,depth_intrinsic)
o3d.visualization.draw_geometries([pcd])