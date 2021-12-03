import time

import numpy as np
import pandas as pd

from pyntcloud import PyntCloud

from kinect_retro import InfraredSource
import cv2

import pcl
import pcl.pcl_visualization

s = InfraredSource()

cv2.namedWindow('depth')
cv2.namedWindow('color')
go = True

first = None
frame_count = 0
while go:
    if not s.get_new_frames_available():
        time.sleep(1/30)
        continue
    frame_depth = s.get_depth()
    infra = s.get_infrared()
    if first is None:
        first = time.monotonic()
    frame_count += 1
    color_frame = s.get_color(hsv=False)
    print(color_frame.dtype)
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

    cv2.imshow("depth", frame_depth)
    cv2.imshow("color", color_frame)
    d3_point = s.get_points_color(frame_depth)
    d3_point.resize((1080*1920, 3))
    color_frame = color_frame.reshape((1080*1920, 3)) *0.95
    color_frame = color_frame.astype(np.uint8)
    print('color frame', np.min(color_frame), np.max(color_frame), color_frame.shape)
    red = color_frame[:, 0].astype(np.uint32)
    green = color_frame[:, 1].astype(np.uint32)
    blue = color_frame[:, 2].astype(np.uint32)
    # alpha = color_frame[:, 3].astype(np.uint32)
    print('blue dtype', blue.dtype, blue)
    rgb = (np.left_shift(red, 16)  # red
          + np.left_shift(green, 8)  # green
          + np.left_shift(blue, 0))  #blue
    # rgb.dtype = np.float32
    print(rgb.shape)
    print(np.min(d3_point), np.max(d3_point))
    points = np.hstack((d3_point, rgb[:, None].astype(np.float32))).astype(np.float32)

    print(points.shape)
    print(np.isfinite(points).sum(axis=0))
    points = points[np.isfinite(points).all(axis=1)]
    print(points.shape)
    cloud = pcl.PointCloud_PointXYZRGBA(points)
    cloud.from_array(points)
    pcl.save(cloud, "./cloud/output{}.ply".format(time.monotonic()), binary=True)

    # visual = pcl.pcl_visualization.CloudViewing()
    # visual.ShowColorACloud(cloud, b'cloud')
    # flag = True
    # while flag:
    #     flag = not visual.WasStopped()

    key = cv2.waitKey(1)
    if key == 113:
        fps = frame_count/(time.monotonic() - first)
        print(fps)
        break





