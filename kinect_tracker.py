import time
import numpy as np

from kinect_retro import InfraredSource
import cv2
import pylab as pb
s = InfraredSource()

cv2.namedWindow('Infrared')
cv2.namedWindow('thresholded')

thr = 25000
max_thr = 2**16
area = 20
min_distance_squared = 0.3**2
def min_thr(x):
    global thr
    thr = x

cv2.createTrackbar("Min", "Infrared",0,2**16, min_thr)

fgbg = cv2.createBackgroundSubtractorKNN()
# fgbg = cv2.createBackgroundSubtractorMOG2()

go = True

class IRPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def filter_contours(contours, fgmask):
    accepted = []
    for contour in contours:
        if np.any(fgmask.T[contour]):
            accepted.append(contour)
    return accepted

def contours_to_masses(contours, points):
    masses = []
    for contour in contours:
        # import IPython
        # IPython.embed()
        center = np.mean(points[contour[:, 0, 1], contour[:, 0, 0], :], axis=0)
        masses.append(center)
    visualise_masses(masses)
    return masses

def contours_to_masses2(contours, depth_frame, source:InfraredSource, color_frame):
    masses = []
    contour_centers = []
    for contour in contours:
        x = np.mean(contour[:, 0, 1],axis=0)
        y = np.mean(contour[:, 0, 0],axis=0)
        contour_centers.append([x, y])

    for x,y in contour_centers:
        depth = depth_frame[int(x), int(y)]
        point = source.get_point_location(y, x, depth)
        center = np.array([point.x, point.y, point.z])
        color_center = s.get_depth_point_in_rgb(y, x, depth)

        if np.all(np.isfinite(center)):
            try:
                center_y = int(color_center.y)
                center_x = int(color_center.x)
                color_blob = color_frame[center_y-3:center_y+3, center_x-3:center_x+3]
                color = np.mean(np.mean(color_blob, axis=0), axis=0)
                if not np.isfinite(color).all():
                    continue
                masses.append([center, color])
            except IndexError:
                pass
    visualise_masses(masses)
    return masses

def visualise_masses(masses):

    img_top_down = np.zeros((480, 640,3 ), dtype=np.uint8)
    img_front = np.zeros((480, 640,3 ), dtype=np.uint8)
    for nr, mass in enumerate(masses):
        center, color = mass
        mass_x = center[0]
        mass_y = center[1]
        mass_z = center[2]

        color = np.array([color[0], 240, 255], dtype=np.uint8).reshape(1,1,3)
        color_rgb = cv2.cvtColor(color, cv2.COLOR_HSV2RGB).flatten()
        color_rgb = (np.asscalar(color_rgb[0]), np.asscalar(color_rgb[1]), np.asscalar(color_rgb[2]))
        # print(color_rgb)

        mass_x2 = int(mass_x/1.5*640/2+640/2)
        mass_y2 = int(-mass_y/1.5*480/2+480/2)
        mass_z2 = int(mass_z/6*480)
        cv2.circle(img_top_down, (mass_x2, mass_z2, ), 10, color_rgb, -1)
        cv2.circle(img_front, (mass_x2, mass_y2, ), 10, color_rgb, -1)
    if len(masses):
        cv2.circle(img_front, (int(640/2), int(480/2),), 23, (255, 0, 0), -1)
        cv2.circle(img_top_down, (int(640/2), 0,), 23, (255, 0, 0), -1)
        cv2.imshow("positions_front", img_front)
        cv2.imshow("positions_top_down", img_top_down)


kernel = np.ones((5,5),np.uint16)
kernel_depth = np.ones((9,9),np.uint16)

while go:
    frame_raw = s.get_infrared()
    frame_depth = s.get_depth()
    color_frame = s.get_color(hsv=True)

    color_frame_color = color_frame.copy()
    color_frame_color[:, :, 1] = 255
    color_frame_color[:, :, 2] = 255
    cv2.imshow('COLOR_FRAME', cv2.cvtColor(color_frame_color, cv2.COLOR_HSV2RGB))

    frame = (frame_raw.astype(float))
    frame_color = np.concatenate([frame[:, :, None]]*3, axis=2)/2**16
    blured = cv2.medianBlur(frame_raw, 3)
    frame_thr = (frame.astype(float)>thr).astype(np.uint8)

    dilation = cv2.dilate(frame_thr, kernel, iterations=10)
    # print(frame_depth.dtype)


    frame_depth[frame_depth>0] = 2**16 - frame_depth[frame_depth>0]

    frame_depth = cv2.medianBlur(frame_depth, 3)
    blured_depth = cv2.dilate(frame_depth, kernel_depth, iterations=1)

    blured_depth[blured_depth > 0] = 2 ** 16 - blured_depth[blured_depth > 0]
    # blured_depth = frame_depth
    # blured_depth = cv2.medianBlur(frame_depth, 5)

    fgmask = fgbg.apply((blured/2**16*255).astype(np.uint8))

    frame_thr = fgmask * frame_thr
    cv2.imshow("fgmask", fgmask)

    # im2, contours, hierarchy = cv2.findContours(frame_thr, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2, contours, hierarchy = cv2.findContours(frame_thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    contours = filter_contours(contours, fgmask)
    # print(contours)
    # print(len(contours))
    # points = s.get_points(blured_depth)
    # points_flat = points.reshape((-1, 3))
    #
    # points_flat[np.logical_not(np.isfinite(points_flat))] = 0
    # points_flat[points_flat>3] = 3
    # points_flat[points_flat<-3] = -3
    # points_flat -= np.mean(points_flat, axis=0)

    # contours_to_masses(contours, points)
    contours_to_masses2(contours, blured_depth, s, color_frame)
    cv2.drawContours(frame_color, contours, -1, (0, 255, 0), 1)

    blured_depth_color = np.concatenate([blured_depth.astype(float)[:, :, None]]*3, axis=2)/2**16

    cv2.drawContours(blured_depth_color, contours, -1, (0, 255, 0), 1)


    cv2.imshow("Infrared", frame_color)
    cv2.imshow("Depth", blured_depth_color*10)
    # cv2.imshow("Infrared", frame)
    cv2.imshow("thresholded", frame_thr.astype(float))
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    time.sleep(1/30)
frame = s.get_infrared()
pb.hist(frame.flatten(), bins=100)
pb.show()
