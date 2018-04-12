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

def min_thr(x):
    global thr
    thr = x

cv2.createTrackbar("Min", "Infrared",0,2**16, min_thr)

fgbg = cv2.createBackgroundSubtractorKNN()

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

def visualise_masses(masses):

    img_top_down = np.zeros((480, 640,3 ), dtype=np.uint8)
    img_front = np.zeros((480, 640,3 ), dtype=np.uint8)

    for nr, mass in enumerate(masses):
        print("BLOB", nr, mass)
        mass_x = mass[0]
        mass_y = mass[1]
        mass_z = mass[2]

        mass_x2 = int(mass_x/1.5*640/2+640/2)
        mass_y2 =  int(-mass_y/1.5*480/2+480/2)
        mass_z2 =  int(mass_z/1.5*480/2+480/2)
        print(mass_x2, mass_y2, mass_z2)
        cv2.circle(img_top_down, (mass_x2, mass_z2), 3, (0,0,255), -1)
        cv2.circle(img_front, (mass_x2, mass_y2), 3, (0,0,255), -1)
    if len(masses):
        cv2.imshow("positions_front", img_front)
        cv2.imshow("positions_top_down", img_top_down)


kernel = np.ones((5,5),np.uint8)

while go:
    frame_raw = s.get_infrared()
    frame_depth = s.get_depth()
    frame = (frame_raw.astype(float))
    frame_color = np.concatenate([frame[:, :, None]]*3, axis=2)/2**16
    blured = cv2.medianBlur(frame_raw, 3)
    frame_thr = (blured.astype(float)>thr).astype(np.uint8)

    # dilation = cv2.dilate(frame_thr, kernel, iterations=10)

    blured_depth = frame_depth
    # blured_depth = cv2.medianBlur(frame_depth, 5)

    fgmask = fgbg.apply((blured/2**16*255).astype(np.uint8))

    frame_thr = fgmask * frame_thr
    cv2.imshow("fgmask", fgmask)

    # im2, contours, hierarchy = cv2.findContours(frame_thr, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2, contours, hierarchy = cv2.findContours(frame_thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    contours = filter_contours(contours, fgmask)
    # print(contours)
    # print(len(contours))
    points = s.get_points(blured_depth)
    points_flat = points.reshape((-1, 3))

    points_flat[np.logical_not(np.isfinite(points_flat))] = 0
    points_flat[points_flat>3] = 3
    points_flat[points_flat<-3] = -3
    points_flat -= np.mean(points_flat, axis=0)

    contours_to_masses(contours, points)
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

