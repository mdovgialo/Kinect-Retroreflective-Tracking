import time
import numpy as np

from kinect_retro import InfraredSource
import cv2
import pylab as pb

from kinect_retro.tracker import Tracker
from pyopenvrinputemu import VRInputSystem

s = InfraredSource()

# cv2.namedWindow('Infrared')
# cv2.namedWindow('thresholded')

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


def contours_to_masses(contours, depth_frame, source:InfraredSource, color_frame):
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
                color_blob = color_frame[center_y-1:center_y+1, center_x-1:center_x+1]
                color = np.mean(np.mean(color_blob, axis=0), axis=0)
                if not np.isfinite(color).all():
                    continue
                masses.append([center, color])
            except IndexError:
                pass
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

def visualise_joints(joints):

    img_top_down = np.zeros((480, 640,3 ), dtype=np.uint8)
    img_front = np.zeros((480, 640,3 ), dtype=np.uint8)
    for nr, joint in enumerate(joints):
        mass_x = joint.x
        mass_y = joint.y
        mass_z = joint.z

        color = np.array([joint.color_hue, 240, 255], dtype=np.uint8).reshape(1,1,3)
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
        cv2.imshow("joints_front", img_front)
        cv2.imshow("joints_top_down", img_top_down)


kernel = np.ones((5,5),np.uint16)
kernel_depth = np.ones((9,9),np.uint16)


kinect_location = [-0.017777538299560483, -0.9032255387306214, -2.0564754199981845]
kinect_rotation = [0.5578027367591858, 0.26856940865516643, -0.3000000000000001]

vrinput = VRInputSystem(global_offset=kinect_location, global_rotation=kinect_rotation)
l_leg_tracker = vrinput.add_tracker('l_leg')
r_leg_tracker = vrinput.add_tracker('r_leg')
hip_tracker = vrinput.add_tracker('hip')



l_leg = Tracker([[105, 125]], l_leg_tracker, color_hue=115)
r_leg = Tracker([[75, 90]], r_leg_tracker, color_hue=90)
hips = Tracker([[0, 30]], hip_tracker, color_hue=18)

joints = [l_leg, r_leg, hips]



while go:
    if not s.get_new_frames_available():
        time.sleep(1/30)
        continue
    frame_raw = s.get_infrared()
    frame_depth = s.get_depth()
    color_frame = s.get_color(hsv=True)

    frame = (frame_raw.astype(float))
    blured = cv2.medianBlur(frame_raw, 3)
    frame_thr = (frame.astype(float)>thr).astype(np.uint8)

    dilation = cv2.dilate(frame_thr, kernel, iterations=10)

    frame_depth[frame_depth>0] = 2**16 - frame_depth[frame_depth>0]

    frame_depth = cv2.medianBlur(frame_depth, 3)
    blured_depth = cv2.dilate(frame_depth, kernel_depth, iterations=1)

    blured_depth[blured_depth > 0] = 2 ** 16 - blured_depth[blured_depth > 0]

    fgmask = fgbg.apply((blured/2**16*255).astype(np.uint8))

    im2, contours, hierarchy = cv2.findContours(frame_thr * fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    masses = contours_to_masses(contours, blured_depth, s, color_frame)

    for joint in joints:
        joint.update_from_masses(masses)

    # visualise_masses(masses)
    visualise_joints(joints)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        vrinput.global_rotation[0] += 0.01
    if key & 0xFF == ord('a'):
        vrinput.global_rotation[0] -= 0.01

    if key & 0xFF == ord('w'):
        vrinput.global_rotation[1] += 0.01

    if key & 0xFF == ord('s'):
        vrinput.global_rotation[1] -= 0.01

    if key & 0xFF == ord('e'):
        vrinput.global_rotation[2] += 0.01

    if key & 0xFF == ord('d'):
        vrinput.global_rotation[2] -= 0.01



    if key & 0xFF == ord('r'):
        vrinput.global_offset[0] += 0.01
    if key & 0xFF == ord('f'):
        vrinput.global_offset[0] -= 0.01

    if key & 0xFF == ord('t'):
        vrinput.global_offset[1] += 0.01

    if key & 0xFF == ord('g'):
        vrinput.global_offset[1] -= 0.01

    if key & 0xFF == ord('y'):
        vrinput.global_offset[2] += 0.01
    if key & 0xFF == ord('h'):
        vrinput.global_offset[2] -= 0.01
    # print(vrinput.global_offset, vrinput.global_rotation)


    cv2.imshow("depth", blured_depth)
frame = s.get_infrared()
pb.hist(frame.flatten(), bins=100)
pb.show()
