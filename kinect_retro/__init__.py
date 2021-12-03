import ctypes

import numpy as np
from pykinect2 import PyKinectV2, PyKinectRuntime
import cv2
from pykinect2.PyKinectV2 import _DepthSpacePoint


class InfraredSource:
    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
        self.infrared_width = self._kinect.infrared_frame_desc.Width
        self.infrared_height = self._kinect.infrared_frame_desc.Height
        self.color_width = self._kinect.color_frame_desc.Width
        self.color_height = self._kinect.color_frame_desc.Height

    def get_new_frames_available(self):
        return self._kinect.has_new_color_frame() and self._kinect.has_new_depth_frame() and self._kinect.has_new_infrared_frame()

    def get_infrared(self):
        return self._kinect.get_last_infrared_frame().reshape(self.infrared_height, self.infrared_width)

    def get_depth(self):
        return self._kinect.get_last_depth_frame().reshape(self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width)

    def get_color(self, hsv=False):
        frame = self._kinect.get_last_color_frame().reshape(self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4)
        if hsv == False:
            return frame
        else:
            hsv_frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2HSV)
            return hsv_frame

    def get_points(self, frame):
        L = frame.size
        TYPE_CameraSpacePoint_Array = PyKinectV2._CameraSpacePoint * L
        csps = TYPE_CameraSpacePoint_Array()
        ptr_depth = np.ctypeslib.as_ctypes(frame.flatten())
        error_state = self._kinect._mapper.MapDepthFrameToCameraSpace(
            L, ptr_depth,
            L, csps)
        if error_state:
            raise Exception("Could not map depth frame to camera space! " + str(error_state))

        pf_csps = ctypes.cast(csps, ctypes.POINTER(ctypes.c_float))
        data = np.copy(np.ctypeslib.as_array(pf_csps, shape=(self.infrared_height, self.infrared_width,
                                                             3)))
        del pf_csps, csps, ptr_depth, TYPE_CameraSpacePoint_Array
        return data

    def get_points_color(self, frame):
        L = frame.size
        S=1080*1920
        TYPE_CameraSpacePoint_Array = PyKinectV2._CameraSpacePoint * S
        csps = TYPE_CameraSpacePoint_Array()
        ptr_depth = np.ctypeslib.as_ctypes(frame.flatten())
        error_state = self._kinect._mapper.MapColorFrameToCameraSpace(
            L, ptr_depth,
            S, csps)
        if error_state:
            raise Exception("Could not map depth frame to camera space! " + str(error_state))

        pf_csps = ctypes.cast(csps, ctypes.POINTER(ctypes.c_float))
        data = np.copy(np.ctypeslib.as_array(pf_csps, shape=(self.color_height, self.color_width,
                                                             3)))
        del pf_csps, csps, ptr_depth, TYPE_CameraSpacePoint_Array
        return data

    def get_point_location(self, x, y, depth):  # from depth to camera space
        # depth - uint16
        # x - y - pixel position
        point = _DepthSpacePoint()
        point.x = x
        point.y = y
        return self._kinect._mapper.MapDepthPointToCameraSpace(point, depth)

    def get_depth_point_in_rgb(self, x, y, depth):
        point = _DepthSpacePoint()
        point.x = x
        point.y = y
        return self._kinect._mapper.MapDepthPointToColorSpace(point, depth)