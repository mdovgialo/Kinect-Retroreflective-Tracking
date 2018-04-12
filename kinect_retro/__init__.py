import ctypes

import numpy as np
from pykinect2 import PyKinectV2, PyKinectRuntime
import cv2

class InfraredSource:
    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared | PyKinectV2.FrameSourceTypes_Depth)
        self.infrared_width = self._kinect.infrared_frame_desc.Width
        self.infrared_height = self._kinect.infrared_frame_desc.Height

    def get_infrared(self):
        return self._kinect.get_last_infrared_frame().reshape(self.infrared_height, self.infrared_width)
    def get_depth(self):
        return self._kinect.get_last_depth_frame().reshape(self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width)

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