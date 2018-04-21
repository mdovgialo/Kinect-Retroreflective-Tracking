import cv2
import numpy as np

class Tracker:
    def __init__(self, color_ranges, vr_emu_tracker, color_hue=None):
        self.x = 0
        self.y = 0
        self.z = 0
        self.color_ranges = color_ranges  # list of 2 ints between 0-180 of the hue value
        self.predictor = cv2.KalmanFilter(3, 3)
        if color_hue is None:
            self.color_hue = np.random.randint(0, 180)
        else:
            self.color_hue = color_hue
        self.vr_emu_tracker = vr_emu_tracker

    def update_from_masses(self, masses):
        my_masses = []
        for mass in masses:
            color = mass[1]
            for range in self.color_ranges:

                hue = color[0]
                saturation = color[1]
                lightness = color[2]

                # print(range)
                # print(color)
                # print()
                # print()
                # print()
                #
                # import IPython
                # IPython.embed()
                if hue >= range[0] and hue <=range[1]:# and saturation > 0.2*255 and lightness > 0.1*255 and lightness<0.8:
                    my_masses.append(mass[0])

        if my_masses:
            center = np.median(my_masses, axis=0)
            self.x, self.y, self.z = center
            try:
                self.vr_emu_tracker.update(*center)
            except:
                pass
