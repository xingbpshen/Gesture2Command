# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import numpy as np
import cv2


WRNCH_BLUE = (226.95, 168.3, 38.25)
RED = (22.5, 22.5, 229)
FILLED = -1
AA_LINE = 16


class Visualizer:
    def __init__(self):
        self.frame = np.zeros((0, 0), dtype=np.uint8)
        self.name = ""
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

    def draw_image(self, data):
        self.frame = data

    def draw_points(self, points, colour=WRNCH_BLUE, joint_size=8):
        width = self.frame.shape[1]
        height = self.frame.shape[0]
        for i in range(len(points)//2):
            x = int(points[2 * i] * width)
            y = int(points[2 * i + 1] * height)

            if x >= 0 and y >= 0:
                cv2.circle(self.frame, (x, y), joint_size,
                           colour, FILLED, AA_LINE)

    def draw_points3d(self, points, colour=WRNCH_BLUE, joint_size=8):
        width = self.frame.shape[1]
        height = self.frame.shape[0]
        for i in range(len(points)//3):
            x = int(points[3 * i] * width)
            y = int(points[3 * i + 1] * height)
            # z = np.float32(points[3 * i + 2] * height)  Depth is store here

            if x >= 0 and y >= 0:
                cv2.circle(self.frame, (x, y), joint_size,
                           colour, FILLED, AA_LINE)

    def draw_lines(self, points, bone_pairs, colour=WRNCH_BLUE, bone_width=3):
        width = self.frame.shape[1]
        height = self.frame.shape[0]

        for joint_idx_0, joint_idx_1 in bone_pairs:
            x1 = int(points[joint_idx_0 * 2] * width)
            y1 = int(points[joint_idx_0 * 2 + 1] * height)
            x2 = int(points[joint_idx_1 * 2] * width)
            y2 = int(points[joint_idx_1 * 2 + 1] * height)

            if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                cv2.line(self.frame, (x1, y1), (x2, y2), colour,
                         bone_width, AA_LINE)

    def draw_box(self, x, y, box_width, box_height, colour=WRNCH_BLUE, thickness=3):
        frame_width = self.frame.shape[1]
        frame_height = self.frame.shape[0]

        p1 = (int(x*frame_width), int(y*frame_height))
        p2 = (int((x+box_width) * frame_width),
              int((y+box_height) * frame_height))

        cv2.rectangle(self.frame, p1, p2, colour,
                      int(thickness), AA_LINE)

    def draw_arrow(self, arrow, colour=RED, thickness=3):
        frame_width = self.frame.shape[1]
        frame_height = self.frame.shape[0]

        scaled_arrow = [(int(x*frame_width), int(y*frame_height))
                        for (x, y) in arrow]

        base = scaled_arrow[0]
        tip = scaled_arrow[1]

        if base[0] >= 0 and tip[0] >= 0:
            cv2.line(self.frame, base, tip, colour, thickness, AA_LINE)

    def draw_text_in_frame(self, text, x, y,
                           color=(0, 0, 255),
                           font_face=cv2.FONT_HERSHEY_DUPLEX,
                           font_scale=0.75,
                           thickness=1):
        cv2.putText(self.frame, text, (x, y),
                    fontFace=font_face, color=color, fontScale=font_scale, thickness=thickness)

    def get_text_size(self, text, font_face=cv2.FONT_HERSHEY_DUPLEX,
                      font_scale=0.75,
                      thickness=1):
        return cv2.getTextSize(text, fontFace=font_face, fontScale=font_scale, thickness=thickness)

    def show(self, wait=False):
        cv2.resizeWindow(self.name, self.frame.shape[1], self.frame.shape[0])
        cv2.imshow(self.name, self.frame)
        if wait:
            cv2.waitKey()
