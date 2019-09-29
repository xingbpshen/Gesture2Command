# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import sys

import cv2
import wrnchAI
from visualizer import Visualizer

if not wrnchAI.license_check():
    sys.exit('A valid license is required to run the samples')


num_args = len(sys.argv)
if num_args < 2 or num_args > 3:
    sys.exit('Usage: python wrHands_sample.py <model path> [camera index {0}]')

if num_args == 3:
    webcam_index = int(sys.argv[2])
else:
    webcam_index = 0

print('Initializing networks...')
models_dir = sys.argv[1]
estimator = wrnchAI.PoseEstimator(sys.argv[1])
estimator.initialize_hands2d(models_dir)
print('Initialization done!')

options = wrnchAI.PoseEstimatorOptions()
options.estimate_all_handboxes = True
options.estimate_hands = True

print('Opening webcam...')
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    sys.exit('Cannot open webcam.')

visualizer = Visualizer()

joint_definition = estimator.hand_output_format()
bone_pairs = joint_definition.bone_pairs()


def draw_hands(hands):
    for hand in hands:
        joints = hand.joints()

        visualizer.draw_points(joints, joint_size=6)
        visualizer.draw_lines(joints, bone_pairs, bone_width=2)


while True:
    ret, frame = cap.read()

    if frame is not None:

        estimator.process_frame(frame, options)

        visualizer.draw_image(frame)

        draw_hands(estimator.left_hands())
        draw_hands(estimator.right_hands())
        hand_box_pairs = estimator.hand_boxes()

        for hand_box_pair in hand_box_pairs:
            left_box = hand_box_pair.left
            right_box = hand_box_pair.right

            visualizer.draw_box(left_box.min_x, left_box.min_y,
                                left_box.width, left_box.height)
            visualizer.draw_box(right_box.min_x, right_box.min_y,
                                right_box.width, right_box.height)

        visualizer.show()

    key = cv2.waitKey(1)

    if key & 255 == 27:  # escape key
        break


cap.release()

cv2.destroyAllWindows()
