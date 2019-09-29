# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import sys
import cv2
import wrnchAI
from visualizer import Visualizer

if not wrnchAI.license_check():
    sys.exit("A valid license is required to run the samples")

num_args = len(sys.argv)
if num_args < 2 or num_args > 3:
    sys.exit(
        "Usage: python wrPose2d_sample.py <model path> [camera index {0}]")

if num_args == 3:
    webcam_index = int(sys.argv[2])
else:
    webcam_index = 0

print("Initializing networks...")
models_path = sys.argv[1]
estimator = wrnchAI.PoseEstimator(models_path)
estimator.initialize_3d(models_path)
print("Initialization done!")

options = wrnchAI.PoseEstimatorOptions()
options.estimate_3d = True

print("Opening webcam...")
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    sys.exit("Cannot open webcam.")

visualizer = Visualizer()

joint_definition = estimator.human_2d_output_format()
bone_pairs = joint_definition.bone_pairs()

while True:
    ret, frame = cap.read()

    if frame is not None:

        estimator.process_frame(frame, options)
        humans3d = estimator.raw_humans_3d()

        visualizer.draw_image(frame)

        for human in humans3d:
            positions = human.positions()

            visualizer.draw_points3d(positions)

        visualizer.show()

    key = cv2.waitKey(1)

    if key & 255 == 27:
        break

cap.release()
cv2.destroyAllWindows()
