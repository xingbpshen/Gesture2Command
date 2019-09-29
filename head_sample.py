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
    webcam_index = int(argv[2])
else:
    webcam_index = 0

print('Initializing networks...')
models_path = sys.argv[1]
estimator = wrnchAI.PoseEstimator(models_path)
estimator.initialize_head(models_path)
print('Initialization done!')

options = wrnchAI.PoseEstimatorOptions()
options.estimate_heads = True
options.estimate_face_poses = True

print('Opening webcam...')
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    sys.exit('Cannot open webcam.')

visualizer = Visualizer()

while True:
    ret, frame = cap.read()

    if frame is not None:

        estimator.process_frame(frame, options)
        heads = estimator.heads()
        faces = estimator.faces()

        visualizer.draw_image(frame)
        for head in heads:
            bounding_box = head.bounding_box
            visualizer.draw_box(bounding_box.min_x, bounding_box.min_y,
                                bounding_box.width, bounding_box.height)

        for face in faces:
            landmarks = face.landmarks()
            arrow = face.arrow()

            visualizer.draw_points(landmarks, joint_size=2)
            visualizer.draw_arrow(arrow)

        visualizer.show()

    key = cv2.waitKey(1)

    if key & 255 == 27:  # escape key
        break


cap.release()

cv2.destroyAllWindows()
