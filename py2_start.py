# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import json

import os
import sys
import thread

import numpy as np

import cv2
import wrnchAI
from visualizer import Visualizer
'''
if not wrnchAI.license_check():
    sys.exit('A valid license is required to run the samples')

num_args = len(sys.argv)
if num_args < 2 or num_args > 3:
    sys.exit(
        'Usage: python pose2d_sample.py <model path> [camera index {0}]')

if num_args == 3:
    webcam_index = int(sys.argv[2])
else:
    webcam_index = 0
'''
webcam_index = 0

params = wrnchAI.PoseParams()
params.bone_sensitivity = wrnchAI.Sensitivity.high
params.joint_sensitivity = wrnchAI.Sensitivity.high
params.enable_tracking = True

# Default Model resolution
params.preferred_net_width = 328
params.preferred_net_height = 184

output_format = wrnchAI.JointDefinitionRegistry.get('j25')

print('Initializing networks...')
estimator = wrnchAI.PoseEstimator(
    # models_path=sys.argv[1], params=params, gpu_id=0, output_format=output_format)
    models_path="wrModels", params=params, gpu_id=0, output_format=output_format)
print('Initialization done!')

options = wrnchAI.PoseEstimatorOptions()

print('Opening webcam...')
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    sys.exit('Cannot open webcam.')

visualizer = Visualizer()

joint_definition = estimator.human_2d_output_format()
bone_pairs = joint_definition.bone_pairs()

result = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

count = 0

os.system(r'start "happy" C:\Users\zhang\Anaconda3\python py3_start.py')

while True:
    ret, frame = cap.read()

    if frame is not None:
        estimator.process_frame(frame, options)
        humans2d = estimator.humans_2d()

        visualizer.draw_image(frame)
        for human in humans2d:
            count += 1
            joints = human.joints()

            # print(joints)
            result = np.concatenate((result, np.array([joints.tolist()])), axis=0)

            visualizer.draw_points(joints)
            visualizer.draw_lines(joints, bone_pairs)

        visualizer.show()

    key = cv2.waitKey(1)

    # with open("temp.json", "a") as file:
    #     json.dump({"0": joints[14:42].tolist()})
    # os.system("C:\Users\zhang\Anaconda3\python test.py")
    # os.system("del temp.json")
    # if count % 30 == 0 and count != 0:
    if count == 15:
        with open("temp.json", "a") as file:
            epoch = int(count/15 - 1)
            # if (epoch>0):
            #     os.system("del temp.json")
            print(epoch)
            json.dump({"0": result[(epoch*15+1):(epoch*15+16), 14:42].tolist()}, file)
        # os.system(r'start "happy" C:\Users\zhang\Anaconda3\python py3_start.py')
        break

    if key & 255 == 27:
        break
# os.system("del temp.json")
result = result[1:, 14:42]
# print(result)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

# os.system("C:\Users\zhang\Anaconda3\python --version")

# with open("result3.json", "a") as file:
#     json.dump({"3": result.tolist()}, file)
# print(np.size(result, 0))
