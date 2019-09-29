# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import cv2
import wrnchAI
import sys


if not wrnchAI.license_check():
    sys.exit('A valid license is required to run the samples')

num_args = len(sys.argv)
if num_args != 3:
    sys.exit('Usage: python wrHello_sample.py <image file> <model path> ')

frame = cv2.imread(sys.argv[1])

print('Initializing networks...')
estimator = wrnchAI.PoseEstimator(sys.argv[2])
print('Initialization done')

options = wrnchAI.PoseEstimatorOptions()

print('Inferring ...')

estimator.process_frame(frame, options)

numPersons = len(estimator.humans_2d())

print('Inference done! Found ', numPersons, ' humans')
