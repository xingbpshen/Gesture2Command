# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import sys
import time

import wrnchAI


if not wrnchAI.license_check():
    sys.exit('A valid license is required to run the samples')

if len(sys.argv) != 2:
    sys.exit(
        'Usage: python pose2d_sample.py <model path>')

params = wrnchAI.PoseParams()

output_format = wrnchAI.JointDefinitionRegistry.get('j23')

print('Initializing networks... (slow, innit?)')
t0 = time.time()
estimator = wrnchAI.PoseEstimator(
    models_path=sys.argv[1], params=params, gpu_id=0, output_format=output_format)
print('Initialization done. Took', time.time() - t0, 'seconds')

print('serializing pose estimator to memory ...')
serialized_estimator = estimator.serialize()
print('done serializing. Deserializing ...')
t0 = time.time()
deserialized_estimator = wrnchAI.PoseEstimator.deserialize(
    serialized_estimator)
print('deserializing (from memory) took', time.time() - t0,
      'seconds. Compare to initializing the other way!')
