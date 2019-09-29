# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import os.path
import sys

import cv2
import numpy as np
import wrnchAI

from visualizer import Visualizer


def annotate_predicted_classes(visualizer,
                               submodel_index,
                               submodel_name,
                               class_probabilities,
                               class_names,
                               joints):
    predicted_class_name = class_names[np.argmax(class_probabilities)]
    text = submodel_name + ': ' + predicted_class_name

    (_, text_height), _ = visualizer.get_text_size(text)

    if submodel_index == 0:
        vertical_offset = 0
    else:
        vertical_offset = text_height + 5

    visualizer.draw_text_in_frame(text,
                                  x=int(joints[0] * visualizer.frame.shape[1]),
                                  y=int(
                                      joints[1] * visualizer.frame.shape[0]+vertical_offset))


if not wrnchAI.license_check():
    sys.exit('A valid license is required to run the samples')

num_args = len(sys.argv)
if num_args < 2 or num_args > 3:
    sys.exit(
        'Usage: python wrPose2d_sample.py <model path> [camera index {0}]')

if num_args == 3:
    webcam_index = int(sys.argv[2])
else:
    webcam_index = 0

models_dir = sys.argv[1]

#  the activity model wrnch_activity_v1.0.enc predicts 7 classes:
#
#  Gesture Name            | Type | Explaination
#  ==========================================================================================
#  None                   | No gesture is predicted.
#  -----------------------------------------------------------------------------------------
#  (left) Fist Pump       | Left hand is in a fist and left arm is repeatedly extending
#                         | up and down overhead.
#  -----------------------------------------------------------------------------------------
#  (left) Wave            | Left hand is waving.
#  -----------------------------------------------------------------------------------------
#  (left) Come here       | Left hand makes a come here gestures, wrist and/or finger tips
#                         | are moving cyclically (in a scooping motion) towards the body
#  -----------------------------------------------------------------------------------------
#  (left) Stop            | Static gesture. Arm is fully or nearly extended outwards (not
#                         | upwards or downwards). Fingers may be together or somewhat apart
#                         | but not fully splayed.
#  -----------------------------------------------------------------------------------------
#  Clap                   | Both hands come together, forming a clap. Note that unlike the
#                         | previous 4 gestures, this is not predicting something about just
#                         | the left arm of the body: it's predicting something about both
#                         | arms together.
#  -----------------------------------------------------------------------------------------
#  Summon                 | Arms are straight, starting at the sides of the body, then are
#                         | raised in tandem forming a V in front of the body. Like the Clap
#                         | gesture,  this gestures is not just predicting something about
#                         | the left arm of the body: it's predicting something about both
#                         | arms together.
#  -----------------------------------------------------------------------------------------
activity_model_name = 'wrnch_activity_v1.0.enc'
activity_model_path = os.path.join(models_dir, activity_model_name)

# Builders are how we create `ActivityModelGroup` instances
builder = wrnchAI.ActivityModelGroup.Builder()

# Add a submodel to predict gestures on the left side of the body.
builder.add_submodel(activity_model_path)

#  Add a reflected submdel to predict gestures on the right side of the body.
#
#  Due to the reflection, the predicted gestures of this submodel are effectively:
#  * (right) Fist Pump
#  * (right) Wave
#  * (right) Come here
#  * (right) Stop
#  * Clap
#  * Summon
#
#  We say 'effectively' because some of the strings in ClassNames() returned by this submodel
#  will still have 'left' in them.
builder.add_reflected_submodel(activity_model_path)

print('Initializing networks... (this could take some time)')
print('Building activity model ...')
activity_model_group = builder.build()
print('Building pose estimator ... ')
pose_estimator = builder.build_compatible_estimator(
    models_dir, device_id=0, desired_net_width=328, desired_net_height=184)
options = builder.build_compatible_options()
print('Initialization done!')


print('Opening webcam...')
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    sys.exit('Cannot open webcam.')

visualizer = Visualizer()

bone_pairs = pose_estimator.human_2d_output_format().bone_pairs()

assert activity_model_group.num_submodels == 2, ('To create activityModelGroup, we called builder.add_submodel exactly once and'
                                                 '`builder.add_reflected_submodel` exactly once as well.')

class_names = activity_model_group.submodel(0).class_names
submodel_names = 'unreflected model', 'reflected model'

while True:
    ret, frame = cap.read()

    if frame is None:
        break

    pose_estimator.process_frame(frame, options)

    activity_model_group.process_poses(
        pose_estimator=pose_estimator, image_width=frame.shape[1], image_height=frame.shape[0])

    humans2d = pose_estimator.humans_2d()

    visualizer.draw_image(frame)

    for human in humans2d:
        person_id = human.id

        joints = human.joints()

        visualizer.draw_points(joints)
        visualizer.draw_lines(joints, bone_pairs)

        for submodel_index in (0, 1):
            # `submodel_index` 0 corresponds to `builder.add_submodel(activity_model_path)`, the
            # unreflected model. `submodel_index` 1 corresponds to
            # `builder.AddReflectedSubmodel(activity_model_path)`, the reflected model
            activity_model = activity_model_group.submodel(submodel_index)

            # an individual model for a person holds probabilities of gestures for that person
            individual_model = activity_model.individual_model(person_id)

            class_probabilities = individual_model.probabilities()

            annotate_predicted_classes(
                visualizer, submodel_index, submodel_names[submodel_index], class_probabilities, class_names, joints)

    visualizer.show()

    key = cv2.waitKey(1)

    if key & 255 == 27:
        break


cap.release()
cv2.destroyAllWindows()
