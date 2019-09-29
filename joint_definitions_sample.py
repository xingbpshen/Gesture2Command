# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import wrnchAI

for definition in wrnchAI.JointDefinitionRegistry.available_definitions():
    wrnchAI.JointDefinitionRegistry.get(definition).print_joint_definition()
