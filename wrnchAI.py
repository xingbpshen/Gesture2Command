'''
wrnchAI.py -- a pure Python interface to wrnchAI
'''
from collections import namedtuple
import json

import _wrnchAI

__version__ = _wrnchAI.__version__
with_openvino = _wrnchAI.with_openvino


def license_check():
    return _wrnchAI.license_check()


def license_check_path(license_path=''):
    '''
    The `license_path` argument must either be '.', to search current
    directory, or the path to the license. If it points directly to a
    file, the file is probed for a valid license. If it points to a
    directory, then all files ending with '.lic' are probed. for a valid
    license. If not provided, the default license search path is the user
    home directory.
    '''
    return _wrnchAI.license_check_path(license_path)


def num_available_gpu():
    return _wrnchAI.num_available_gpu()


class PoseParams(object):
    '''
    "Configure-time" parameters of a pose estimator. These parameters may only
    be set once on a PoseEstimator and may not be changed dynamically.
    '''

    def __init__(self, impl=None):
        if impl is None:
            self.impl = _wrnchAI.PoseParams()
        else:
            self.impl = impl

    @property
    def bone_sensitivity(self):
        return self.impl.bone_sensitivity

    @bone_sensitivity.setter
    def bone_sensitivity(self, sensitivity):
        self.impl.bone_sensitivity = sensitivity

    @property
    def joint_sensitivity(self):
        return self.impl.joint_sensitivity

    @joint_sensitivity.setter
    def joint_sensitivity(self, sensitivity):
        self.impl.joint_sensitivity = sensitivity

    @property
    def net_precision_2d(self):
        return self.impl.net_precision_2d

    @net_precision_2d.setter
    def net_precision_2d(self, precision):
        self.impl.net_precision_2d = precision

    @property
    def net_precision_3d(self):
        return self.impl.net_precision_3d

    @net_precision_3d.setter
    def net_precision_3d(self, precision):
        self.impl.net_precision_3d = precision

    @property
    def enable_tracking(self):
        return self.impl.enable_tracking

    @property
    def tracker_kind(self):
        return self.impl.tracker_kind

    @tracker_kind.setter
    def tracker_kind(self, tracker_kind):
        self.impl.tracker_kind = tracker_kind

    @enable_tracking.setter
    def enable_tracking(self, true_or_false):
        '''
        Enable or disable the (stateful) tracking module on a pose estimator
        (which is on by default).

        The tracker attempts to set consistent ids on persons in frames through time by
        looking at positions and "appearance" of persons (ids are accessed through the
        `id` property on pose types, eg, `Pose2d.id`). The tracker works best with stable lighting
        conditions, BGR (color) frames, and accurate, complete detections. In order to improve
        detections, a higher 2d net resolution may be needed. In order to have
        complete detections, setting a high min valid joints on PoseParams (for example, 12
        or more) which configure a PoseEstimator may help. Note that the default min valid
        joints used by PoseEstimators is currently 3. This tracker comes with a nontrivial
        runtime cost -- users not interested in tracking should disable tracking (by using
        `pose_params.enable_tracking = False`.

        :param true_or_false: bool. `False` disables tracking,
            `True` enables tracking (which is set by default).
        '''
        self.impl.enable_tracking = true_or_false

    @property
    def preferred_net_width(self):
        return self.impl.preferred_net_width

    @preferred_net_width.setter
    def preferred_net_width(self, net_width):
        self.impl.preferred_net_width = net_width

    @property
    def preferred_net_height(self):
        return self.impl.preferred_net_height

    @preferred_net_height.setter
    def preferred_net_height(self, net_height):
        self.impl.preferred_net_height = net_height

    @property
    def preferred_net_width_3d(self):
        return self.impl.preferred_net_width_3d

    @preferred_net_width_3d.setter
    def preferred_net_width_3d(self, net_width_3d):
        self.impl.preferred_net_width_3d = net_width_3d

    @property
    def preferred_net_height_3d(self):
        return self.impl.preferred_net_height_3d

    @preferred_net_height_3d.setter
    def preferred_net_height_3d(self, net_height_3d):
        self.impl.preferred_net_height_3d = net_height_3d

    @property
    def smoothing_betaX(self):
        return self.impl.smoothing_betaX

    @smoothing_betaX.setter
    def smoothing_betaX(self, beta_x):
        self.impl.smoothing_betaX = beta_x

    @property
    def smoothing_betaY(self):
        return self.impl.smoothing_betaY

    @smoothing_betaY.setter
    def smoothing_betaY(self, beta_y):
        self.impl.smoothing_betaY = beta_y

    @property
    def smoothing_betaZ(self):
        return self.impl.smoothing_betaZ

    @smoothing_betaZ.setter
    def smoothing_betaZ(self, beta_z):
        self.impl.smoothing_betaZ = beta_z

    @property
    def smoothing_cutoff_freq_velocity(self):
        return self.impl.smoothing_cutoff_freq_velocity

    @smoothing_cutoff_freq_velocity.setter
    def smoothing_cutoff_freq_velocity(self, velocity):
        self.impl.smoothing_cutoff_freq_velocity = velocity

    @property
    def smoothing_min_cutoff_freq_position(self):
        return self.impl.smoothing_min_cutoff_freq_position

    @smoothing_min_cutoff_freq_position.setter
    def smoothing_min_cutoff_freq_position(self, position):
        self.impl.smoothing_min_cutoff_freq_position = position

    @property
    def min_valid_joints(self):
        return self.impl.min_valid_joints

    @min_valid_joints.setter
    def min_valid_joints(self, num_joints):
        self.impl.min_valid_joints = num_joints


class IKParams(object):
    '''
    Parameters for PoseIK IK-solvers.
    '''

    def __init__(self):
        self.impl = _wrnchAI.IKParams()

    @property
    def trans_reach(self):
        return self.impl.trans_reach

    @trans_reach.setter
    def trans_reach(self, reach):
        self.impl.trans_reach = reach

    @property
    def rot_reach(self):
        return self.impl.rot_reach

    @rot_reach.setter
    def rot_reach(self, reach):
        self.impl.rot_reach = reach

    @property
    def pull(self):
        return self.impl.pull

    @pull.setter
    def pull(self, pull):
        self.impl.pull = pull

    @property
    def resist(self):
        return self.impl.resist

    @resist.setter
    def resist(self, resist):
        self.impl.resist = resist

    @property
    def max_angular_velocity(self):
        return self.impl.max_angular_velocity

    @max_angular_velocity.setter
    def max_angular_velocity(self, velocity):
        self.impl.max_angular_velocity = velocity

    @property
    def fps(self):
        return self.impl.fps

    @fps.setter
    def fps(self, fps):
        self.impl.fps = fps

    @property
    def joint_visibility_thresh(self):
        return self.impl.joint_visibility_thresh

    @joint_visibility_thresh.setter
    def joint_visibility_thresh(self, thresh):
        self.impl.joint_visibility_thresh = thresh


class PoseEstimatorOptions(object):
    '''
    Options used by `PoseEstimator.process_frame`. Options select which
    features to run on calls to process_frame (eg, whether to estimate
    hands or not).
    '''

    def __init__(self):
        self.impl = _wrnchAI.PoseEstimatorOptions()

    @property
    def estimate_mask(self):
        return self.impl.estimate_mask

    @estimate_mask.setter
    def estimate_mask(self, true_or_false):
        self.impl.estimate_mask = true_or_false

    @property
    def estimate_3d(self):
        return self.impl.estimate_3d

    @estimate_3d.setter
    def estimate_3d(self, true_or_false):
        self.impl.estimate_3d = true_or_false

    @property
    def estimate_hands(self):
        return self.impl.estimate_hands

    @estimate_hands.setter
    def estimate_hands(self, true_or_false):
        self.impl.estimate_hands = true_or_false

    @property
    def estimate_all_handboxes(self):
        return self.impl.estimate_all_handboxes

    @estimate_all_handboxes.setter
    def estimate_all_handboxes(self, true_or_false):
        self.impl.estimate_all_handboxes = true_or_false

    @property
    def estimate_heads(self):
        return self.impl.estimate_heads

    @estimate_heads.setter
    def estimate_heads(self, true_or_false):
        self.impl.estimate_heads = true_or_false

    @property
    def estimate_face_poses(self):
        return self.impl.estimate_face_poses

    @estimate_face_poses.setter
    def estimate_face_poses(self, true_or_false):
        self.impl.estimate_face_poses = true_or_false

    @property
    def estimate_single(self):
        return self.impl.estimate_single

    @estimate_single.setter
    def estimate_single(self, true_or_false):
        self.impl.estimate_single = true_or_false

    @property
    def use_ik(self):
        return self.impl.use_ik

    @use_ik.setter
    def use_ik(self, true_or_false):
        self.impl.use_ik = true_or_false

    @property
    def main_person_id_mode(self):
        return self.impl.main_person_id_mode

    @main_person_id_mode.setter
    def main_person_id_mode(self, mode):
        self.impl.main_person_id_mode = mode

    @property
    def enable_joint_smoothing(self):
        return self.impl.enable_joint_smoothing

    @enable_joint_smoothing.setter
    def enable_joint_smoothing(self, true_or_false):
        self.impl.enable_joint_smoothing = true_or_false

    @property
    def enable_head_smoothing(self):
        return self.impl.enable_head_smoothing

    @enable_head_smoothing.setter
    def enable_head_smoothing(self, true_or_false):
        self.impl.enable_head_smoothing = true_or_false


class PoseEstimator(object):
    '''
    Class for estimating 2d, 3d, hands, head, and face poses.
    '''

    def __init__(self,
                 models_path=None,
                 model_name_2d='',
                 license_path='',
                 params=None,
                 gpu_id=0,
                 output_format=None,
                 impl=None):
        '''
        Initialize a pose estimator

        :param models_path: str or None. full path to the models directory
        :param model_name_2d: str. must be the full name of the 2D pose model,
            e.g., wrnch_pose2d_seg_v4.2.enc (or empty)
        :param license_path: str. must either be '.', to search current
            directory, or the path to the license. If it points directly to a
            file, the file is probed for a valid license. If it points to a
            directory, then all files ending with '.lic' are probed. for a
            valid license. If not provided, the default license search path
            is the user home directory.
        :param params: PoseParams. The parameters which set "configure time"
            data on the pose estimator.
        :param gpu_id: int. The GPU id on which the pose estimator will run.
        :param output_format: JointDefinition. The joint format in which the
            pose estimator returns poses.
        :param impl: _wrnchAI.PoseEstimator. Not normally used by clients of
            wrnchAI.
        '''
        def impl_or_none(obj):
            return obj.impl if obj is not None else None

        if impl is None:
            self.impl = _wrnchAI.PoseEstimator(
                models_path=models_path,
                model_name_2d=model_name_2d,
                license_path=license_path,
                params=impl_or_none(params),
                gpu_id=gpu_id,
                output_format=impl_or_none(output_format))
        else:
            self.impl = impl

    def initialize_3d(self, models_path, ik_params=None):
        '''
        Initialize the 3d pose estimation capabilities of a pose estimator.

        :param models_path: str. Full path to the models directory.
        :param ik_params: IKParams or None. Optional IK parameters to set
            on the 3d estimation subsystem.
        '''
        if ik_params is None:
            self.impl.initialize_3d(models_path)
        else:
            self.impl.initialize_3d(ik_params, models_path)

    def initialize_hands2d(self, models_path):
        '''
        Initialize the hand pose estimation capabilities of a pose estimator.

        :param models_path: str. Full path to the models directory.
        '''
        self.impl.initialize_hands2d(models_path)

    def initialize_head(self,
                        models_directory,
                        beta_x=0.1,
                        beta_y=0.1,
                        min_freq_cutff_position=0.1,
                        freq_cutoff_velocity=0.2):
        '''
        Initialize the hand pose estimation capabilities of a pose estimator.

        :param models_path: str. Full path to the models directory.
        '''
        self.impl.initialize_head(
            models_directory=models_directory,
            beta_x=beta_x,
            beta_y=beta_y,
            min_freq_cutff_position=min_freq_cutff_position,
            freq_cutoff_velocity=freq_cutoff_velocity)

    def process_frame(self, bgr_array, options):
        '''
        The `bgr_array` argument should be a numpy array representing an image
        in openCV 3-channel format, i.e., a row major sequence of bgr triples
        [b1,g1,r1,b2,g2,r2, ... ] intensity values in the range [0,255]
        '''
        return self.impl.process_frame(bgr_array, options.impl)

    def process_frame_gray(self, gray_array, options):
        '''
        The `gray_array` argument should be a numpy array representing an image
        in openCV 1-channel format, i.e., a row major sequence of intensity
        values in the range [0,255]
        '''
        return self.impl.process_frame_gray(gray_array, options.impl)

    def has_IK(self):
        return self.impl.has_IK()

    def humans_2d(self):
        '''
        Get the Pose2d's last estimated by this PoseEstimator.
        This poses should be interpreted with respect to the
        `human_2d_output_format` JointDefinition.

        :returns: list of Pose2d
        '''
        return pose_2d_list(self.impl.humans_2d())

    def left_hands(self):
        '''
        Get hand pose of left hands estimated by this PoseEstimator.
        This poses should be interpreted with respect to the
        `hand_output_format` JointDefinition.

        :returns: list of Pose2d
        '''
        return pose_2d_list(self.impl.left_hands())

    def right_hands(self):
        '''
        Get hand pose of right hands estimated by this PoseEstimator.
        This poses should be interpreted with respect to the
        `hand_output_format` JointDefinition.

        :returns: list of Pose2d
        '''
        return pose_2d_list(self.impl.right_hands())

    def heads(self):
        '''
        Get the head pose last estimated by this PoseEstimator.

        :returns: list of PoseHead
        '''
        return [PoseHead(head) for head in self.impl.heads()]

    def faces(self):
        '''
        Get the pose face last estimated by this PoseEstimator.

        :returns: list of PoseHead
        '''
        return [PoseFace(face) for face in self.impl.faces()]

    def humans_3d(self):
        '''
        Get the Pose3d's last estimated by this PoseEstimator.
        This poses should be interpreted with respect to the
        `human_3d_output_format` JointDefinition.

        :returns: list of Pose2d
        '''
        return pose_3d_list(self.impl.humans_3d())

    def raw_humans_3d(self):
        '''
        Get the "raw" Pose3d's last estimated by this PoseEstimator.
        This poses should be interpreted with respect to the
        `human_3d_output_format_raw` JointDefinition.

        :returns: list of Pose2d
        '''
        return pose_3d_list(self.impl.raw_humans_3d())

    def mask(self):
        '''
        The mask is stored as four 8-bit single channel images
        whose values represent the confidence that a human is present in that
        pixel.  The four generated masks are stored consecutively as follows:
        (0) body; (1) right hand; (2) left hand; and (3) both hands.
        '''
        return self.impl.mask()

    def hand_boxes(self):
        def translate(native_handbox_pair):
            return HandboxPair(
                left=_translate_box_2d(native_handbox_pair.left()),
                right=_translate_box_2d(native_handbox_pair.right()),
                id=native_handbox_pair.id)

        return [translate(box_pair) for box_pair in self.impl.hand_boxes()]

    def set_hand_segmenter(self, segmenter_type):
        '''
        Set the hand segmenter type on a pose estimator.

        :param segmenter_type: HandSegmenterType.
        '''
        self.impl.set_hand_segmenter(segmenter_type)

    def hand_segmenter(self):
        return self.impl.hand_segmenter()

    def input_width(self):
        return self.impl.input_width()

    def input_height(self):
        return self.impl.input_height()

    def params(self):
        return self.impl.params()

    def get_IK_property(self, prop, solver_id):
        '''
        :param prop: IKProperty.
        :param solver_id: int. Should be between 0 and get_num_IK_solvers.

        :returns: float.
        '''
        return self.impl.get_IK_property(prop, solver_id)

    def set_IK_property(self, prop, value, solver_id):
        '''
        :param prop: IKProperty
        :param value: float
        :param solver_id: int
        '''
        self.impl.set_IK_property(prop, value, solver_id)

    def tpose_3d(self):
        '''
        Get the "T-Pose" used by the 3d solver.

        :returns: Pose3d.
        '''
        return Pose3d(self.impl.tpose_3d())

    def get_num_IK_solvers(self):
        return self.impl.get_num_IK_solvers()

    def human_2d_output_format(self):
        '''
        Return the JointDefinition format used with which 2d poses returned by
        `self.humans_2d` are to be interpreted.

        :returns: JointDefinition
        '''
        return JointDefinition(self.impl.human_2d_output_format())

    def human_3d_output_format(self):
        '''
        Return the JointDefinition format used with which 3d poses returned by
        `self.humans_3d` are to be interpreted.

        :returns: JointDefinition
        '''
        return JointDefinition(self.impl.human_3d_output_format())

    def human_3d_output_format_raw(self):
        '''
        Return the JointDefinition format used with which 3d poses returned by
        `self.raw_humans_3d` are to be interpreted.

        :returns: JointDefinition
        '''
        return JointDefinition(self.impl.human_3d_output_format_raw())

    def hand_output_format(self):
        '''
        Return the JointDefinition format used with which 3d poses returned by
        `self.left_hands` and `self.right_hands` are to be interpreted.

        :returns: JointDefinition
        '''
        return JointDefinition(self.impl.hand_output_format())

    def pose_params(self):
        '''
        Get the pose params which were used when configuring this
        PoseEstimator.

        :returns: PoseParams
        '''
        return PoseParams(self.impl.pose_params())

    def serialize(self):
        '''
        Serialize a pose estimator to memory as a string.
        Such a string can be written to disk (using binary mode) and re-read
        later using the `PoseEstimator.deserialize` method to greatly speed up
        initialization times.

        :returns: str
        '''
        return self.impl.serialize()

    @classmethod
    def deserialize(cls, string, device_id=0):
        '''
        Deserialize a serialized estimator string into a fresh PoseEstimator.
        Such a string may be read from disk (using binary mode) from a previous
        call to `serialize`.

        :param string: str. the serialized data.
        :param device_id: int. the GPU device id on which the new pose
            estimator will run.

        :returns: PoseEstimator
        '''
        return cls(impl=_wrnchAI.PoseEstimator.deserialize(string, device_id))

    def results_to_json(self):
        '''
        Return a JSON object holding the current results held in the estimator.
        '''
        return json.loads(self.impl.results_to_json().decode('utf-8'))

    def clone(self):
        '''
        Clone a pose estimator: this is a quick way to create an identical copy
        of a given pose estimator -- much faster than calling `initialize`.

        :returns: a fresh pose estimator, identical to `self`.
        '''
        return PoseEstimator(impl=self.impl.clone())

    @property
    def tracker_kind(self):
        return self.impl.tracker_kind

    @property
    def is_3d_initialized(self):
        return self.impl.is_3d_initialized

    @property
    def is_head_initialized(self):
        return self.impl.is_head_initialized

    @property
    def are_hands_initialized(self):
        return self.impl.are_hands_initialized

    @property
    def supports_mask_estimation(self):
        return self.impl.supports_mask_estimation


class Pose2d(object):
    '''
    A Pose2d holds joint positions and scores computed by a PoseEstimator.
    By itself, a Pose2d cannot be interpreted -- one needs a JointDefinition to
    be able to make sense of the floats inside `joints` and `scores`.
    '''

    def __init__(self, impl):
        self.impl = impl

    @property
    def is_main(self):
        return self.impl.is_main

    @property
    def id(self):
        return self.impl.id

    @id.setter
    def id(self, new_id):
        self.impl.id = new_id

    def joints(self):
        '''
        :returns: 1d numpy array of float. This is a flattened array of
        the x,y pairs of joint positions inside the pose. Negative values
        represent "invisible" joints.
        '''
        return self.impl.joints()

    def bounding_box(self):
        return _translate_box_2d(self.impl.bounding_box())

    def scores(self):
        '''
        :returns: 1d numpy array of float. Giving the scores of each joint
        in the Pose.
        '''
        return self.impl.scores()

    def score(self):
        '''
        :returns: float. The clustering score of the pose, if clustered.
        '''
        return self.impl.score()


def pose_2d_list(pose_2ds):
    return [Pose2d(pose_2d) for pose_2d in pose_2ds]


class Pose3d(object):
    '''
    A Pose3d holds joint positions, scores, and rotations computed by a
    PoseEstimator or a PoseIK solver.
    By itself, a Pose3d cannot be interpreted -- one needs a JointDefinition to
    be able to make sense of the floats inside `joints`, `scores`, and
    `rotations`.
    '''

    def __init__(self, impl):
        self.impl = impl

    @property
    def id(self):
        return self.impl.id

    def positions(self):
        '''
        :returns: 1d numpy array of float. This is a flattened array of the
        x,y,z triples inside the pose. Negative or NaN values represent
        "invisible" joints.
        '''
        return self.impl.positions()

    def rotations(self):
        '''
        :returns: 1d numpy array of float. This is a flattened array of the
        4-tuple quaternions giving the local roations of joints.
        '''
        return self.impl.rotations()

    def scale_hint(self):
        return self.impl.scale_hint()


def pose_3d_list(pose_3ds):
    return [Pose3d(pose_3d) for pose_3d in pose_3ds]


class PoseHead(object):
    def __init__(self, _pose_head):
        self.impl = _pose_head

    @property
    def id(self):
        return self.impl.id

    @property
    def estimation_success(self):
        return self.impl.estimation_success

    @property
    def bounding_box(self):
        return _translate_box_2d(self.impl.bounding_box())

    def head_rotation(self):
        return self.impl.head_rotation


class PoseFace(object):
    def __init__(self, _pose_face):
        self.impl = _pose_face

    def landmarks(self):
        return self.impl.landmarks()

    def arrow(self):
        return self.impl.arrow()


class JointDefinition(object):
    '''
    JointDefinition describes a graph, with vertices being "joints"
    and edges being "bones". Joints are defined by name, from the
    `joint_names` function. Bones are defined by a list of pairs of
    integers (from the `bone_pairs` function). Each integer in a bone pair
    represents a bone connecting the corresponding pair of joint names.
    '''

    def __init__(self, _format):
        self.impl = _format

    def joint_names(self):
        '''
        :returns: list of str. a list of the joint names in this
        JointDefinition
        '''
        return self.impl.joint_names()

    @property
    def num_joints(self):
        return self.impl.num_joints

    def bone_pairs(self):
        '''
        :returns: list of pair of int. A list of joint pairs in this
            JointDefinition.
        '''
        return self.impl.bone_pairs()

    def name(self):
        return self.impl.name()

    def get_joint_index(self, joint_name):
        '''
        Given a Pose object (eg, Pose2d or Pose3d) in this JointDefinition,
        get the index in the joints (or scores, or rotations) array giving
        the value of the joint with this name. If joint_name is not in this
        definition, an exception is raised.

        For example, if `p` is a Pose2d in JointDefinition joint_def,
        then to find the joint values for joint named "HEAD", we can do:

          joint_index = joint_def.get_joint_index("HEAD")
          x = p.joints()[2 * joint_index]
          y = p.joints()[2 * joint_index + 1]
        '''
        return self.impl.get_joint_index(joint_name)

    def print_joint_definition(self):
        return self.impl.print_joint_definition()


class JointDefinitionRegistry(object):
    '''
    A registry of all available JointDefinitions inside wrnchAI.
    '''

    def __init__(self):
        self.impl = _wrnchAI.JointDefinitionRegistry()

    @staticmethod
    def get(name):
        return JointDefinition(_wrnchAI.JointDefinitionRegistry.get(name))

    @staticmethod
    def available_definitions():
        return _wrnchAI.JointDefinitionRegistry.available_definitions()


class PoseIK(object):
    '''
    Inverse Kinematics solver.
    '''

    def __init__(self,
                 input_format,
                 params,
                 initial_pose=None):
        if initial_pose is not None:
            self.impl = _wrnchAI.PoseIK(input_format=input_format.impl,
                                        params=params.impl,
                                        initial_pose=self._extract_pose_impl(
                                            initial_pose))
        else:
            self.impl = _wrnchAI.PoseIK(input_format=input_format.impl,
                                        params=params.impl)

    @staticmethod
    def _extract_pose_impl(pose):
        if isinstance(pose, Pose3d):
            return pose.impl
        return pose

    def set_params(self, params):
        self.impl.set_params(params)

    def get_params(self):
        return self.impl.get_params()

    def solve(self, pose, visibilities):
        return Pose3d(self.impl.solve(self._extract_pose_impl(pose),
                                      visibilities))

    def reset(self, initial_pose=None):
        if initial_pose is None:
            self.impl.reset()
        else:
            self.impl.reset(self._extract_pose_impl(initial_pose))

    def get_ik_property(self, prop):
        return self.impl.get_ik_property(prop)

    def set_ik_property(self, prop, value):
        self.impl.set_ik_property(prop, value)

    @property
    def get_output_format(self):
        return JointDefinition(self.impl.get_output_format)


Sensitivity = _wrnchAI.Sensitivity
MainPersonID = _wrnchAI.MainPersonID
NetPrecision = _wrnchAI.NetPrecision
HandSegmenterType = _wrnchAI.HandSegmenterType
IKProperty = _wrnchAI.IKProperty
TrackerKind = _wrnchAI.TrackerKind


Box2d = namedtuple('Box2d', ['min_x', 'min_y', 'width', 'height'])
Box3d = namedtuple(
    'Box3d', ['min_x', 'min_y', 'min_z', 'width', 'height', 'depth'])
HandboxPair = namedtuple('HandboxPair', ['left', 'right', 'id'])


def _translate_box_2d(box):
    return Box2d(min_x=box.min_x,
                 min_y=box.min_y,
                 width=box.width,
                 height=box.height)


class ActivityPoseEstimatorRequirementsView(object):
    '''
    describes requirements on PoseEstimator instances to be compatible with an
    ActivityModel or ActivityModelGroup
    '''

    def __init__(self, impl):
        self.impl = impl

    @property
    def requires_hands(self):
        '''
        :returns: bool
        '''
        return self.impl.requires_hands

    @property
    def preferred_output_format(self):
        '''return the "preferred" joint format to be used by a PoseEstimator'''
        return JointDefinition(self.impl.preferred_output_format)

    def is_compatible(self, pose_estimator):
        '''
        :returns: bool
        '''
        return self.impl.is_compatible(pose_estimator.impl)


class IndividualActivityModelView(object):
    '''
    Estimates activities (or "gestures") and their probabilities for an
    individual person. Currently, poses cannot be passed directly to
    IndividualActivityModel to update its internal state. Instead, they are
    passed through ActivityModelGroup instances.
    '''

    def __init__(self, impl):
        self.impl = impl

    @property
    def num_classes(self):
        return self.impl.num_classes

    @property
    def class_names(self):
        '''
        :returns: a list of str. contains the class names estimated by the
            model
        '''
        return self.impl.class_names

    def probabilities(self):
        '''
        :return: a list of the current estimated gesture-wise probabilities
        held in this  model. The order of probabilities corresponds to the
        order of the classes returned in `class_names`.
        '''
        return self.impl.probabilities()


class ActivityModelView(object):
    '''
    Estimates pose-based activities (or "gestures") for people ("individuals")
    over time. The underlying model is usually temporal, and requires
    processing of poses through time for best behavior. Currently, processing
    of poses is not exposed directly through functions on this class. Instead,
    poses are processed by an `ActivityModelGroup`, which holds a collection of
    `ActivityModel` (even just one).
    '''

    def __init__(self, impl):
        self.impl = impl

    @property
    def num_classes(self):
        return self.impl.num_classes

    @property
    def class_names(self):
        '''
        :returns: list of str. contains the class names estimated by the model
        '''
        return self.impl.class_names

    @property
    def person_ids(self):
        '''
        :returns: list of int. person ids which have been estimated by this
            model. Note that some of these personIds may not have been seen
            in many frames.
        '''
        return self.impl.person_ids

    def individual_model(self, person_id):
        '''
        :param person_id: int.

        attempt to access the individual activity model for a person by
        person_id. If no such person_id has been estimated, throw an exception.
        Note that even if an IndividualActivityModel exists for a given
        `personId`, that person may not have been seen for many frames
        '''
        return IndividualActivityModelView(
            self.impl.individual_model(person_id))


class ActivityModelGroup(object):
    '''
    Represents a collection of ActivityModel submodels
    '''
    class Builder(object):
        '''
        Builder pattern for ActivityModelGroup instances.
        '''

        def __init__(self):
            self.impl = _wrnchAI.ActivityModelGroupBuilder()

        def set_device_id(self, device_id):
            self.impl.set_device_id(device_id)

        def add_submodel(self, model_path):
            self.impl.add_submodel(model_path)

        def add_reflected_submodel(self, model_path):
            self.impl.add_reflected_submodel(model_path)

        def build(self):
            return ActivityModelGroup(self.impl.build())

        def build_compatible_estimator(self,
                                       pose_model_dir,
                                       device_id,
                                       desired_net_width,
                                       desired_net_height):
            return PoseEstimator(
                impl=self.impl.build_compatible_estimator(pose_model_dir,
                                                          device_id,
                                                          desired_net_width,
                                                          desired_net_height))

        def build_compatible_options(self):
            return ActivityPoseEstimatorRequirementsView(
                self.impl.build_compatible_options())

    def __init__(self, impl):
        self.impl = impl

    @property
    def num_submodels(self):
        '''
        Return the number of ActivityModel submodels contained in this model
        group. This number corresponds to the number of times the
        ActivityModelGroupBuilder which built this model group called
        add_submodel or add_reflected_submodel
        '''
        return self.impl.num_submodels

    def submodel(self, index):
        '''
        Return a submodel at a given index. Throw an `IndexError` if `index`
        is not within the integer range [0 ... num_submodels())
        '''
        return ActivityModelView(self.impl.submodel(index))

    def process_frame_with_pose_estimator(self,
                                          bgr_array,
                                          pose_estimator):
        '''
        Attempt to process an image: first pass image to the PoseEstimator
        estimator argument, which produces joint data, then pass computed
        joints to the ActivityModel` submodels. Modifies the internal state
        of estimator and of the submodels. An exception is thrown in the
        event of an error.
        '''
        return self.impl.process_frame_with_pose_estimator(bgr_array,
                                                           pose_estimator.impl)

    def process_poses(self,
                      image_width,
                      image_height,
                      pose_estimator):
        '''
        Process joint data held in a pose estimator. Pass the joint data to the
        ActivityModel submodels.

        :param image_width: int. The width of the image (in pixels) used in the
             last call to `process_frame` on the estimator passed in.
        :param image_height: int. The height of the image (in pixels) used in
            the last call to `process_frame` on the estimator passed in.
        :param pose_estimator: the PoseEstimator which just processed the
            frame.
        '''
        return self.impl.process_poses(image_width,
                                       image_height,
                                       pose_estimator.impl)

    def pose_estimator_requirements(self):
        '''
        Return the requirements a pose estimator should satisfy in order to be
        used in process_frame_with_pose_estimator
        '''
        return ActivityPoseEstimatorRequirementsView(
            self.impl.pose_estimator_requirements())
