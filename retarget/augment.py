from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data, Batch

import retarget.utils.AnimationStructure as AnimationStructure
from retarget.utils.Animation import forward_rotations
from retarget.utils.Quaternions import Quaternions, d6_2_rotmat, rotmat_2_d6
from retarget.types import AnimationData, AnimationBatch
from retarget.model import mask_from_batch, graph_to_batch

from typing import Union

def augment_global_skeleton(item: AnimationData, max_scale: float = 1.1, min_scale: float = 0.9, probability: float = 0.25) -> AnimationData:
    item_augment = item.clone()

    if np.random.rand() < probability:
        global_skel_scale = np.random.uniform(min_scale, max_scale)
    else:
        global_skel_scale = 1.0

    scaled_offsets = item_augment.rest_pose.offsets.numpy().copy() * global_skel_scale

    rotation = item_augment.rotations.numpy()
    rotation_prev = item_augment.rotations_prev.numpy()
    
    parents = item_augment.rest_pose.parents.numpy()
    edges = item_augment.rest_pose.edge_index.numpy().T

    position = forward_rotations(
        parents,
        scaled_offsets,
        Quaternions(rotation[None, ...])
    )[0]
    position_prev = forward_rotations(
        parents,
        scaled_offsets,
        Quaternions(rotation_prev[None, ...])
    )[0]

    scaled_rest_pose = AnimationStructure.rest_pose(scaled_offsets, edges)

    item_augment.rest_pose.offsets = torch.Tensor(scaled_offsets)
    item_augment.rest_pose.positions = torch.Tensor(scaled_rest_pose)
    item_augment.positions = torch.Tensor(position)
    item_augment.position_prev = torch.Tensor(position_prev)

    default_feature_list = [
        "d6",               # 6D rotation representation
        "position",         # Joint positions
        "position_prev",    # Previous joint positions
        "velocity",         # Joint velocities
        "root_trajectory"   # Root joint trajectory
    ]

    item_augment.build_feature_vector(default_feature_list, position=item_augment.positions, position_prev=item_augment.position_prev, d6=item_augment.d6, root_trajectory=item_augment.root_trajectory, velocity=item_augment.velocity)

    return item_augment

def augment_rest_pose(item: Union[AnimationData, AnimationBatch], max_large_angle: float = 0, max_small_angle: float = 0, probability: float = 0.5) -> Union[AnimationData, AnimationBatch]:
    item_augment = item.clone()

    # check if item is a batch
    if isinstance(item, AnimationBatch):
        motion_list = item_augment.to_data_list()

        augmented_motion_list = []
        for motion in motion_list:
            augmented_motion = _rest_pose_augmentor(motion, max_large_angle, max_small_angle, probability)
            augmented_motion_list.append(augmented_motion)

        item_augment = AnimationBatch.from_data_list(augmented_motion_list)
    else:
        item_augment = _rest_pose_augmentor(item, max_large_angle, max_small_angle, probability)

    return item_augment

def _rest_pose_augmentor(item: AnimationData, max_large_angle: float = 0, max_small_angle: float = 0, probability: float = 0.5) -> AnimationData:
    item_augment = item.clone()

    num_joints = item_augment.rest_pose.offsets.shape[0]

    if np.random.rand() < probability:
        small_euler_angles = (
            np.random.rand(num_joints, 3) * 2 - 1
        ) * max_small_angle
    else:
        small_euler_angles = np.zeros((num_joints, 3))

    if np.random.rand() < probability:
        large_euler_angles = (
            np.random.rand(num_joints, 3) * 2 - 1
        ) * max_large_angle
    else:
        large_euler_angles = np.zeros((num_joints, 3))

    large_rotations = R.from_euler(
        "XYZ", large_euler_angles, degrees=True
    ).as_matrix()
    small_rotations = R.from_euler(
        "XYZ", small_euler_angles, degrees=True
    ).as_matrix()

    rotations_offset = torch.Tensor(large_rotations @ small_rotations)

    parent_indices = item_augment.rest_pose.edge_index[0]
    child_indices = item_augment.rest_pose.edge_index[1]

    unique = torch.unique(child_indices).equal(child_indices)

    if unique:
        item_augment.rest_pose.offsets[child_indices] = (
            rotations_offset[parent_indices] @ item_augment.rest_pose.offsets[child_indices][..., None]
        ).squeeze()
    else:
        for parent, children in item_augment.rest_pose.edge_index.T:
            item_augment.rest_pose.offsets[children] = (
                rotations_offset[parent] @ item_augment.rest_pose.offsets[children][..., None]
            ).squeeze()

    transform = torch.Tensor(d6_2_rotmat(item_augment.d6))
    rotations_new = torch.zeros(transform.shape)
    
    # Handle the root node (index 0)
    rotations_new[..., 0, :, :] = transform[..., 0, :, :] @ rotations_offset[0].T
    # Handle all other nodes in a vectorized way
    # For each child node, apply the parent's rotation offset, then the original transform, then the child's inverse rotation offset
    child_indices = torch.arange(1, item_augment.rest_pose.parents.shape[0])
    parent_indices = item_augment.rest_pose.parents[1:]

    rotations_new[..., child_indices, :, :] = (
        rotations_offset[parent_indices]
        @ transform[..., child_indices, :, :]
        @ rotations_offset[child_indices].transpose(1, 2)
    )

    item_augment.d6 = torch.Tensor(
        rotmat_2_d6(rotations_new.reshape(-1, 3, 3))
    ).reshape(rotations_new.shape[0], rotations_new.shape[1], -1)

    item_augment.rest_pose.positions = torch.Tensor(
        AnimationStructure.rest_pose(
            item_augment.rest_pose.offsets.numpy(), item_augment.rest_pose.edge_index.numpy().T
        )
    )

    default_feature_list = [
        "d6",               # 6D rotation representation
        "position",         # Joint positions
        "position_prev",    # Previous joint positions
        "velocity",         # Joint velocities
        "root_trajectory"   # Root joint trajectory
    ]

    item_augment.build_feature_vector(default_feature_list, position=item_augment.positions, position_prev=item_augment.position_prev, d6=item_augment.d6, root_trajectory=item_augment.root_trajectory, velocity=item_augment.velocity)

    return item_augment


def augment_root_trajectory(item: AnimationData, max_translation: float = 10, probability: float = 1) -> AnimationData:
    item_augment = item.clone()

    if np.random.rand() < probability:
        x_translation = torch.rand(1) * max_translation
        z_translation = torch.rand(1) * max_translation

        item_augment.root_trajectory[..., 0] += x_translation
        item_augment.root_trajectory[..., 2] += z_translation

    default_feature_list = [
        "d6",               # 6D rotation representation
        "position",         # Joint positions
        "position_prev",    # Previous joint positions
        "velocity",         # Joint velocities
        "root_trajectory"   # Root joint trajectory
    ]

    item_augment.build_feature_vector(default_feature_list, position=item_augment.positions, position_prev=item_augment.position_prev, d6=item_augment.d6, root_trajectory=item_augment.root_trajectory, velocity=item_augment.velocity)

    return item_augment


class Augmentor:
    """
    Base class for all augmentors.
    """

    def reset(self):
        """
        Reset the augmentor to new random parameters.
        """
        pass
    
    def __call__(self, item: Data) -> Data:
        """
        Apply the augmentor to the input data.
        """
        pass

class RestPoseAugmentor(Augmentor):
    """
    Class for augmenting T-pose data by applying random rotations to the skeleton.

    Attributes:
        max_large_angle: Maximum angle in degrees for large rotations
        max_small_angle: Maximum angle in degrees for small rotations
    """

    def __init__(self, max_large_angle: float = 0, max_small_angle: float = 0):
        """
        Initialize the T-pose augmentor.

        Parameters
        ----------
        max_large_angle: float
            Maximum angle in degrees for large rotations
        max_small_angle: float
            Maximum angle in degrees for small rotations
        """
        self.max_large_angle = max_large_angle
        self.max_small_angle = max_small_angle

        self.seed = np.random.randint(0, 1000000)

    def reset(self):
        """
        Reset the augmentor to new random parameters.
        """
        self.seed = np.random.randint(0, 1000000)

    def random_rotation_generator(self, offsets_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Generate random rotation matrices.

        Parameters
        ----------
        offsets_shape: Tuple[int, int]
            Shape of the offsets tensor (num_joints, 3)

        Returns
        -------
        torch.Tensor
            Random rotation matrices as a tensor
        """
        np.random.seed(self.seed)
        if np.random.rand() < 0.5:
            small_euler_angles = (
                np.random.rand(offsets_shape[0], offsets_shape[1]) * 2 - 1
            ) * self.max_small_angle
        else:
            small_euler_angles = np.zeros((offsets_shape[0], offsets_shape[1]))

        if np.random.rand() < 0.5:
            large_euler_angles = (
                np.random.rand(offsets_shape[0], offsets_shape[1]) * 2 - 1
            ) * self.max_large_angle
        else:
            large_euler_angles = np.zeros((offsets_shape[0], offsets_shape[1]))

        large_rotations = R.from_euler(
            "XYZ", large_euler_angles, degrees=True
        ).as_matrix()
        small_rotations = R.from_euler(
            "XYZ", small_euler_angles, degrees=True
        ).as_matrix()

        rotations_offset = torch.Tensor(large_rotations @ small_rotations)
        return rotations_offset

    def create_new_rotations(
        self, rotations_offset: torch.Tensor, rotations: torch.Tensor, edges: List[int]
    ) -> torch.Tensor:
        """
        Create new rotations by applying offset rotations.

        Parameters
        ----------
        rotations_offset: torch.Tensor
            Offset rotation matrices
        rotations: torch.Tensor
            Original rotation matrices in 6D representation
        edges: List[int]
            Parent-child relationships in the skeleton

        Returns
        -------
        torch.Tensor
            New rotations in 6D representation
        """
        transform = torch.Tensor(d6_2_rotmat(rotations))
        rotations_new = torch.zeros(transform.shape)
    
        rotations_new = torch.zeros(transform.shape)
        
        # Handle the root node (index 0)
        rotations_new[..., 0, :, :] = transform[..., 0, :, :] @ rotations_offset[0].T
        # Handle all other nodes in a vectorized way
        # For each child node, apply the parent's rotation offset, then the original transform, then the child's inverse rotation offset
        child_indices = torch.arange(1, edges.shape[0])
        parent_indices = edges[1:]

        rotations_new[..., child_indices, :, :] = (
            rotations_offset[parent_indices]
            @ transform[..., child_indices, :, :]
            @ rotations_offset[child_indices].transpose(1, 2)
        )

        new_rotations = torch.Tensor(
            rotmat_2_d6(rotations_new)
        )

        return new_rotations

    def offset_rotation(
        self, offset: np.ndarray, rotations: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotations to joint offsets.

        Parameters
        ----------
        offset: np.ndarray
            Joint offsets
        rotations: torch.Tensor
            Rotation matrices
        edges: torch.Tensor
            Edge indices representing skeleton connectivity

        Returns
        -------
        torch.Tensor
            Rotated offsets
        """

        parent_indices = edges[0]
        child_indices = edges[1]

        # check if child_indices are unique
        unique = torch.unique(child_indices).equal(child_indices)

        if unique:
            offset[child_indices] = (
                rotations[parent_indices] @ offset[child_indices][..., None]
            ).squeeze()
        else:
            for pair in edges.T:
                parent, children = pair
            offset[children] = (
                rotations[parent] @ offset[children][..., None]
            ).squeeze()

        return torch.Tensor(offset)

    def augment_data(self, item: Data) -> Data:
        """
        Augment a single data item by applying random rotations.

        Parameters
        ----------
        item: Data
            Data item containing skeleton information

        Returns
        -------
        Data
            Augmented data item
        """
        item_augment = item.clone()

        rnd_rotations = self.random_rotation_generator(item_augment.offsets.shape)

        item_augment.offsets = self.offset_rotation(
            item_augment.offsets, rnd_rotations, item.edge_index
        )

        item_augment.d6 = self.create_new_rotations(
            rnd_rotations, item_augment.d6, item_augment.parents
        )
        
        item_augment.pos = torch.Tensor(
            AnimationStructure.rest_pose(
                item_augment.offsets.numpy(), item_augment.edge_index.numpy().T
            )
        )

        return item_augment

    def __call__(self, item: Data) -> Tuple[Data, Data]:
        """
        Generate two augmented versions of the input data.

        Parameters
        ----------
        item: Data
            Original data item

        Returns
        -------
        Tuple[Data, Data]
            Tuple containing two independently augmented versions of the input data:
                - encoder_item: First augmented item for the encoder
                - decoder_item: Second augmented item for the decoder
        """
        encoder_item = self.augment_data(item)
        decoder_item = self.augment_data(item)

        return encoder_item, decoder_item
    

class GlobalSkeletonAugmentor(Augmentor):
    """
    Class for augmenting the global skeleton by scaling and translating the skeleton.
    """
    
    def __init__(self):
        """
        Initialize the global skeleton augmentor with random scaling and translation parameters.
        """
        self.reset()

    def reset(self):
        """
        Reset the global skeleton augmentor to its initial state.
        """
        if np.random.rand() < 0.25:
            self.global_skel_scale = np.random.uniform(0.5, 1.5)
        else:
            self.global_skel_scale = 1.0

        
    def __call__(self, item: Data) -> Data:
        """
        Scale the skeleton globally by the predefined scaling factor.

        Parameters
        ----------
        item: Data
            Torch geometric graph data, storing the pose information for one frame

        Returns
        -------
        item_aug: Data
            Torch geometric graph data with scaled skeleton
        """

        item_aug = item.clone()
        scaled_offsets = item_aug.offsets.numpy().copy() * self.global_skel_scale
        scaled_offsets_prev = scaled_offsets.copy()

        # For original frame
        parents = item_aug.parents.numpy()
        rotation = item_aug.rotation.numpy()
        rotation_prev = item_aug.rotation_prev.numpy()
        edges = item_aug.edge_index.numpy().T

        position = forward_rotations(
            parents, scaled_offsets, Quaternions(rotation[None, ...])
        )[0]

        position_prev = forward_rotations(
            parents, scaled_offsets_prev, Quaternions(rotation_prev[None, ...])
        )[0]

        rest_pose = AnimationStructure.rest_pose(scaled_offsets, edges)

        item_aug.root_trajectory *= self.global_skel_scale
        item_aug.position = torch.Tensor(position)
        item_aug.x[:, 6:9] = torch.Tensor(position).clone()
        item_aug.offsets = torch.Tensor(scaled_offsets)
        item_aug.pos = torch.Tensor(rest_pose)

        # Set the scaled skeletons new velocity and previous frame
        item_aug.x[:, 9:12] = torch.Tensor(position_prev).clone()
        item_aug.x[:, 12:15] = torch.Tensor(position - position_prev).clone()
        item_aug.x[:, 15:] *= self.global_skel_scale

        return item_aug

class XZTranslationAugmentor(Augmentor):
    """
    Class for augmenting the x and z translations of the skeleton.
    """
    
    def __init__(self, max_translation: float = 10):
        self.x_translation = torch.rand(1) * max_translation
        self.z_translation = torch.rand(1) * max_translation

    def __call__(self, item: Data) -> Data:
        """
        Augment the root trajectory by translating along x and z axes.

        Parameters
        ----------
        item: Data
            Torch geometric graph data, storing the pose information for one frame

        Returns
        -------
        item_aug: Data
            Torch geometric graph data with augmented x and z root trajectories
        """

        # Augment root trajectory
        item_aug = item.clone()
        item_aug.root_trajectory[:, 0] += self.x_translation
        item_aug.root_trajectory[:, 2] += self.z_translation
        item_aug.x[:, 15] += self.x_translation
        item_aug.x[:, 17] += self.z_translation

        return item_aug

class Augmentions:
    """
    Class to handle various augmentations for data training including
    translation, scaling, and T-pose augmentation.

    Attributes:
        augmentors: List of augmentors to apply
    """

    def __init__(self, rest_pose_augmentor: Optional[RestPoseAugmentor] = None, global_skeleton_augmentor: Optional[GlobalSkeletonAugmentor] = None, x_z_translation_augmentor: Optional[XZTranslationAugmentor] = None):
        """
        Initialize the augmentor with random scaling and translation parameters.

        Parameters
        ----------
        rest_pose_augmentor: RestPoseAugmentor
            Augmentor for T-pose augmentation
        global_skeleton_augmentor: GlobalSkeletonAugmentor
            Augmentor for global skeleton augmentation
        xz_translation_augmentor: XZTranslationAugmentor
            Augmentor for x and z translation augmentation
        """
        self.rest_pose_augmentor = rest_pose_augmentor
        self.global_skeleton_augmentor = global_skeleton_augmentor
        self.x_z_translation_augmentor = x_z_translation_augmentor

    def reset(self):
        """
        Reset the augmentor to new random parameters.
        """
        if self.rest_pose_augmentor:
            self.rest_pose_augmentor.reset()
        if self.global_skeleton_augmentor:
            self.global_skeleton_augmentor.reset()
        if self.x_z_translation_augmentor:
            self.x_z_translation_augmentor.reset()

    def __call__(self, item: Data) -> Tuple[Data, Data, Data]:
        """
        Apply multiple augmentations to the input data.

        Parameters
        ----------
        item: Data
            Torch geometric graph data, storing the pose information for one frame

        Returns
        -------
        Tuple[Data, Data, Data]
            Tuple containing:
                encoder_item: Data
                    Augmented item for the encoder with scaled skeleton and T-pose augmentation

                decoder_item: Data
                    Augmented item for the decoder with scaled skeleton and T-pose augmentation

                encoder_item_translated: Data
                    Encoder item with additional x-z translation augmentation
        """

        if self.global_skeleton_augmentor:
            item_scaled_skel = self.global_skeleton_augmentor(item)

        if self.rest_pose_augmentor:
            encoder_item, decoder_item = self.rest_pose_augmentor(item_scaled_skel)

        if self.x_z_translation_augmentor:
            encoder_item_translated = self.x_z_translation_augmentor(encoder_item)

        if self.rest_pose_augmentor and self.global_skeleton_augmentor:
            return encoder_item, decoder_item, encoder_item_translated
        elif self.rest_pose_augmentor:
            return encoder_item, decoder_item
        elif self.global_skeleton_augmentor:
            return item_scaled_skel


AUGMENTORS = {
    "rest_pose_augmentor": RestPoseAugmentor,
    "global_skeleton_augmentor": GlobalSkeletonAugmentor,
    "x_z_translation_augmentor": XZTranslationAugmentor,
}

def get_augmentors(config: Dict[str, Any]) -> List[Augmentor]:
    """
    Get the augmentors from the config.
    """

    augmentors = []

    for key, value in config["dataset"]["augmentors"].items():
        if value:
            augmentors.append(AUGMENTORS[key](**value))
        else:
            augmentors.append(AUGMENTORS[key]())

    return augmentors