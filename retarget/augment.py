from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data

import retarget.utils.AnimationStructure as AnimationStructure
from retarget.utils.Animation import forward_rotations
from retarget.utils.Quaternions import Quaternions, d6_2_quat, quat_2_d6


class RestPoseAugmentor:
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
        transform = torch.Tensor(Quaternions(d6_2_quat(rotations)).transforms())
        rotations_new = torch.zeros(transform.shape)

        for i, parent in enumerate(edges):
            if parent == -1:
                rotations_new[..., 0, :, :] = (
                    transform[..., 0, :, :] @ rotations_offset[0].T
                )
                continue

            rotations_new[..., i, :, :] = (
                rotations_offset[parent]
                @ transform[..., i, :, :]
                @ rotations_offset[i].T
            )

        new_rotations = torch.Tensor(
            quat_2_d6(np.array(Quaternions.from_transforms(rotations_new)))
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
            AnimationStructure.t_pose(
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


class Augmentor:
    """
    Class to handle various augmentations for data training including
    translation, scaling, and T-pose augmentation.

    Attributes:
        global_skel_scale: Global scaling factor for the skeleton
        x_translation: Translation along the x-axis
        z_translation: Translation along the z-axis
        Tpose_Augmentor: Augmentor for T-pose variations
    """

    def __init__(self, max_translation: float = 10):
        """
        Initialize the augmentor with random scaling and translation parameters.

        Parameters
        ----------
        max_translation: float
            Maximum translation distance in any direction
        """
        # Define self.global_skel_scale, to augment the skeleton size
        if np.random.rand() < 0.25:
            self.global_skel_scale = np.random.uniform(0.5, 1.5)
        else:
            self.global_skel_scale = 1.0

        # Define self.x/z_translation, to augment the root trajectory along the x and z axis
        self.x_translation = torch.rand(1) * max_translation
        self.z_translation = torch.rand(1) * max_translation
        self.rest_pose_augmentor = RestPoseAugmentor()

    def augment_x_z_translation(self, item: Data) -> Data:
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

    def augment_scale_skeleton_global(self, item: Data) -> Data:
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

        t_pose = AnimationStructure.t_pose(scaled_offsets, edges)

        item_aug.root_trajectory *= self.global_skel_scale
        item_aug.position = torch.Tensor(position)
        item_aug.x[:, 6:9] = torch.Tensor(position).clone()
        item_aug.offsets = torch.Tensor(scaled_offsets)
        item_aug.pos = torch.Tensor(t_pose)

        # Set the scaled skeletons new velocity and previous frame
        item_aug.x[:, 9:12] = torch.Tensor(position_prev).clone()
        item_aug.x[:, 12:15] = torch.Tensor(position - position_prev).clone()
        item_aug.x[:, 15:] *= self.global_skel_scale

        return item_aug

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

        item_scaled_skel = self.augment_scale_skeleton_global(item)
        encoder_item, decoder_item = self.rest_pose_augmentor(item_scaled_skel)

        encoder_item_translated = self.augment_x_z_translation(encoder_item)

        return encoder_item, decoder_item, encoder_item_translated
