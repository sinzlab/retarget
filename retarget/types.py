from dataclasses import dataclass
from typing import Optional, Union, List

import numpy as np
import torch
from pydantic import BaseModel

from retarget.utils.Quaternions import d6_2_rotmat
from torch_geometric.data import Data, Batch

def reshape_for_losses(tensor: torch.Tensor, num_sequences: int, consecutive_frames: int, num_joints: int, last_dims: tuple[int, ...] = (-1,)) -> torch.Tensor:
    """
    Reshape the tensor for the losses

    Parameters
    ----------

    tensor: torch.Tensor
        Tensor to reshape

    num_sequences: int
        Number of sequences

    consecutive_frames: int
        Number of consecutive frames

    num_joints: int
        Number of joints

    last_dims: tuple[int, ...]
        Last dimensions

    Returns
    -------

    torch.Tensor
    """
    return tensor.reshape(
        num_sequences,
        consecutive_frames,
        num_joints,
        *last_dims
    )

def create_children_mask(
        position: torch.Tensor,
        edge_indexs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a children mask for the joints
        """
        children_mask: torch.Tensor = torch.zeros(
            position.shape[0], position.shape[1], device=position.device
        )
        item, idx = torch.where(edge_indexs[:, :, 0] == 0)
        children_mask[item, edge_indexs[item, idx, 1]] = 1

        return children_mask

@dataclass
class AnimationData:
    position: torch.Tensor
    d6: torch.Tensor
    root_trajectory: torch.Tensor
    mask: torch.Tensor
    fps: float
    edge_indexs: Optional[torch.Tensor] = None
    children_mask: Optional[torch.Tensor] = None
    rotmat: Optional[torch.Tensor] = None

    def prepare(self, num_sequences: int, consecutive_frames: int, num_joints: int) -> None:
        if self.edge_indexs is not None:
            self.children_mask = create_children_mask(
                self.position,
                self.edge_indexs
            )
            self.children_mask = reshape_for_losses(self.children_mask, num_sequences, consecutive_frames, num_joints, last_dims=())

        self.position = reshape_for_losses(self.position, num_sequences, consecutive_frames, num_joints)
        self.d6 = reshape_for_losses(self.d6, num_sequences, consecutive_frames, num_joints)
        self.mask = reshape_for_losses(self.mask, num_sequences, consecutive_frames, num_joints, last_dims=())

        rotmat = d6_2_rotmat(
            torch.flatten(self.d6.clone(), start_dim=0, end_dim=-2)
        )
        self.rotmat = reshape_for_losses(rotmat, num_sequences, consecutive_frames, num_joints, last_dims=(3, 3))

@dataclass
class EncoderOutputs:
    z_pose: torch.Tensor
    z_root_trajectory: torch.Tensor
    z_pose_augmented: Optional[torch.Tensor] = None
    z_root_trajectory_augmented: Optional[torch.Tensor] = None

class RestPose(BaseModel):
    """
    Represents the rest pose of a skeleton.
    
    Attributes:
        orients: Initial orientations of joints
        offsets: Offset vectors between joints
        parents: Parent indices for each joint
        positions: 3D positions of joints in rest pose
    """
    orients: np.ndarray
    offsets: np.ndarray
    parents: np.ndarray
    positions: torch.Tensor
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class AnimationData:
    """
    Data item for the animation.
    """
    rotations: torch.Tensor
    rotations_prev: torch.Tensor
    positions: torch.Tensor
    position_prev: torch.Tensor
    d6: torch.Tensor
    fps: Union[float, List[float]]
    root_trajectory: torch.Tensor
    velocity: torch.Tensor
    rest_pose: Data
    features: Optional[torch.Tensor] = None

    def build_feature_vector(self, feature_list, **feature_dict):
        """
        Build a feature vector from the animation data.
        """
        features_list = [feature_dict[feature_name] for feature_name in feature_list]
        self.features = torch.cat(features_list, dim=-1)

    def clone(self):
        """
        Create a deep copy of the animation data.
        """
        return self.__class__(
            rotations=self.rotations.clone(),
            rotations_prev=self.rotations_prev.clone(),
            positions=self.positions.clone(),
            position_prev=self.position_prev.clone(),
            d6=self.d6.clone(),
            fps=self.fps,
            root_trajectory=self.root_trajectory.clone(),
            velocity=self.velocity.clone(),
            rest_pose=self.rest_pose.clone(),
            features=self.features.clone() if self.features is not None else None,
        )

    
@dataclass
class AnimationBatch(AnimationData):
    def to_data_list(self) -> List["AnimationData"]:
        """convert into a list of AnimationData objects"""
        rest_pose_list = self.rest_pose.to_data_list()

        individual_joints = [skel.positions.shape[-2] for skel in rest_pose_list]
        
        data_list = [dict() for _ in range(len(individual_joints))]

        # print(self.root_trajectory[0])
        for key in self.__dict__:
            if key == 'rest_pose':
                continue

            attr = getattr(self, key)

            for i, joint_count in enumerate(individual_joints):
                data_list[i][key] = attr[i, :, :joint_count]
        
        output_list = []
        for i, data in enumerate(data_list):
            data['rest_pose'] = rest_pose_list[i]
            output_list.append(AnimationData(**data))

        return output_list
        
    
    @classmethod
    def from_data_list(cls, data_list: List["AnimationData"]) -> "AnimationBatch":
        """convert a list of AnimationData objects into a single AnimationData object"""
        joint_counts = [b.features.shape[-2] for b in data_list]
        max_joints = max(joint_counts)
        batch_size = len(data_list)
        consequtive_frames = data_list[0].features.shape[0]

        mask = torch.zeros(batch_size, consequtive_frames, max_joints, dtype=torch.bool)

        dict = {}
        for key in data_list[0].__dict__:
            if key == 'rest_pose':
                continue

            dict[key] = torch.zeros(batch_size, consequtive_frames, max_joints, *getattr(data_list[0], key).shape[2:])

            for i, (item, joint_count) in enumerate(zip(data_list, joint_counts)):
                dict[key][i, :, :joint_count] = getattr(item, key)
                mask[i, :, :joint_count] = 1

        rest_pose = Batch.from_data_list([b.rest_pose for b in data_list])

        return AnimationBatch(
            **dict,
            rest_pose=rest_pose,
        ), rest_pose, mask
