from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel

from retarget.utils.Quaternions import d6_2_rotmat


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