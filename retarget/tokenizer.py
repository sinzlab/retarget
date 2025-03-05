from typing import List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from torch_geometric.data import Batch

from retarget.model import mask_from_batch
from retarget.types import RestPose
from retarget.utils.Animation import Animation, fk_for_batch
from retarget.utils.Quaternions import Quaternions, d6_2_quat


class TokenizerOutput(BaseModel):
    """
    Output container for the tokenizer.
    
    Attributes:
        batch: Batched graph data containing node features and connectivity
        mask: Boolean mask indicating valid nodes in the batch
        rest_pose: Rest pose information for the skeleton
    """
    batch: Batch
    mask: torch.Tensor
    rest_pose: RestPose
    
    class Config:
        arbitrary_types_allowed = True

class Tokenizer:
    def __init__(self, feature_list: Optional[List[str]] = None):
        self.feature_list = feature_list
        self.device = "cpu"

    def to(self, device: str = "cpu"):
        self.device = device

    def encode(self, data: Animation) -> TokenizerOutput:
        """
        Encodes a data list into a batch and a mask tensor.

        Parameters
        ----------
        data: Data
            PyTorch Geometric Data object to encode

        Returns:
            tuple containing:
                - batch: PyTorch Geometric Batch object
                - mask: Boolean mask tensor
        """
        graph_data = data.as_graph(feature_list=self.feature_list)

        batch = Batch.from_data_list(graph_data)
        mask = mask_from_batch(batch)

        batch.x = batch.x.to(self.device)
        batch.pos = batch.pos.to(self.device)
        batch.edge_index = batch.edge_index.to(self.device)
        batch.root_trajectory = batch.root_trajectory.to(self.device)
        mask = mask.to(self.device)

        rest_pose = RestPose(
            orients=data.orients,
            offsets=data.offsets,
            parents=data.parents,
            positions=batch.pos
        )

        return TokenizerOutput(batch=batch, mask=mask, rest_pose=rest_pose)

    def decode(self, batch: Batch, rotation_pred: torch.Tensor, trajectory_pred: torch.Tensor, rest_pose: RestPose) -> Animation:
        """
        Decodes a batch and a d6 tensor by computing the forward kinematics and returning the positions

        Parameters
        ----------
        batch: Batch
            PyTorch Geometric Batch object
        d6: torch.Tensor
            Tensor containing 6D rotation representations

        Returns:
            tuple containing:
                - fk_pose: Forward kinematics pose with root position subtracted
                - edge_indexs: Edge indices for the pose graph
        """
        fk_pose, edge_indexs = fk_for_batch(
            batch, rotation_pred, quater=False, device="cpu", rotations_fmt="d6"
        )
        fk_pose = fk_pose - fk_pose[..., 0:1, :]

        # Convert 6D rotations to quaternions
        quaternion_rotations = Quaternions(np.stack([d6_2_quat(d6) for d6 in rotation_pred]))
        
        # Process positions
        local_positions = (fk_pose - fk_pose[..., 0:1, :]).detach().numpy()
        global_positions = local_positions + trajectory_pred.detach().numpy()

        # Create reconstructed animation with proper scaling
        animation = Animation(
            quaternion_rotations,
            global_positions,
            rest_pose.orients,
            rest_pose.offsets,
            rest_pose.parents,
        )

        return animation
