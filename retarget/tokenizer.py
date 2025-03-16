from typing import List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from torch_geometric.data import Batch, Data

from retarget.model import mask_from_batch
from retarget.types import RestPose, AnimationData
from retarget.utils.Animation import Animation, fk_for_batch, forward_rotations_torch
from retarget.utils.Quaternions import Quaternions, d6_2_quat, quat_2_d6, d6_2_rotmat

class TokenizerOutput(BaseModel):
    """
    Output container for the tokenizer.
    
    Attributes:
        batch: Batched graph data containing node features and connectivity
        mask: Boolean mask indicating valid nodes in the batch
        rest_pose: Rest pose information for the skeleton
    """
    motion: AnimationData
    mask: torch.Tensor
    rest_pose: Data
    
    class Config:
        arbitrary_types_allowed = True

class TokenizerNew:
    def __init__(self, feature_list: Optional[List[str]] = None):
        self.feature_list = feature_list

        if feature_list is None:
            self.feature_list = [
                "d6",
                "position",
                "position_prev",
                "velocity",
                "root_trajectory"
            ]

        self.device = "cpu"

    def to(self, device: str = "cpu"):
        self.device = device

    def encode(self, animation: Animation, stride: int = 1, frame_time: float = 1/30) -> List[AnimationData]:
        rotations = animation.rotations.qs
        positions = animation.positions

        n_frames = rotations.shape[0]
        n_joints = rotations.shape[1]
        
        d6 = quat_2_d6(rotations.reshape(-1, 4)).reshape(n_frames, n_joints, -1)

        root_trajectory = np.zeros_like(positions)
        root_trajectory[..., 0, :] += positions[..., 0, :]

        # subtract root trajectory from positions
        positions = positions - root_trajectory[..., :1, :]

        position_prev = positions.copy()
        position_prev[:, 0, :] = positions[:, 0, :]
        position_prev[:, 1:, :] = positions[:, :-1, :]

        rotations_prev = rotations.copy()
        rotations_prev[:, 0, :] = rotations[:, 0, :]
        rotations_prev[:, 1:, :] = rotations[:, :-1, :]

        velocity = positions - position_prev

        rotations = torch.Tensor(rotations)
        rotations_prev = torch.Tensor(rotations_prev)
        positions = torch.Tensor(positions)
        position_prev = torch.Tensor(position_prev)
        d6 = torch.Tensor(d6)
        fps = torch.Tensor([1/frame_time])
        velocity = torch.Tensor(velocity)
        root_trajectory = torch.Tensor(root_trajectory)

        feature_vector = animation.build_feature_vector(self.feature_list, position=positions, position_prev=position_prev, d6=d6, root_trajectory=root_trajectory, velocity=velocity)

        rest_pose = torch.Tensor(animation.rest_pose)
        offsets = torch.Tensor(animation.offsets)
        parents = torch.LongTensor(animation.parents)
        edges = torch.LongTensor(animation.edges.T)

        rest_pose_graph = Data(x=rest_pose, edge_index=edges, offsets=offsets, parents=parents, positions=rest_pose)
        data_item = AnimationData(features=feature_vector, rotations=rotations, rotations_prev=rotations_prev, positions=positions, position_prev=position_prev, d6=d6, fps=fps, root_trajectory=root_trajectory, velocity=velocity, rest_pose=rest_pose_graph)

        mask = torch.ones((n_joints, 1), dtype=torch.bool)

        return TokenizerOutput(motion=data_item, mask=mask, rest_pose=rest_pose_graph)
    
    def get_feature_vector(self, d6, traj, rest_pose):
        rest_pose_list = rest_pose.to_data_list()

        feature_vectors = []
        for i in range(len(rest_pose_list)):
            positions_retargetted = forward_rotations_torch(
                rest_pose_list[i].edge_index.T,
                rest_pose_list[i].offsets,
                rotations=d6_2_rotmat(d6[i]),
                device=self.device
            )

            position_prev_retargetted = positions_retargetted.clone()
            position_prev_retargetted[..., 0, :] = positions_retargetted[..., 0, :]
            position_prev_retargetted[..., 1:, :] = positions_retargetted[..., :-1, :]

            velocities_retargetted = positions_retargetted - position_prev_retargetted

            feature_vector = torch.cat([
                d6[i],
                positions_retargetted,
                position_prev_retargetted,
                velocities_retargetted,
                traj[i].repeat(1, positions_retargetted.shape[1], 1)
            ], dim=-1)

            feature_vectors.append(feature_vector)

        feature_vector = torch.stack(feature_vectors, dim=0)

        return feature_vector
    
    def get_positions(self, d6, rest_pose):
        if isinstance(rest_pose, Batch):
            rest_pose = rest_pose.to_data_list()
        else:
            rest_pose = [rest_pose]

        positions = []
        for i in range(len(rest_pose)):
            positions.append(forward_rotations_torch(
                rest_pose[i].edge_index.T,
                rest_pose[i].offsets,
                rotations=d6_2_rotmat(d6[i]),
                device=self.device))

        return torch.stack(positions, dim=0)

    def decode(self, rotations, trajectory, rest_pose):
        positions = self.get_positions(rotations, rest_pose)
        quaternion_rotations = Quaternions(np.stack([d6_2_quat(d6) for d6 in rotations]))
        global_positions = positions + trajectory

        animation = Animation(
            quaternion_rotations,
            global_positions,
            rest_pose.orients,
            rest_pose.offsets,
            rest_pose.parents,
        )

        return animation

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
