import torch
from torch_geometric.data import Batch, Data

from retarget.model import mask_from_batch
from retarget.utils.Animation import fk_for_batch


class Tokenizer:
    def encode(self, data: Data) -> tuple[Batch, torch.Tensor]:
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
        batch = Batch.from_data_list(data)
        mask = mask_from_batch(batch)
        return batch, mask

    def decode(self, batch: Batch, d6: torch.Tensor) -> torch.Tensor:
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
            batch, d6, quater=False, device="cpu", rotations_fmt="d6"
        )
        fk_pose = fk_pose - fk_pose[..., 0:1, :]

        return fk_pose, edge_indexs
