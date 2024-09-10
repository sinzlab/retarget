from torch_geometric.data import Data, Batch
import torch

from retarget.model import mask_from_batch
from retarget.utils.Animation import fk_for_batch

class Tokenizer:
    def encode(self, data: Data) -> tuple[Batch, torch.Tensor]:
        """
        Encodes a data list into a batch and a mask tensor.
        """
        batch = Batch.from_data_list(data)
        mask = mask_from_batch(batch)
        return batch, mask
    
    def decode(self, batch: Batch, d6: torch.Tensor) -> torch.Tensor:
        """
        Decodes a batch and a d6 tensor by computing the forward kinematics and returning the positions
        """
        fk_pose, edge_indexs = fk_for_batch(batch, d6, quater=False, device='cpu', rotations_fmt='d6')
        fk_pose = fk_pose - fk_pose[..., 0:1, :]

        return fk_pose, edge_indexs