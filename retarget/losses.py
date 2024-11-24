import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Union, List

from retarget.model import graph_to_batch, mask_from_batch
from retarget.utils.Animation import fk_for_batch
from retarget.utils.Quaternions_old import d6_2_rotmat


class Losses:
    """
    Losses is an object, which combines all the necessary losses for the training
    of SKiP.
    """

    def __init__(
        self,
        fk_pose: torch.Tensor,
        position: torch.Tensor,
        mask: torch.Tensor,
        fk_poses_prev: List[torch.Tensor] = None,
        position_prev: List[torch.Tensor] = None,
        children_mask: torch.Tensor = None,
        d6: torch.Tensor = None,
        d6_pred: torch.Tensor = None,
        log_var: torch.Tensor = None,
        mean: torch.Tensor = None,
        frame_time: torch.tensor = None,
        mode: str = "train",
    ):
        """
        Initialisation of the Losses Object

        Parameters
        ----------

        fk_pose: torch.Tensor
            Reconstructed joint positions using the predicted rotations.
            Shape: (batch_size, joints, 3)

        position: torch.Tensor
            Ground truth joint positions.
            Shape: (batch_size, joints, 3)

        mask: torch.Tensor
           Mask, defining which joints are padded and which not
           Shape: (batch_size, joints, 3)

        fk_poses_prev: List[torch.Tensor]
            List of the first, second and third previous reconstructed joint positions using the predicted rotations.
            First entry is one previous frame, second entry two previous frames and third entry the three previous frames
            Lenghts of List: 3
            Shape of each Entry: (batch_size, joints, 3)

        position_prev: List[torch.Tensor]
            List of the first, second and third previous ground truth joint positions.
            First entry is one previous frame, second entry two previous frames and third entry the three previous frames
            Lenghts of List: 3
            Shape of each Entry: (batch_size, joints, 3)

        children_mask: torch.Tensor
           ???
           Shape: ???

        d6: torch.Tensor
            Tensor of the 6d representation for every joint
            Shape: (batch_size, joints, 6)

        d6_pred: torch.Tensor
            Tensor of the predicted 6d representation for every joint
            Shape: (batch_size, joints, 6)

        log_var: torch.Tensor
            Log variance, where the variance is used for the latent space vector sampling
            Shape: (batch_size, N_latent, N_latent)
            N_latent is here the length of the latent space vector

        mean: torch.Tensor
            Mean used to sample the latent space vector
            Shape: (batch_size, N_latent)

        frame_time: torch.Tensor
            Time difference between two frames
            Shape: (batch_size, 1, 1)

        mode: str
            Mode, to define if the losses if for training or validation

        Returns
        -------

        None
        """

        self.fk_pose = fk_pose
        self.position = position
        self.mask = mask
        self.mode = mode

        if self.mode == "train":

            # Check that all values used for train is not None
            assert fk_poses_prev is not None
            assert children_mask is not None
            assert d6 is not None
            assert d6_pred is not None
            assert log_var is not None
            assert frame_time is not None
            assert position_prev is not None
            assert mean is not None

            self.fk_poses_prev = fk_poses_prev
            self.position_prev = position_prev
            self.d6 = d6
            self.d6_pred = d6_pred
            self.log_var = log_var
            self.mean = mean
            self.children_mask = children_mask
            self.frame_time = frame_time

            self.v_t = self.get_velocity(idx=0, vel_pred=False)
            self.v_t_pred = self.get_velocity(idx=0, vel_pred=True)
            self.v_t_minus_1_pred = self.get_velocity(idx=1, vel_pred=True)
            self.v_t_minus_2_pred = self.get_velocity(idx=2, vel_pred=True)
            self.a_t_pred = self.get_acceleration(idx=0)
            self.a_t_minus_1_pred = self.get_acceleration(idx=1)

    def get_velocity(
        self,
        idx: int,
        vel_pred: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the velocity (Frame changing rate) of the (predicted-) frame t-idx.
        v_t = (p_{t} - p_{t-1}) / frame_time
        --> To get v^{t}, choose idx = 0,
        --> v^{t-1}, choose idx = 1, ....

        Parameters
        ----------

        idx: int
            Define for which timestep to get the velocity

        vel_pred: bool
            Define if we want the predicted velocity or the ground truth

        Returns:
            torch.Tensor
            Shape: (batch_size, joints, 3)
        """

        # Make sure that the idx does not go over 2, since only 3 offset frames are considered
        assert idx < 3

        if (idx == 0) and not vel_pred:
            v_t = (self.position - self.position_prev[0]) / self.frame_time
            return v_t
        elif (idx == 0) and vel_pred:
            v_t_pred = (self.fk_pose - self.fk_poses_prev[idx]) / self.frame_time
            return v_t_pred
        elif (idx > 0) and vel_pred:
            v_t_pred_prev = (
                self.fk_poses_prev[idx - 1] - self.fk_poses_prev[idx]
            ) / self.frame_time
            return v_t_pred_prev
        else:
            raise ValueError("Trying to get velocity of idx > 0: Not possible")

    def get_acceleration(
        self,
        idx: int,
    ) -> torch.Tensor:
        """
        Calculate the acceleration of the predicted frame t-idx.
        a_t = (v_{t} - v_{t-1}) / frame_time
        --> To get a^{t}, choose idx = 0,
        --> a^{t-1}, choose idx = 1.

        Parameters
        ----------

        idx: int
            Define for which timestep to get the predicted acceleration

        Returns
        -------

        torch.Tensor
            Shape: (batch_size, joints, 3)
        """

        assert idx < 2

        if idx == 0:
            a_t_pred = (self.v_t_pred - self.v_t_minus_1_pred) / self.frame_time
            return a_t_pred
        if idx == 1:
            a_t_minus_1_pred = (
                self.v_t_minus_1_pred - self.v_t_minus_2_pred
            ) / self.frame_time
            return a_t_minus_1_pred

    @property
    def losses(
        self,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the losses used for the training/validation
        Return the losses as a dictionary

        Parameters
        ----------

        None

        Returns
        -------

        Dict[str, torch.Tensor]
        """

        recn_loss = (
            torch.norm(self.position - self.fk_pose, dim=-1) * self.mask
        ).sum() / self.mask.sum()

        if self.mode == "train":
            vel_loss = (
                torch.norm(self.v_t - self.v_t_pred, dim=-1) * self.mask
            ).sum() / self.mask.sum()
            acc_loss = (
                torch.norm(self.a_t_pred - self.a_t_minus_1_pred, dim=-1) * self.mask
            ).sum() / self.mask.sum()
            recn_loss_root_children = (
                torch.norm(self.position - self.fk_pose, dim=-1) * self.children_mask
            ).sum() / self.children_mask.sum()
            d6_loss = (
                torch.norm(self.d6 - self.d6_pred, dim=-1) * self.mask
            ).sum() / self.mask.sum()
            kl_loss = -0.5 * torch.sum(
                1 + self.log_var - self.mean.pow(2) - self.log_var.exp()
            )
            return {
                "recn_loss": recn_loss,
                "vel_loss": vel_loss,
                "acc_loss": acc_loss,
                "recn_loss_root_children": recn_loss_root_children,
                "d6_loss": d6_loss,
                "kl_loss": kl_loss,
            }

        return {"recn_loss": recn_loss}
