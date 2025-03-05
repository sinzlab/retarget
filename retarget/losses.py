from typing import Dict, List, Optional

import torch

from retarget.types import AnimationData, EncoderOutputs
from retarget.utils.Quaternions import d6_2_rotmat


def reconstruction_loss(
        position_gt: torch.Tensor,
        position_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss for all joint positions

        Parameters
        ----------

        position_gt: torch.Tensor
            Ground truth joint positions.
            Shape: (batch_size, frames, joints, 3)

        position_pred: torch.Tensor
            Predicted joint positions.
            Shape: (batch_size, frames, joints, 3)

        mask: Optional[torch.Tensor] = None
            Mask, defining which joints are padded and which not
            Shape: (batch_size, frames, joints)

        Returns
        -------

        torch.Tensor
        """
        if mask is None:
            mask = torch.ones(position_gt.shape[:-1])

        recn_loss = (
            torch.norm(position_gt - position_pred, dim=-1) * mask
        ).sum() / mask.sum()

        return recn_loss

def reconstruction_root_children_loss(
        position_gt: torch.Tensor,
        position_pred: torch.Tensor,
        children_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss for joint positions,
        where the joint is directly connected to the root

        Parameters
        ----------

        position_gt: torch.Tensor
            Ground truth joint positions.
            Shape: (batch_size, frames, joints, 3)

        position_pred: torch.Tensor
            Predicted joint positions.
            Shape: (batch_size, frames, joints, 3)

        children_mask: torch.Tensor
            Mask, defining which joints are the children of the root
            Shape: (batch_size, frames, joints)

        Returns
        -------

        torch.Tensor
        """
        recn_loss_root_children = (
            torch.norm(position_gt - position_pred, dim=-1) * children_mask
        ).sum() / children_mask.sum()

        return recn_loss_root_children

def velocity_loss(
        position_gt: torch.Tensor,
        position_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate velocity loss for all joint position velocities

        Parameters
        ----------

        position_gt: torch.Tensor
            Ground truth joint positions.
            Shape: (batch_size, frames, joints, 3)

        position_pred: torch.Tensor
            Predicted joint positions.
            Shape: (batch_size, frames, joints, 3)

        mask: Optional[torch.Tensor] = None
            Mask, defining which joints are padded and which not
            Shape: (batch_size, frames, joints)

        Returns
        -------

        torch.Tensor
        """
        if mask is None:
            mask = torch.ones(position_gt.shape[:-1])

        vel_loss = (
            torch.norm(
                (position_gt[:, 1:] - position_gt[:, :-1])
                - (position_pred[:, 1:] - position_pred[:, :-1]),
                dim=-1,
            )
            * mask[:, 1:]
        ).sum() / mask[:, 1:].sum()

        return vel_loss

def jerk_loss(
        position_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fps: float = 1 / 30
    ) -> torch.Tensor:
        """
        Calculate jerk loss for all joint position accelaration changing rates

        Parameters
        ----------

        position_pred: torch.Tensor
            Predicted joint positions.
            Shape: (batch_size, frames, joints, 3)

        mask: Optional[torch.Tensor] = None
            Mask, defining which joints are padded and which not
            Shape: (batch_size, frames, joints)

        fps: float
            Frames per second

        Returns
        -------

        torch.Tensor
        """
        if mask is None:
            mask = torch.ones(position_pred.shape[:-1])

        acc_loss = (
            torch.norm(
                position_pred[:, 3:]
                - 3 * position_pred[:, 2:-1]
                + 3 * position_pred[:, 1:-2]
                - position_pred[:, :-3],
                dim=-1,
            )
            / (1 / fps)**3
            * mask[:, 3:]
        ).sum() / mask[:, 3:].sum()

        return acc_loss

def d6_angle_loss(
        d6_gt: torch.Tensor,
        d6_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate distance loss for all joint d6's

        Parameters
        ----------

        d6_gt: torch.Tensor
            Ground truth joint d6's.
            Shape: (batch_size, frames, joints, 6)

        d6_pred: torch.Tensor
            Predicted joint d6's.
            Shape: (batch_size, frames, joints, 6)

        mask: Optional[torch.Tensor] = None
            Mask, defining which joints are padded and which not
            Shape: (batch_size, frames, joints)

        Returns
        -------

        torch.Tensor
        """
        if mask is None:
            mask = torch.ones(d6_gt.shape[:-1])

        d6_loss = (
            torch.norm(d6_gt - d6_pred, dim=-1) * mask
        ).sum() / mask.sum()

        return d6_loss

def geodesic_loss(
        rotmat_gt: torch.Tensor,
        rotmat_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate geodesic loss for all joint 3x3 rotation matrices

        Parameters
        ----------

        rotmat_gt: torch.Tensor
            Ground truth joint rotation matrices.
            Shape: (batch_size, frames, joints, 3, 3)

        rotmat_pred: torch.Tensor
            Predicted joint rotation matrices.
            Shape: (batch_size, frames, joints, 3, 3)

        mask: torch.Tensor
            Mask, defining which joints are padded and which not
            Shape: (batch_size, frames, joints)

        Returns
        -------

        torch.Tensor
        """
        if mask is None:
            mask = torch.ones(rotmat_gt.shape[:-2])

        matrix_product = rotmat_pred @ torch.transpose(rotmat_gt, -2, -1)
        diag_sum = matrix_product.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # Clamp the acos input to valid range
        acos_input = torch.clamp((diag_sum - 1) / 2, min=-1 + 1e-7, max=1 - 1e-7)

        geodesic_loss = (
            torch.acos(acos_input) * mask
        ).sum() / mask.sum()

        return geodesic_loss

def z_pose_loss(
        z_pose: torch.Tensor,
        z_pose_augmented: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate Pose latent space loss

        Parameters
        ----------

        z_pose: torch.Tensor
            Pose token part of the latent space vector
            Shape: (batch_size, frames, N_latent/2)

        z_pose_augmented: torch.Tensor
            Augmented Pose token part of the latent space vector
            Shape: (batch_size, frames, N_latent/2)

        Returns
        -------

        torch.Tensor
        """

        z_pose_loss = torch.norm(z_pose - z_pose_augmented, dim=-1).mean()

        return z_pose_loss

def root_trajectory_loss(
        root_trajectory: torch.Tensor,
        root_trajectory_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate root trajectory loss for only the zeroth joint

        Parameters
        ----------

        root_trajectory: torch.Tensor
            Ground truth root trajectory.
            Shape: (batch_size, frames, 3)

        root_trajectory_pred: torch.Tensor
            Predicted root trajectory.
            Shape: (batch_size, frames, 3)

        Returns
        -------

        torch.Tensor
        """

        root_trajectory_loss = torch.norm(
            root_trajectory[:, 0:1] - root_trajectory_pred, dim=-1
        ).mean()

        return root_trajectory_loss

def kl_loss(
        log_var: torch.Tensor,
        mean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate kl-divergence loss for variational autoencoders

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
        """

        kl_loss = -0.5 * torch.sum(
            1 + log_var - mean.pow(2) - log_var.exp()
        )

        return kl_loss

def get_training_losses(
        gt: AnimationData,
        encoder_outputs: EncoderOutputs,
        decoder_outputs: AnimationData,
        consecutive_frames: int,
        losses_to_compute: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the losses used for the training

        Parameters
        ----------

        gt: AnimationData
            Ground truth data

        encoder_outputs: EncoderOutputs
            Encoder outputs

        decoder_outputs: AnimationData
            Decoder outputs

        consecutive_frames: int
            Integer, defining how many consecutive frames where loaded

        losses_to_compute: List[str]
            List of losses to compute

        Returns
        -------

        Dict[str, torch.Tensor]
        """

        batch_size = gt.position.shape[0]
        num_joints = gt.position.shape[1]
        num_sequences = batch_size // consecutive_frames

        decoder_outputs.prepare(num_sequences, consecutive_frames, num_joints)
        gt.prepare(num_sequences, consecutive_frames, num_joints)

        # Define loss functions and their required inputs
        loss_functions = {
            "recn_loss": lambda: reconstruction_loss(
                gt.position, decoder_outputs.position, decoder_outputs.mask
            ),
            "vel_loss": lambda: velocity_loss(
                gt.position, decoder_outputs.position, decoder_outputs.mask
            ),
            "acc_loss": lambda: jerk_loss(
                decoder_outputs.position, decoder_outputs.mask, decoder_outputs.fps
            ),
            "recn_loss_root_children": lambda: reconstruction_root_children_loss(
                gt.position, decoder_outputs.position, decoder_outputs.children_mask
            ),
            "d6_loss": lambda: d6_angle_loss(
                gt.d6, decoder_outputs.d6, decoder_outputs.mask
            ),
            "root_trajectory_loss": lambda: root_trajectory_loss(
                gt.root_trajectory, decoder_outputs.root_trajectory
            ),
            "geodesic_loss": lambda: geodesic_loss(
                gt.rotmat, decoder_outputs.rotmat, decoder_outputs.mask
            ),
            "z_pose_loss": lambda: z_pose_loss(
                encoder_outputs.z_pose, encoder_outputs.z_pose_augmented
            ),
        }

        # Calculate only the losses that are in the loss_weights dictionary
        losses = {
            loss_name: loss_fn()
            for loss_name, loss_fn in loss_functions.items()
            if loss_name in losses_to_compute
        }

        return losses
        