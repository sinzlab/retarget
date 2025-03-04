from typing import Dict

import torch

from retarget.utils.Quaternions import d6_2_rotmat


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
        children_mask: torch.Tensor = None,
        d6: torch.Tensor = None,
        d6_pred: torch.Tensor = None,
        root_trajectory: torch.Tensor = None,
        root_trajectory_pred: torch.Tensor = None,
        # log_var: torch.Tensor = None,
        z_pose=None,
        z_pose_augmented=None,
        mean: torch.Tensor = None,
        frame_time: torch.tensor = None,
        consec_frames: int = None,
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
           Shape: (batch_size, joints)

        children_mask: torch.Tensor
           Mask, defining which joints are the children of the root
           Shape: (batch_size, joints)

        d6: torch.Tensor
            Tensor of the ground truth 6d representation for every joint
            Shape: (batch_size, joints, 6)

        d6_pred: torch.Tensor
            Tensor of the predicted 6d representation for every joint
            Shape: (batch_size, joints, 6)

        root_trajectory: torch.Tensor
            Tensor of the predicted root trajectory, where all joitns except for zero is masked to zero
            Shape: (batch_size, joints, 3)

        root_trajectory: torch.Tensor
            Tensor of the predicted root trajectory
            Shape: (batch_size, joints, 3)

        log_var: torch.Tensor
            Log variance, where the variance is used for the latent space vector sampling
            Shape: (batch_size, N_latent, N_latent)
            N_latent is here the length of the latent space vector

        mean: torch.Tensor
            Mean used to sample the latent space vector
            Shape: (batch_size, N_latent)

        z_pose: torch.Tensor
            Pose token part of the latent space vector
            Shape: (batch_size, N_latent/2)

        z_pose_augmented: torch.Tensor
            Augmented Pose token part of the latent space vector
            Shape: (batch_size, N_latent/2)


        frame_time: torch.Tensor
            Time difference between two frames
            Shape: (batch_size, 1, 1)

        consec_frames: int
            Integer, defining how many consecutive frames where loaded

        mode: str
            Mode, to define if the losses if for training or validation

        Returns
        -------

        None
        """

        self.fk_pose = fk_pose
        self.position = position
        self.root_trajectory = root_trajectory
        self.root_trajectory_pred = root_trajectory_pred
        self.mask = mask
        self.mode = mode

        if self.mode == "train":

            # Check that all values used for train is not None
            assert children_mask is not None
            assert d6 is not None
            # assert log_var is not None
            assert frame_time is not None
            # assert mean is not None
            assert consec_frames is not None

            self.d6 = d6
            self.d6_pred = d6_pred
            # self.log_var = log_var
            # self.mean = mean
            self.children_mask = children_mask
            self.frame_time = frame_time
            self.consec_frames = consec_frames

            self.batch_size = self.position.shape[0]
            self.N_joints = self.position.shape[1]
            self.N_consecs = self.batch_size // self.consec_frames

            self.z_pose = z_pose
            self.z_pose_augmented = z_pose_augmented

            self.fk_pose = self.fk_pose.reshape(
                self.N_consecs,
                self.consec_frames,
                self.N_joints,
                3,
            )

            self.position = self.position.reshape(
                self.N_consecs,
                self.consec_frames,
                self.N_joints,
                3,
            )

            self.d6 = self.d6.reshape(
                self.N_consecs,
                self.consec_frames,
                self.N_joints,
                6,
            )

            self.d6_pred = self.d6_pred.reshape(
                self.N_consecs,
                self.consec_frames,
                self.N_joints,
                6,
            )

            self.rotmat = d6_2_rotmat(
                torch.flatten(self.d6.clone(), start_dim=0, end_dim=-2)
            )

            self.rotmat_pred = d6_2_rotmat(
                torch.flatten(self.d6_pred.clone(), start_dim=0, end_dim=-2)
            )

            self.mask = self.mask.reshape(
                self.N_consecs,
                self.consec_frames,
                self.N_joints,
            )

            self.children_mask = self.children_mask.reshape(
                self.N_consecs,
                self.consec_frames,
                self.N_joints,
            )

    def reconstruction_loss(
        self,
    ) -> None:
        """
        Calculate reconstruction loss for all joint positions

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
        """
        recn_loss = (
            torch.norm(self.position - self.fk_pose, dim=-1) * self.mask
        ).sum() / self.mask.sum()

        return recn_loss

    def reconstruction_root_children_loss(
        self,
    ) -> None:
        """
        Calculate reconstruction loss for joint positions,
        where the joint is directly connected to the root

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
        """
        recn_loss_root_children = (
            torch.norm(self.position - self.fk_pose, dim=-1) * self.children_mask
        ).sum() / self.children_mask.sum()

        return recn_loss_root_children

    def velocity_loss(
        self,
    ) -> None:
        """
        Calculate velocity loss for all joint position velocities

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
        """

        vel_loss = (
            torch.norm(
                (self.position[:, 1:] - self.position[:, :-1])
                - (self.fk_pose[:, 1:] - self.fk_pose[:, :-1]),
                dim=-1,
            )
            * self.mask[:, 1:]
        ).sum() / self.mask[:, 1:].sum()

        return vel_loss

    def jerk_loss(
        self,
    ) -> None:
        """
        Calculate jerk loss for all joint position accelaration changing rates

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
        """

        acc_loss = (
            torch.norm(
                self.fk_pose[:, 3:]
                - 3 * self.fk_pose[:, 2:-1]
                + 3 * self.fk_pose[:, 1:-2]
                - self.fk_pose[:, :-3],
                dim=-1,
            )
            / (self.frame_time**3)
            * self.mask[:, 3:]
        ).sum() / self.mask[:, 3:].sum()

        return acc_loss

    def d6_angle_loss(
        self,
    ) -> None:
        """
        Calculate distance loss for all joint d6's

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
        """

        d6_loss = (
            torch.norm(self.d6 - self.d6_pred, dim=-1) * self.mask
        ).sum() / self.mask.sum()

        return d6_loss

    def geodesic_loss(
        self,
    ) -> None:
        """
        Calculate geodesic loss for all joint 3x3 rotation matrices

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
        """

        matrix_product = self.rotmat_pred @ torch.transpose(self.rotmat, -2, -1)
        diag_sum = matrix_product.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # Clamp the acos input to valid range
        acos_input = torch.clamp((diag_sum - 1) / 2, min=-1 + 1e-7, max=1 - 1e-7)

        geodesic_loss = (
            torch.acos(acos_input) * torch.flatten(self.mask)
        ).sum() / self.mask.sum()

        return geodesic_loss

    def z_pose_loss(
        self,
    ) -> None:
        """
        Calculate Pose latent space loss

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
        """

        z_pose_loss = torch.norm(self.z_pose - self.z_pose_augmented, dim=-1).mean()

        return z_pose_loss

    def root_trajectory_loss(
        self,
    ) -> None:
        """
        Calculate root trajectory loss for only the zeroth joint

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
        """

        root_trajectory_loss = torch.norm(
            self.root_trajectory[:, 0:1] - self.root_trajectory_pred, dim=-1
        ).mean()

        return root_trajectory_loss

    def kl_loss(
        self,
    ) -> None:
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
            1 + self.log_var - self.mean.pow(2) - self.log_var.exp()
        )

        return kl_loss

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

        if self.mode == "train":
            return {
                "recn_loss": self.reconstruction_loss(),
                "vel_loss": self.velocity_loss(),
                "acc_loss": self.jerk_loss(),
                "recn_loss_root_children": self.reconstruction_root_children_loss(),
                "d6_loss": self.d6_angle_loss(),
                "root_trajectory_loss": self.root_trajectory_loss(),
                "geodesic_loss": self.geodesic_loss(),
                "z_pose_loss": self.z_pose_loss(),
                # "kl_loss": kl_loss,
            }

        return {
            "recn_loss": self.reconstruction_loss(),
            "root_trajectory_loss": self.root_trajectory_loss(),
        }
