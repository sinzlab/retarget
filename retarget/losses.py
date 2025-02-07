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
        children_mask: torch.Tensor = None,
        d6: torch.Tensor = None,
        #log_var: torch.Tensor = None,
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
            Tensor of the 6d representation for every joint
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
        self.mask = mask
        self.mode = mode

        if self.mode == "train":

            # Check that all values used for train is not None
            assert children_mask is not None
            assert d6 is not None
            #assert log_var is not None
            assert frame_time is not None
            assert mean is not None
            assert consec_frames is not None
            
            self.d6 = d6
            #self.log_var = log_var
            self.mean = mean
            self.children_mask = children_mask
            self.frame_time = frame_time
            self.consec_frames = consec_frames

            self.batch_size = self.position.shape[0]
            self.N_joints = self.position.shape[1]
            self.N_consecs = self.batch_size // self.consec_frames

            self.fk_pose = self.fk_pose.reshape(self.N_consecs,
                                                self.consec_frames,
                                                self.N_joints,
                                                3,
            )

            self.position = self.position.reshape(self.N_consecs,
                                                  self.consec_frames,
                                                  self.N_joints,
                                                  3,
            )

            self.d6 = self.d6.reshape(self.N_consecs,
                                      self.consec_frames,
                                      self.N_joints,
                                      6,
            )

            self.mask = self.mask.reshape(self.N_consecs,
                                          self.consec_frames,
                                          self.N_joints,
            )

            self.children_mask = self.children_mask.reshape(self.N_consecs,
                                                            self.consec_frames,
                                                            self.N_joints,
            )
            
            
    def reconstruction_loss():
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

    def reconstruction_root_children_loss():
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
    
    def velocity_loss():
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
                torch.norm(self.position[:,1:] - self.fk_pose[:,:-1], dim=-1) * self.mask[:,1:]).sum() / self.mask[:,1:].sum()

        return vel_loss

    def jerk_loss():
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
                torch.norm(self.position[:,3:] - 3 * self.position[:,2:-1] + 3 * self.position[:,1:-2] - self.position[:,:-3], dim=-1) / (self.frame_time**3) * self.mask[:,3:]
        ).sum() / self.mask[:,3:].sum()

        return acc_loss

    def d6_angle_loss():
        """
        Calculate jerk loss for all joint d6's
        
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

    def kl_loss():
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
                "recn_loss": reconstruction_loss(),
                "vel_loss": velocity_loss(),
                "acc_loss": jerk_loss(),
                "recn_loss_root_children": reconstruction_root_children_loss(),
                "d6_loss": d6_angle_loss(),
                #"kl_loss": kl_loss,
            }

        return {"recn_loss": reconstruction_loss}
