from typing import Optional

import torch


def geodesic_loss(
    rotmat_gt: torch.Tensor,
    rotmat_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate geodesic loss for all joint 3x3 rotation matrices

    Parameters
    ----------

    None

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


def reconstruction_loss(
    position_gt: torch.Tensor,
    position_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate reconstruction loss for all joint positions

        Parameters
        ----------

        None

        Returns
        -------

        torch.Tensor
    """
    if mask is None:
        mask = torch.ones(position_gt.shape[:-1])

    recn_loss = (
        torch.norm(position_gt * 170 - position_pred * 170, dim=-1) * mask
    ).sum() / mask.sum()

    return recn_loss

def root_trajectory_loss(
    root_trajectory_gt: torch.Tensor,
    root_trajectory_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate root trajectory loss for only the zeroth joint

    Parameters
    ----------

    None

    Returns
    -------

    torch.Tensor
    """

    if mask is None:
        mask = torch.ones(root_trajectory_gt.shape[:-1])

    root_trajectory_loss = torch.norm(
        root_trajectory_gt * 170 - root_trajectory_pred * 170, dim=-1
    ).mean()

    return root_trajectory_loss