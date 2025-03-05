try:
    import wandb
except ImportError:
    import warnings

    warnings.warn("Wandb is not installed. Installing it now.")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm

from retarget.losses import get_training_losses
from retarget.model import graph_to_batch, mask_from_batch
from retarget.types import AnimationData, EncoderOutputs
from retarget.utils.Animation import fk_for_batch
from retarget.utils.scheduler import CosineAnnealingWarmupRestarts


def loaded_graphs_to_batch(
    batch: List, mode: str = "train"
) -> Union[Tuple[List[Data], List[Data], List[Data]], List[Data]]:
    """
    Convert loaded graph data into batches for processing.

    Parameters
    ----------
    batch : list
        List of graph data from dataloader
    mode : str, default="train"
        Mode of operation, either "train" or "test"

    Returns
    -------
    tuple or list
        If mode is "train", returns (encoder_graphs, decoder_graphs, encoder_graphs_translated)
        If mode is "test", returns encoder_graphs
    """
    encoder_graphs: List[Data] = []

    if mode == "train":
        decoder_graphs: List[Data] = []
        encoder_graphs_translated: List[Data] = []

        for encoder_graph, decoder_graph, encoder_graph_translated in batch:
            encoder_graphs += encoder_graph
            decoder_graphs += decoder_graph
            encoder_graphs_translated += encoder_graph_translated

        return encoder_graphs, decoder_graphs, encoder_graphs_translated

    else:
        for graphs in batch:
            encoder_graphs += graphs

        return encoder_graphs


def trainer(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    num_epochs: int = 500,
    resume_from_epoch: Optional[int] = None,
    resume_checkpoint: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    scheduler: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Train the model with the given dataloaders.

    Parameters
    ----------
    model : nn.Module
        The model to train
    dataloader : DataLoader
        DataLoader for training data
    test_dataloader : DataLoader
        DataLoader for validation data
    device : str, default="cuda"
        Device to run training on
    num_epochs : int, default=500
        Number of epochs to train for
    resume_from_epoch : int, optional
        Epoch to resume training from
    resume_checkpoint : dict, optional
        Checkpoint to resume training from
    config : dict, optional
        Configuration dictionary containing loss weights and other parameters
    output_dir : str, optional
        Directory to save model checkpoints
    scheduler : dict, optional
        Scheduler configuration

    Returns
    -------
    None
        Model checkpoints are saved to output_dir
    """
    wandb_name: str = wandb.run.name

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0001)

    if resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])

    fps: float = 1 / 30
    losses: List[float] = []
    root_trajectory_losses: List[float] = []
    prev_best_val_loss: float = 1e6

    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=len(dataloader),
        cycle_mult=scheduler["cycle_mult"],
        max_lr=float(scheduler["max_lr"]),
        min_lr=float(scheduler["min_lr"]),
        warmup_steps=scheduler["warmup_steps"],
        gamma=1e-1 ** (1 / num_epochs),
        last_epoch=-1,
    )

    if resume_checkpoint:
        lr_scheduler.load_state_dict(resume_checkpoint["scheduler"])

    for epoch in range(resume_from_epoch, num_epochs):
        n_batches: int = len(dataloader)
        batch_idx: int = 0
        epoch_loss: List[float] = []
        epoch_root_trajectory_loss: List[float] = []
        pbar = tqdm(dataloader)

        for batch in pbar:
            ### PREPARE DATA

            batch_idx += 1

            batch_encoder, batch_decoder, batch_encoder_translated = (
                loaded_graphs_to_batch(batch, mode="train")
            )
            batch_encoder, batch_decoder, batch_encoder_translated = (
                loaded_graphs_to_batch(batch, mode="train")
            )

            batch_encoder = torch_geometric.data.Batch.from_data_list(batch_encoder)
            batch_decoder = torch_geometric.data.Batch.from_data_list(batch_decoder)
            batch_encoder_translated = torch_geometric.data.Batch.from_data_list(
                batch_encoder_translated
            )

            batch_encoder = batch_encoder.to(device)
            batch_decoder = batch_decoder.to(device)
            batch_encoder_translated = batch_encoder_translated.to(device)

            # Create mask, position and d6 for original frame
            mask: torch.Tensor = mask_from_batch(batch_decoder)

            gt = AnimationData(
                position=graph_to_batch(batch_decoder.position, mask),
                d6=graph_to_batch(batch_decoder.d6, mask),
                root_trajectory=graph_to_batch(batch_decoder.root_trajectory, mask),
                mask=mask,
                fps=1 / 30
            )

            # Set optimiser grad to zero
            optimizer.zero_grad()

            ### FORWARD PASS

            # Create prediction for original frame
            d6_pred, root_traj_pred, z_pose, z_root_traj = model(
                batch_encoder.x,
                batch_encoder.pos,
                batch_encoder.edge_index,
                batch_decoder.pos,
                mask=mask,
            )
            (
                d6_pred_augmented,
                root_traj_pred_augmented,
                z_pose_augmented,
                z_root_traj_augmented,
            ) = model(
                batch_encoder_translated.x,
                batch_encoder_translated.pos,
                batch_encoder_translated.edge_index,
                mask=mask,
            )

            ### POST PROCESSING
            fk_pose, edge_indexs = fk_for_batch(
                batch_decoder, d6_pred, quater=False, rotations_fmt="d6", device=device
            )
            fk_pose = fk_pose - fk_pose[..., 0:1, :]

            encoder_outputs = EncoderOutputs(
                z_pose=z_pose,
                z_root_trajectory=z_root_traj,
                z_pose_augmented=z_pose_augmented,
                z_root_trajectory_augmented=z_root_traj_augmented
            )

            decoder_outputs = AnimationData(
                position=fk_pose,
                d6=d6_pred,
                root_trajectory=root_traj_pred,
                mask=mask,
                edge_indexs=edge_indexs,
                fps=1 / 30
            )

            ### COMPUTE LOSSES
            
            # compute losses
            train_losses: Dict[str, torch.Tensor] = get_training_losses(
                gt=gt,
                encoder_outputs=encoder_outputs,
                decoder_outputs=decoder_outputs,
                consecutive_frames=8,
                losses_to_compute=config["train"]["loss_weights"].keys()
            )

            # use the weights defined in the config to compute the loss
            loss: torch.Tensor = sum(
                [
                    config["train"]["loss_weights"][key] * train_losses[key]
                    for key in config["train"]["loss_weights"]
                ]
            )

            loss = loss.mean()

            ### BACKWARD PASS

            loss.backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            ### LOGGING

            losses.append(train_losses["recn_loss"].item() * 170)
            root_trajectory_losses.append(
                train_losses["root_trajectory_loss"].item() * 170
            )

            epoch_loss.append(losses[-1])
            epoch_root_trajectory_loss.append(root_trajectory_losses[-1])

            wandb.log(
                {
                    "train/loss": 170 * train_losses["recn_loss"].item(),
                    "train/angle loss": train_losses["d6_loss"].item(),
                    "train/geodesic loss": train_losses["geodesic_loss"].item(),
                    "train/velocity loss": 170 * train_losses["vel_loss"].item() / fps,
                    "train/jerk loss": 170 * train_losses["acc_loss"].item() * fps,
                    "train/root traj loss": 170 * train_losses["root_trajectory_loss"].item(),
                    "train/z pose loss": train_losses["z_pose_loss"].item(),
                }
            )

            pbar.set_description(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(epoch_loss[-10:]):.2f}"
            )

            pbar.set_postfix(
                recn_loss=train_losses["recn_loss"].item() * 170,
                root_traj_loss=train_losses["root_trajectory_loss"].item() * 170,
                angle_loss=train_losses["d6_loss"].item(),
                geodesic_loss=train_losses["geodesic_loss"].item(),
            )

            lr_scheduler.step()

        # evaluate
        with torch.no_grad():
            val_losses: List[float] = []
            val_root_trajectory_losses: List[float] = []
            for batch in test_dataloader:

                batch = loaded_graphs_to_batch(batch, mode="test")
                batch = torch_geometric.data.Batch.from_data_list(batch)
                batch = batch.to(device)

                mask = mask_from_batch(batch).to(device)

                gt = AnimationData(
                    position=graph_to_batch(batch.position, mask),
                    d6=graph_to_batch(batch.d6, mask),
                    root_trajectory=graph_to_batch(batch.root_trajectory, mask),
                    mask=mask,
                    fps=1 / 30
                )

                d6_pred, root_traj_pred, z_pose, z_root_traj = model(
                    batch.x, batch.pos, batch.edge_index, mask=mask
                )

                fk_pose, edge_indexs = fk_for_batch(
                    batch,
                    d6_pred,
                    quater=False,
                    rotations_fmt="d6",
                    device=device,
                )

                fk_pose = fk_pose - fk_pose[..., 0:1, :]

                encoder_outputs = EncoderOutputs(
                    z_pose=z_pose,
                    z_root_trajectory=z_root_traj
                )

                decoder_outputs = AnimationData(
                    position=fk_pose,
                    d6=d6_pred,
                    root_trajectory=root_traj_pred,
                    mask=mask,
                    edge_indexs=edge_indexs,
                    fps=1 / 30
                )

                # compute losses
                val_loss: Dict[str, torch.Tensor] = get_training_losses(
                    gt=gt,
                    encoder_outputs=encoder_outputs,
                    decoder_outputs=decoder_outputs,
                    consecutive_frames=8,
                    losses_to_compute=config["train"]["validation_losses"]
                )

                val_losses.append(170 * val_loss["recn_loss"].item())
                val_root_trajectory_losses.append(
                    170 * val_loss["root_trajectory_loss"].item()
                )

        if np.mean(val_losses) < prev_best_val_loss:
            prev_best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), f"{output_dir}/{wandb_name}_best_model.pt")
            wandb.save(f"{output_dir}/{wandb_name}_best_model.pt")

        # Save the latest model with optimizer and scheduler and epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            },
            f"{output_dir}/{wandb_name}_latest_checkpoint.tar",
        )

        wandb.save(f"{output_dir}/{wandb_name}_latest_checkpoint.tar")

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx} / {n_batches}] Loss: {np.mean(epoch_loss)} | val_losses: {np.mean(val_losses)}"
        )
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx} / {n_batches}] Loss: {np.mean(epoch_root_trajectory_loss)} | val_losses: {np.mean(val_root_trajectory_losses)}"
        )
        wandb.log(
            {
                "mean/loss": np.mean(epoch_loss),
                "mean/root traj loss": np.mean(epoch_root_trajectory_loss),
                "val/loss": np.mean(val_losses),
                "val/root traj loss": np.mean(val_root_trajectory_losses),
            }
        )
