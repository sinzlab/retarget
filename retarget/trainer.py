try:
    import wandb
except ImportError:
    import warnings

    warnings.warn("Wandb is not installed. Installing it now.")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb

import numpy as np
import torch
from tqdm import tqdm

from retarget.losses import Losses
from retarget.model import graph_to_batch, mask_from_batch
from retarget.utils.Animation import fk_for_batch
from retarget.utils.Quaternions_old import d6_2_rotmat
from retarget.utils.scheduler import CosineAnnealingWarmupRestarts
import torch_geometric

# try:
#     from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
# except ImportError:
#     import warnings
#     warnings.warn("Transformers is not installed. Installing it now.")
#     import subprocess
#     import sys
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
#     from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


def loaded_graphs_to_batch(batch):
    graph_list = []
    for graphs in batch:
        graph_list += graphs

    return graph_list


def trainer(
    model,
    dataloader,
    test_dataloader,
    device="cuda",
    num_epochs=500,
    val_loss_scale=1 / 30,
    acc_loss_scale=0.00001,
    resume_from_epoch=None,
    resume_checkpoint=None,
):

    wandb_name = wandb.run.name

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0001)

    if resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])

    fps = 1 / 30
    losses = []
    root_trajectory_losses = []
    prev_best_val_loss = 1e6

    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=len(dataloader),
        cycle_mult=1,
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=50,
        gamma=1e-1 ** (1 / num_epochs),
        last_epoch=-1,
    )

    if resume_checkpoint:
        lr_scheduler.load_state_dict(resume_checkpoint["scheduler"])

    for epoch in range(resume_from_epoch, num_epochs):
        n_batches = len(dataloader)
        batch_idx = 0
        epoch_loss = []
        epoch_root_trajectory_loss = []
        pbar = tqdm(dataloader)
        # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=len(dataloader))
        # lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=len(dataloader) * num_epochs, num_cycles=num_epochs)
        for batch in pbar:
            batch_idx += 1

            batch = loaded_graphs_to_batch(batch)
            batch = torch_geometric.data.Batch.from_data_list(batch)
            batch = batch.to(device)
            # frame_time = frame_time[:, None, None].to(device)
            fps = 1 / 30

            # Create mask, position and d6 for original frame
            mask = mask_from_batch(batch)
            position = graph_to_batch(batch.position, mask)
            d6 = graph_to_batch(batch.d6, mask)
            root_trajectory = graph_to_batch(batch.root_trajectory, mask)

            # Set optimiser grad to zero
            optimizer.zero_grad()

            # Create prediction for original frame
            y_pred, mean = model(batch.x, batch.pos, batch.edge_index, mask=mask)

            fk_pose, edge_indexs = fk_for_batch(
                batch, y_pred[:, :, :6], quater=False, rotations_fmt="d6", device=device
            )
            fk_pose = fk_pose - fk_pose[..., 0:1, :]

            # create a boolean mask for the children of the root (idx 0)
            children_mask = torch.zeros(
                fk_pose.shape[0], fk_pose.shape[1], device=device
            )
            item, idx = torch.where(edge_indexs[:, :, 0] == 0)
            children_mask[item, edge_indexs[item, idx, 1]] = 1

            # compute losses
            train_losses = Losses(
                fk_pose=fk_pose,
                position=position,
                mask=mask,
                children_mask=children_mask,
                d6=d6,
                d6_pred=y_pred[:, :, :6],
                root_trajectory=root_trajectory,
                root_trajectory_pred=y_pred[:, :, 6:],
                # log_var=log_var,
                mean=mean,
                frame_time=fps,
                mode="train",
                consec_frames=8,
            ).losses

            loss = (
                100 * train_losses["recn_loss"]
                + 100 * train_losses["recn_loss_root_children"]
                + 5 * train_losses["d6_loss"]
                + 100 * train_losses["vel_loss"]
                + 100 * acc_loss_scale * train_losses["acc_loss"]
                + 100 * train_losses["root_trajectory_loss"]
            )

            loss = loss.mean()

            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            losses.append(train_losses["recn_loss"].item() * 170)
            root_trajectory_losses.append(
                train_losses["root_trajectory_loss"].item() * 170
            )

            epoch_loss.append(losses[-1])
            epoch_root_trajectory_loss.append(root_trajectory_losses[-1])

            pbar.set_description(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(epoch_loss[-10:])}"
            )

            wandb.log(
                {
                    "loss": 170 * train_losses["recn_loss"].item(),
                    "angle loss": train_losses["d6_loss"].item(),
                    "velocity loss": 170 * train_losses["vel_loss"].item() / fps,
                    "jerk loss": 170 * train_losses["acc_loss"].item() * fps,
                    "root traj loss": 170 * train_losses["root_trajectory_loss"].item(),
                }
            )

            lr_scheduler.step()

        # evaluate
        with torch.no_grad():
            val_losses = []
            val_root_trajectory_losses = []
            for batch in test_dataloader:

                batch = loaded_graphs_to_batch(batch)
                batch = torch_geometric.data.Batch.from_data_list(batch)
                batch = batch.to(device)

                mask = mask_from_batch(batch).to(device)
                position = graph_to_batch(batch.position, mask).to(device)
                root_trajectory = graph_to_batch(batch.root_trajectory, mask).to(device)

                y_pred, mean = model(
                    batch.x.to(device),
                    batch.pos.to(device),
                    batch.edge_index.to(device),
                    mask=mask.to(device),
                )

                fk_pose, edge_indexs = fk_for_batch(
                    batch,
                    y_pred[:, :, :6],
                    quater=False,
                    rotations_fmt="d6",
                    device=device,
                )
                fk_pose = fk_pose - fk_pose[..., 0:1, :]

                # compute losses
                val_loss = Losses(
                    fk_pose=fk_pose,
                    position=position,
                    root_trajectory=root_trajectory,
                    root_trajectory_pred=y_pred[:, :, 6:],
                    mask=mask,
                    mode="validation",
                ).losses

                val_losses.append(170 * val_loss["recn_loss"].item())
                val_root_trajectory_losses.append(
                    170 * val_loss["root_trajectory_loss"].item()
                )

        if np.mean(val_losses) < prev_best_val_loss:
            prev_best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), f"./models/local/{wandb_name}_best_model.pt")
            wandb.save(f"./models/local/{wandb_name}_best_model.pt")

        # Save the latest model with optimizer and scheduler and epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            },
            f"./models/local/{wandb_name}_latest_checkpoint.tar",
        )

        wandb.save(f"./models/local/{wandb_name}_latest_checkpoint.tar")

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx} / {n_batches}] Loss: {np.mean(epoch_loss)} | val_losses: {np.mean(val_losses)}"
        )
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx} / {n_batches}] Loss: {np.mean(epoch_root_trajectory_loss)} | val_losses: {np.mean(val_root_trajectory_losses)}"
        )
        wandb.log(
            {
                "mean loss": np.mean(epoch_loss),
                "mean root traj loss": np.mean(epoch_root_trajectory_loss),
                "val_loss": np.mean(val_losses),
                "val root traj loss": np.mean(val_root_trajectory_losses),
            }
        )
