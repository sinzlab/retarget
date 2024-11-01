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

from retarget.model import graph_to_batch, mask_from_batch
from retarget.utils.Animation import fk_for_batch
from retarget.utils.Quaternions_old import d6_2_rotmat
from retarget.utils.scheduler import CosineAnnealingWarmupRestarts

# try:
#     from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
# except ImportError:
#     import warnings
#     warnings.warn("Transformers is not installed. Installing it now.")
#     import subprocess
#     import sys
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
#     from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


def trainer(
    model,
    dataloader,
    test_dataloader,
    device="cuda",
    num_epochs=500,
    val_loss_scale=1 / 30,
    acc_loss_scale=(1 / 30) ** 2,
    resume_from_epoch=None,
):

    wandb_name = wandb.run.name
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)

    model = model.to(device)

    losses = []
    prev_best_val_loss = 1e6
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=len(dataloader),
        cycle_mult=1,
        max_lr=1e-3,
        min_lr=1e-6,
        warmup_steps=200,
        gamma=1e-1 ** (1 / num_epochs),
        last_epoch=-1,
    )
    for epoch in range(resume_from_epoch, num_epochs):
        n_batches = len(dataloader)
        batch_idx = 0
        epoch_loss = []
        pbar = tqdm(dataloader)
        # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=len(dataloader))
        # lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=len(dataloader) * num_epochs, num_cycles=num_epochs)
        for (
            batch,
            batch_prev,
            batch_prev_prev,
            batch_prev_prev_prev,
            frame_time,
        ) in pbar:
            batch_idx += 1

            batch = batch.to(device)
            batch_prev = batch_prev.to(device)
            batch_prev_prev = batch_prev_prev.to(device)
            batch_prev_prev_prev = batch_prev_prev_prev.to(device)
            frame_time = frame_time[:, None, None].to(device)

            # Create mask, position and d6 for original frame
            mask = mask_from_batch(batch)
            position = graph_to_batch(batch.position, mask)
            d6 = graph_to_batch(batch.d6, mask)

            # Create mask, position and d6 for previous frame
            mask_prev = mask_from_batch(batch_prev)
            position_prev = graph_to_batch(batch_prev.position, mask_prev)
            d6_prev = graph_to_batch(batch_prev.d6, mask_prev)

            # Create mask, position and d6 for previous previous frame
            mask_prev_prev = mask_from_batch(batch_prev_prev)
            position_prev_prev = graph_to_batch(
                batch_prev_prev.position, mask_prev_prev
            )
            d6_prev_prev = graph_to_batch(batch_prev_prev.d6, mask_prev_prev)

            # Create mask, position and d6 for previous previous previous frame
            mask_prev_prev_prev = mask_from_batch(batch_prev_prev_prev)
            position_prev_prev_prev = graph_to_batch(
                batch_prev_prev_prev.position, mask_prev_prev_prev
            )
            d6_prev_prev_prev = graph_to_batch(
                batch_prev_prev_prev.d6, mask_prev_prev_prev
            )

            # Set optimiser grad to zero
            optimizer.zero_grad()

            # Create prediction for original frame
            y_pred, mean, log_var = model(
                batch.x, batch.pos, batch.edge_index, mask=mask
            )

            fk_pose, edge_indexs = fk_for_batch(
                batch, y_pred, quater=False, rotations_fmt="d6"
            )
            fk_pose = fk_pose - fk_pose[..., 0:1, :]

            # Create prediction for previous frame
            y_pred_prev, mean_prev, log_var_prev = model(
                batch_prev.x, batch_prev.pos, batch_prev.edge_index, mask=mask_prev
            )

            fk_pose_prev, edge_indexs_prev = fk_for_batch(
                batch_prev, y_pred_prev, quater=False, rotations_fmt="d6"
            )
            fk_pose_prev = fk_pose_prev - fk_pose_prev[..., 0:1, :]

            # Create prediction for previous previous frame
            y_pred_prev_prev, mean_prev_prev, log_var_prev_prev = model(
                batch_prev_prev.x,
                batch_prev_prev.pos,
                batch_prev_prev.edge_index,
                mask=mask_prev_prev,
            )

            fk_pose_prev_prev, edge_indexs_prev_prev = fk_for_batch(
                batch_prev_prev, y_pred_prev_prev, quater=False, rotations_fmt="d6"
            )
            fk_pose_prev_prev = fk_pose_prev_prev - fk_pose_prev_prev[..., 0:1, :]

            # Create prediction for previous previous previousframe
            y_pred_prev_prev_prev, mean_prev_prev_prev, log_var_prev_prev_prev = model(
                batch_prev_prev_prev.x,
                batch_prev_prev_prev.pos,
                batch_prev_prev_prev.edge_index,
                mask=mask_prev_prev_prev,
            )

            fk_pose_prev_prev_prev, edge_indexs_prev_prev_prev = fk_for_batch(
                batch_prev_prev_prev,
                y_pred_prev_prev_prev,
                quater=False,
                rotations_fmt="d6",
            )
            fk_pose_prev_prev_prev = (
                fk_pose_prev_prev_prev - fk_pose_prev_prev_prev[..., 0:1, :]
            )

            # create a boolean mask for the children of the root (idx 0)
            children_mask = torch.zeros(
                fk_pose.shape[0], fk_pose.shape[1], device=device
            )
            item, idx = torch.where(edge_indexs[:, :, 0] == 0)
            children_mask[item, edge_indexs[item, idx, 1]] = 1

            # Calculate velocities for time steps t, t-1, t-2 for jerk/velocity loss
            v_t = (position - position_prev) / frame_time
            v_t_pred = (fk_pose - fk_pose_prev) / frame_time
            v_t_minus_1_pred = (fk_pose_prev - fk_pose_prev_prev) / frame_time
            v_t_minus_2_pred = (fk_pose_prev_prev - fk_pose_prev_prev_prev) / frame_time

            # Calculate acceleration for time steps t and t-1 for jerk loss
            a_t_pred = (v_t_pred - v_t_minus_1_pred) / frame_time
            a_t_minus_1_pred = (v_t_minus_1_pred - v_t_minus_2_pred) / frame_time

            # compute losses
            recn_loss = (
                torch.norm(position - fk_pose, dim=-1) * mask
            ).sum() / mask.sum()
            vel_loss = (torch.norm(v_t - v_t_pred, dim=-1) * mask).sum() / mask.sum()
            acc_loss = (
                torch.norm(a_t_pred - a_t_minus_1_pred, dim=-1) * mask
            ).sum() / mask.sum()
            recn_loss_root_children = (
                torch.norm(position - fk_pose, dim=-1) * children_mask
            ).sum() / children_mask.sum()
            d6_loss = (torch.norm(d6 - y_pred, dim=-1) * mask).sum() / mask.sum()
            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

            loss = (
                recn_loss
                + (1e-6 * kl_loss)
                + 10 * recn_loss_root_children
                + d6_loss
                + val_loss_scale * vel_loss
                + acc_loss_scale * acc_loss
            )
            loss = loss.mean()

            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            losses.append(recn_loss.item())
            epoch_loss.append(losses[-1])

            pbar.set_description(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(epoch_loss[-10:])}"
            )

            wandb.log({"loss": recn_loss.item()})

            lr_scheduler.step()

        # evaluate
        with torch.no_grad():
            val_losses = []
            for batch in test_dataloader:
                mask = mask_from_batch(batch).to(device)
                position = graph_to_batch(batch.position, mask).to(device)

                y_pred, mean, log_var = model(
                    batch.x.to(device),
                    batch.pos.to(device),
                    batch.edge_index.to(device),
                    mask=mask.to(device),
                )

                fk_pose, edge_indexs = fk_for_batch(
                    batch, y_pred, quater=False, rotations_fmt="d6"
                )
                fk_pose = fk_pose - fk_pose[..., 0:1, :]

                val_loss = (
                    torch.norm(position - fk_pose, dim=-1) * mask
                ).sum() / mask.sum()

                val_losses.append(val_loss.item())
        
        if np.mean(val_losses) < prev_best_val_loss:
            prev_best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), f"./models/local/{wandb_name}_best_model.pt")
            wandb.save(f"./models/local/{wandb_name}_best_model.pt")
        
        #Save the latest model with optimizer and scheduler and epoch
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    }, 
                    f"./models/local/{wandb_name}_latest_checkpoint.tar",
        )
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx} / {n_batches}] Loss: {np.mean(epoch_loss)} | val_losses: {np.mean(val_losses)}"
        )
        wandb.log({"mean loss": np.mean(epoch_loss), "val_loss": np.mean(val_losses)})
