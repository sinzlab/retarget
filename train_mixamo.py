import os

# remove wandb folder
if os.path.exists("./wandb"):
    import shutil

    shutil.rmtree("./wandb")

try:
    import wandb
except ImportError:
    import warnings

    warnings.warn("Wandb is not installed. Installing it now.")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from retarget.model import (TransformerAutoEncoder, graph_to_batch,
                            mask_from_batch)
from retarget.utils.Animation import fk_for_batch
from retarget.utils.BVH import load
from retarget.utils.Quaternions_old import (d6_2_quat, d6_2_rotmat, quat_2_d6,
                                            quat_2_rotmat, rotmat_2_quat)
from retarget.utils.scheduler import CosineAnnealingWarmupRestarts

try:
    from transformers import (
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup)
except ImportError:
    import warnings

    warnings.warn("Transformers is not installed. Installing it now.")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import (
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup)

torch.backends.cudnn.benchmark = True


class MixamoDataset(Dataset):
    def __init__(self, mode="train"):
        super().__init__()

        directory = "./data/SAME/BVH"

        animations = {}

        characters = list(Path(directory).glob("*"))

        if mode == "train":
            characters = characters[:13]
        elif mode == "test":
            characters = characters[13:]

        stride = 1 if mode == "train" else 64

        for character in characters:
            animations[character.name] = {}
            for action in character.glob("*.bvh"):
                try:
                    animations[character.name][action.name], _, _ = load(action)
                except Exception as e:
                    print(f"Error loading {character.name}/{action.name}: {e}")

            animations[character.name] = list(animations[character.name].values())

        self.animations = list(animations.values())

        # flatten the lists
        self.animations = [
            animation for character in self.animations for animation in character
        ]

        self.data = []
        for animation in self.animations:
            for position, rotation in zip(
                animation.positions[::stride], animation.rotations[::stride]
            ):
                d6 = quat_2_d6(rotation)

                position = torch.Tensor(position)
                position = position - position[0]
                features = torch.cat([torch.Tensor(d6), position], dim=-1)
                edges = torch.LongTensor(animation.edges.T)
                t_pose = torch.Tensor(animation.t_pose)
                offsets = torch.Tensor(animation.offsets)
                self.data.append(
                    Data(
                        features,
                        edges,
                        d6=torch.Tensor(d6),
                        rotation=torch.Tensor(rotation),
                        position=position,
                        pos=t_pose,
                        offsets=offsets,
                    )
                )

        print("=== Mixamo Dataset Summary ===")
        print(
            f"Loaded {len(self.animations)} animation clips for {len(animations)} characters"
        )
        # print summary for each character
        for character, animations in animations.items():
            print(f"{character}")
            print(f"    - Animations: {len(animations)}")
        print(f"Total frames: {len(self.data):,}")
        print("===============================")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def trainer(
    model,
    dataloader,
    test_dataloader,
    device="cuda",
    num_epochs=500,
    resume_from_epoch=None,
):
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
        gamma=0.99,
        last_epoch=-1,
    )
    for epoch in range(resume_from_epoch, num_epochs):
        n_batches = len(dataloader)
        batch_idx = 0
        epoch_loss = []
        pbar = tqdm(dataloader)
        # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=len(dataloader))
        # lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=len(dataloader) * num_epochs, num_cycles=num_epochs)
        for batch in pbar:
            batch_idx += 1

            batch = batch.to(device)

            mask = mask_from_batch(batch)
            position = graph_to_batch(batch.position, mask)
            d6 = graph_to_batch(batch.d6, mask)

            optimizer.zero_grad()
            y_pred, mean, log_var = model(
                batch.x, batch.pos, batch.edge_index, mask=mask
            )

            rotmat = d6_2_rotmat(y_pred)
            fk_pose, edge_indexs = fk_for_batch(batch, rotmat, quater=False)
            fk_pose = fk_pose - fk_pose[..., 0:1, :]

            # create a boolean mask for the childen of the root (idx 0)
            children_mask = torch.zeros(
                fk_pose.shape[0], fk_pose.shape[1], device=device
            )
            item, idx = torch.where(edge_indexs[:, :, 0] == 0)
            children_mask[item, edge_indexs[item, idx, 1]] = 1

            # compute losses
            recn_loss = (
                torch.norm(position - fk_pose, dim=-1) * mask
            ).sum() / mask.sum()  #
            recn_loss_root_children = (
                torch.norm(position - fk_pose, dim=-1) * children_mask
            ).sum() / children_mask.sum()
            d6_loss = (torch.norm(d6 - y_pred, dim=-1) * mask).sum() / mask.sum()

            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

            loss = recn_loss + (1e-6 * kl_loss) + 10 * recn_loss_root_children + d6_loss
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

                rotmat = d6_2_rotmat(y_pred)
                fk_pose, edge_indexs = fk_for_batch(batch, rotmat, quater=False)
                fk_pose = fk_pose - fk_pose[..., 0:1, :]

                val_loss = (
                    torch.norm(position - fk_pose, dim=-1) * mask
                ).sum() / mask.sum()

                # val_loss = (torch.norm(y_true[..., -3:] - y_pred[..., -3:], dim=-1) * mask).mean()
                # val_loss = val_loss.mean()
                val_losses.append(val_loss.item())

        if np.mean(val_losses) < prev_best_val_loss:
            prev_best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), f"./models/local/best_model.pt")
            wandb.save(f"./models/local/best_model.pt")

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx} / {n_batches}] Loss: {np.mean(epoch_loss)} | val_losses: {np.mean(val_losses)}"
        )
        wandb.log({"loss": np.mean(epoch_loss), "val_loss": np.mean(val_losses)})


if __name__ == "__main__":
    resume = "local/model:49"
    resume_from_epoch = 50
    wandb.init(entity="sinzlab", project="retarget")
    batch_size = 64

    # Data
    train_data = MixamoDataset(mode="train")
    test_data = MixamoDataset(mode="test")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Model
    if resume:
        model = TransformerAutoEncoder.from_pretrained(resume)
    else:
        d_model = 64
        d_input = 9
        nhead = 8
        num_layers = 4
        model = TransformerAutoEncoder(
            d_input=d_input, d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    # Training loop
    trainer(
        model, train_dataloader, test_dataloader, resume_from_epoch=resume_from_epoch
    )

    # Save model
    torch.save(model.state_dict(), "./final_model.pth")

    # create wandb artifact
    artifact = wandb.Artifact("mixamo", type="vae")
    artifact.add_file("./model.pth")

    # log artifact
    wandb.log_artifact(artifact)
