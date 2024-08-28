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

from retarget.model import (TransformerAutoEncoder, graph_to_batch,
                            mask_from_batch)
from retarget.utils.Animation import fk_for_batch
from retarget.utils.BVH import load

try:
    from transformers import get_cosine_schedule_with_warmup
except ImportError:
    import warnings

    warnings.warn("Transformers is not installed. Installing it now.")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import get_cosine_schedule_with_warmup


class MixamoDataset(Dataset):
    def __init__(self, mode="train"):
        super().__init__()

        directory = "./data/Truebones"

        animations = {}

        characters = list(Path(directory).glob("*"))

        n_characters = len(characters)
        train_len = int(n_characters * 0.8)

        if mode == "train":
            characters = characters[:train_len]
        elif mode == "test":
            characters = characters[train_len:]

        stride = 1 if mode == "train" else 64

        for character in characters:
            # file = list(character.glob('*TPOSE.bvh'))

            # if len(file) == 0:
            #     print(character.name, "has no TPOSE.bvh file")

            #     continue

            # t_pose, _, _ = load(file[0])

            animations[character.name] = {}

            actions = list(character.glob(f"*.bvh"))
            actions = [f for f in actions if not f.name.endswith("_TPOSE.bvh")]
            for action in actions:
                try:
                    animations[character.name][action.name], _, _ = load(action)

                    # if animations[character.name][action.name].t_pose.shape != t_pose.positions[0].shape:
                    #     print(f'Character {character.name} has different t_pose shape')
                    #     del animations[character.name][action.name]
                    #     break

                    # animations[character.name][action.name].t_pose = t_pose.positions[0]
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
                position = torch.Tensor(position)
                position = position - position[0]

                rotation = torch.Tensor(rotation)

                features = torch.cat([rotation, position], axis=-1)

                edges = torch.LongTensor(animation.edges.T)
                t_pose = torch.Tensor(animation.t_pose)
                offsets = torch.Tensor(animation.offsets)

                self.data.append(Data(features, edges, pos=t_pose, offsets=offsets))

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


def trainer(model, dataloader, test_dataloader, device="cuda", num_epochs=2000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=num_epochs
    )

    model = model.to(device)

    losses = []
    for epoch in range(num_epochs):
        n_batches = len(dataloader)
        batch_idx = 0
        epoch_loss = []
        for batch in dataloader:
            batch_idx += 1
            mask = mask_from_batch(batch).to(device)
            y_true = graph_to_batch(batch.x, mask).to(device)

            optimizer.zero_grad()
            y_pred, mean, log_var = model(
                batch.x.to(device),
                batch.pos.to(device),
                batch.edge_index.to(device),
                mask=mask.to(device),
            )

            y_true[..., -3:] = y_true[..., -3:] - y_true[..., -3:][..., 0:1, :]

            recn_loss = (
                torch.norm(y_true[..., -3:] - y_pred[..., -3:], dim=-1) * mask
            ).mean()
            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

            loss = recn_loss + kl_loss
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch_loss.append(loss.item())

        lr_scheduler.step()

        # evaluate
        with torch.no_grad():
            val_losses = []
            for batch in test_dataloader:
                mask = mask_from_batch(batch).to(device)
                y_true = graph_to_batch(batch.x, mask).to(device)
                y_pred, mean, log_var = model(
                    batch.x.to(device),
                    batch.pos.to(device),
                    batch.edge_index.to(device),
                    mask=mask.to(device),
                )

                y_true[..., -3:] = y_true[..., -3:] - y_true[..., -3:][..., 0:1, :]

                val_loss = (
                    torch.norm(y_true[..., -3:] - y_pred[..., -3:], dim=-1) * mask
                ).mean()
                val_loss = val_loss.mean()
                val_losses.append(val_loss.item())

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx} / {n_batches}] Loss: {np.mean(epoch_loss)} | val_losses: {np.mean(val_losses)} | LR: {lr_scheduler.get_last_lr()[0]}"
        )
        wandb.log(
            {
                "loss": np.mean(epoch_loss),
                "val_loss": np.mean(val_losses),
                "lr": lr_scheduler.get_last_lr()[0],
            }
        )


if __name__ == "__main__":
    wandb.init(entity="sinzlab", project="retarget")
    # Data
    train_data = MixamoDataset(mode="train")
    test_data = MixamoDataset(mode="test")
    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # Model
    d_model = 128
    d_input = 7
    nhead = 8
    num_layers = 4
    model = TransformerAutoEncoder(
        d_input=d_input, d_model=d_model, nhead=nhead, num_layers=num_layers
    )

    # Training loop
    trainer(model, train_dataloader, test_dataloader)

    # Save model
    torch.save(model.state_dict(), "./model.pth")

    # create wandb artifact
    artifact = wandb.Artifact("truebones", type="model")
    artifact.add_file("./model.pth")

    # log artifact
    wandb.log_artifact(artifact)
