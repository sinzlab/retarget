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

import torch
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from retarget.dataset import MixamoDataset
from retarget.model import TransformerAutoEncoder
from retarget.trainer import trainer

def collate_fn(data):
    return data 

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # training parameters
    resume = False#torch.load("./models/local/helpful-elevator-168_latest_checkpoint.tar", map_location="cpu")  # 'local/model:49'
    resume_from_epoch = 0 #resume["epoch"]
    num_epochs = 60

    # model parameters
    batch_size = 128
    d_model = 64
    d_input = 9
    nhead = 8
    num_layers = 4

    wandb.init(entity="sinzlab", project="retarget", dir="./.wandb")

    # Data
    train_data = MixamoDataset(directory="/user/kyang2/u12303/skip-dataset/train_same_cmu_cleaned", mode="train")
    test_data = MixamoDataset(directory="/user/kyang2/u12303/skip-dataset/test", mode="test")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn = collate_fn)

    # Model
    if resume:
        model = TransformerAutoEncoder.from_pretrained(resume,checkpoint = resume)
    else:
        model = TransformerAutoEncoder(
            d_input=d_input, d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    # Training loop
    trainer(
        model,
        train_dataloader,
        test_dataloader,
        num_epochs=num_epochs,
        resume_from_epoch=resume_from_epoch,
        resume_checkpoint = resume,
#        device="cpu",
    )

    # Save model
    torch.save(model.state_dict(), "./final_model.pth")

    # create wandb artifact
    artifact = wandb.Artifact("mixamo", type="model")
    artifact.add_file("./final_model.pth")

    # log artifact
    wandb.log_artifact(artifact)
