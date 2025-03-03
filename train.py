import os
import argparse
import json

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

# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from retarget.dataset import MixamoDataset
from retarget.model import TransformerAutoEncoder
from retarget.trainer import trainer
from retarget.utils.config import load_config

def collate_fn(data):
    return data

def main(config, notes, output_dir):
    torch.backends.cudnn.benchmark = True

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # training parameters
    resume = config["resume"]

    # model parameters
    batch_size = config["model"]["batch_size"]
    d_model = config["model"]["d_model"]
    d_input = config["model"]["d_input"]
    nhead = config["model"]["nhead"]
    num_layers = config["model"]["num_layers"]

    wandb.init(entity="sinzlab", project="retarget", dir="./.wandb", config=config, notes=notes)

    # save the model config
    with open(output_dir + "/model_config.json", "w") as f:
        json.dump(config["model"], f)

    wandb.save(output_dir + "/model_config.json")

    # Data
    train_data = MixamoDataset(
        directory=config["dataset"]["path"] + "/" + config["dataset"]["train_dir"],
        mode="train",
    )
    test_data = MixamoDataset(
        directory=config["dataset"]["path"] + "/" + config["dataset"]["test_dir"], mode="test"
    )
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, collate_fn=collate_fn
    )

    # Model
    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        model = TransformerAutoEncoder.from_pretrained(checkpoint, checkpoint=checkpoint)
    else:
        model = TransformerAutoEncoder(
            d_input=d_input, d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    # Training loop
    trainer(
        model,
        train_dataloader,
        test_dataloader,
        num_epochs=config["num_epochs"],
        resume_from_epoch=config["resume_from_epoch"],
        resume_checkpoint=config["resume"],
        config=config,
        output_dir=output_dir,
    )

    # Save model
    torch.save(model.state_dict(), output_dir + "/final_model.pth")

    # create wandb artifact
    artifact = wandb.Artifact("SkIP", type="model")
    artifact.add_file(output_dir + "/final_model.pth")

    # log artifact
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SkIP model')
    parser.add_argument('--config', type=str, default='configs/main.yaml', help='Path to the configuration file')
    parser.add_argument('--output_dir', type=str, default='./models/local', help='Path to the output directory')

    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    main(config['config'], config['notes'], args.output_dir)
