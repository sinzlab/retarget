import argparse
import json
import os
from typing import Any, Dict, List

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
from torch.utils.data import DataLoader

from retarget.augment import get_augmentors
from retarget.dataset import SkIPDataset
from retarget.model import TransformerAutoEncoder
from retarget.trainer import trainer
from retarget.utils.config import load_config


def collate_fn(data: List[Any]) -> List[Any]:
    """
    Custom collate function for DataLoader that returns the data as is.

    Args:
        data: List of data items from the dataset

    Returns:
        The same data without any additional processing
    """
    return data


def main(
    config: Dict[str, Any],
    notes: str,
    output_dir: str,
    device: str,
    deactivate_wandb: bool,
) -> None:
    """
    Main training function for the SkIP model.

    Args:
        config: Dictionary containing all configuration parameters
        notes: Notes about the current training run
        output_dir: Directory to save model checkpoints and artifacts
        device: Device to train/test the model on
        deactivate_wandb: Boolean to decide wheter the run should be logged
    """
    torch.backends.cudnn.benchmark = True

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # training parameters
    resume = config["train"]["resume"]

    if deactivate_wandb:
        wandb.init("disabled")
    else:
        wandb.init(
            entity="sinzlab",
            project="retarget",
            dir="./.wandb",
            config=config,
            notes=notes,
        )

    # save the model config
    with open(output_dir + "/model_config.json", "w") as f:
        json.dump(config["model"], f)

    wandb.save(output_dir + "/model_config.json")


    # Model
    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        model = TransformerAutoEncoder.from_pretrained(
            checkpoint, checkpoint=checkpoint, model_config=config["model"]
        )
    else:
        model = TransformerAutoEncoder.build_from_config(model_config=config["model"])


    # get augmentors
    augmentors = get_augmentors(config)

    # Data
    train_data = SkIPDataset(
        directory=config["dataset"]["path"] + "/" + config["dataset"]["train_dir"],
        mode="train",
        augmentors=augmentors,
        consequtive_frames=config["train"]["consequtive_frames"],
        feature_list=model.feature_list,
    )
    test_data = SkIPDataset(
        directory=config["dataset"]["path"] + "/" + config["dataset"]["test_dir"],
        mode="test",
        augmentors=augmentors,
        consequtive_frames=config["train"]["consequtive_frames"],
        feature_list=model.feature_list,
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Training loop
    trainer(
        model,
        train_dataloader,
        test_dataloader,
        num_epochs=config["train"]["num_epochs"],
        resume_from_epoch=config["train"]["resume_from_epoch"],
        resume_checkpoint=config["train"]["resume"],
        config=config,
        output_dir=output_dir,
        device=device,
        scheduler=config["train"]["scheduler"],
    )

    # Save model
    torch.save(model.state_dict(), output_dir + "/final_model.pth")

    # create wandb artifact
    artifact = wandb.Artifact("SkIP", type="model")
    artifact.add_file(output_dir + "/final_model.pth")

    # log artifact
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SkIP model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/local",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device, to train/test the model",
    )
    parser.add_argument(
        "--deactivate_wandb",
        action="store_true",
        help="Decision to log wandb or not",
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    main(
        config["config"],
        config["notes"],
        args.output_dir,
        device=args.device,
        deactivate_wandb=args.deactivate_wandb,
    )
