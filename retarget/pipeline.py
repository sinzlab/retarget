import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

from retarget.model import TransformerAutoEncoder
from retarget.tokenizer import Tokenizer
from retarget.utils.Animation import Animation
from retarget.utils.Quaternions import Quaternions, d6_2_quat


def load_pretrained_model(
    model_name: str, device: str = "cpu"
) -> Tuple[TransformerAutoEncoder, Tokenizer]:
    """
    Load a pretrained model from checkpoint.

    Parameters
    ----------
    model_name: str
        Name of the model to load
    device: str
        Device to load the model on ('cpu' or 'cuda')

    Returns
    -------
    Tuple[TransformerAutoEncoder, Tokenizer]
        Tuple containing the loaded model and tokenizer

    Raises
    ------
    FileNotFoundError
        If the checkpoint file doesn't exist
    """
    # print(f"LOADING PRETRAINED MODEL: {model_name}")
    checkpoint_path = f"./models/local/{model_name}_latest_checkpoint.tar"
    model_config_path = f"./models/local/{model_name}_model_config.json"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if not os.path.exists(model_config_path):
        # raise warning and use default config
        # print(f"Model config not found at {model_config_path}, using default config")
        model_config = None
    else:
        with open(model_config_path, "r") as f:
            model_config = json.load(f)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerAutoEncoder.from_pretrained(
        checkpoint["epoch"], checkpoint=checkpoint, model_config=model_config
    )
    model.to(device)
    model.eval()

    tokenizer = Tokenizer(model.feature_list)
    tokenizer.to(device)

    return model, tokenizer

def _retarget_animation_with_model(
    model: TransformerAutoEncoder,
    tokenizer: Tokenizer,
    source_animation: Animation,
    target_animation: Optional[Animation] = None
) -> Animation:
    # If no target animation is provided, use the source animation
    if target_animation is None:
        target_animation = source_animation

     # Prepare source and target animations
    src = tokenizer.encode(source_animation)
    tgt = tokenizer.encode(target_animation)

    with torch.inference_mode():
        # Encode the source animation into the latent space
        pose_latent, trajectory_latent = model.encoder(
            x=src.batch.x,
            rest_pose=src.rest_pose.positions, 
            edge_index=src.batch.edge_index, 
            mask=src.mask
        )

        # Decode the latent space to the target skeleton
        rotation_pred, trajectory_pred = model.decoder(
            pose_latent=pose_latent, 
            root_traj_latent=trajectory_latent, 
            rest_pose=tgt.rest_pose.positions, 
            edge_index=tgt.batch.edge_index, 
            mask=tgt.mask
        )

    # Convert model output to animation
    retargeted_animation = tokenizer.decode(tgt.batch, rotation_pred, trajectory_pred, tgt.rest_pose)

    return retargeted_animation

def retarget_animation(
    model_name: str, 
    source_animation: Animation, 
    target_animation: Optional[Animation] = None, 
    device: str = "cpu"
) -> Animation:
    """
    Retarget an animation from a source skeleton to a target skeleton.

    Parameters
    ----------
    model_name: str
        Name of the model to use for retargeting
    source_animation: Animation
        Source animation to retarget from
    target_animation: Optional[Animation]
        Target skeleton animation to retarget to (if None, uses source)
    device: str
        Device to run inference on ('cpu' or 'cuda')

    Returns
    -------
    Animation
        Retargeted animation
    """
    # Load the model and tokenizer
    model, tokenizer = load_pretrained_model(model_name, device)

    retargeted_animation = _retarget_animation_with_model(
        model,
        tokenizer,
        source_animation,
        target_animation
    )

    return retargeted_animation

def encode_animation(
    model_name: str,
    animation: Animation,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode an animation into latent space using the specified model.

    Parameters
    ----------
    model_name: str
        Name of the model to use for encoding
    animation: Animation
        Animation to encode
    device: str
        Device to run inference on ('cpu' or 'cuda')

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple containing pose latent and trajectory latent tensors
    """
    # Load the model and tokenizer
    model, tokenizer = load_pretrained_model(model_name, device)

    # Prepare source and target animations
    source_animation = tokenizer.encode(animation)

    with torch.inference_mode():
        # Encode the source animation into the latent space
        pose_latent, trajectory_latent = model.encoder(
            x=source_animation.batch.x, 
            rest_pose=source_animation.rest_pose.positions, 
            edge_index=source_animation.batch.edge_index, 
            mask=source_animation.mask
        )

    return pose_latent, trajectory_latent