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

    tokenizer = Tokenizer()

    return model, tokenizer


def prepare_animation_for_model(animation: Animation, tokenizer: Tokenizer, device: str = "cpu") -> Tuple[Batch, torch.Tensor]:
    """
    Prepare an animation for model input by converting to graph and moving to device.
    
    Parameters
    ----------
    animation: Animation
        Animation to preprocess
    tokenizer: Tokenizer
        Tokenizer for encoding the animation
    device: str
        Device to move the data to
        
    Returns
    -------
    Tuple[Batch, torch.Tensor]
        Tuple containing the batch and mask tensors
    """
    graph_data = animation.as_graph()
    batch, mask = tokenizer.encode(graph_data)

    # Move data to the appropriate device
    batch.x = batch.x.to(device)
    batch.pos = batch.pos.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch.root_trajectory = batch.root_trajectory.to(device)
    mask = mask.to(device)

    return batch, mask


def convert_model_output_to_animation(
    rotation_pred: torch.Tensor, 
    trajectory_pred: torch.Tensor, 
    target_batch: Batch, 
    target_animation: Animation, 
    tokenizer: Tokenizer,
) -> Animation:
    """
    Convert model predictions to an Animation object.
    
    Parameters
    ----------
    rotation_pred: torch.Tensor
        Predicted rotations in 6D format
    trajectory_pred: torch.Tensor
        Predicted root trajectory
    target_batch: Batch
        Target batch data
    target_animation: Animation
        Target animation for reference
    tokenizer: Tokenizer
        Tokenizer for decoding
    scale_factor: float
        Scale factor for positions and offsets
        
    Returns
    -------
    Animation
        Reconstructed animation
    """
    # Move predictions to CPU for processing
    rotation_pred = rotation_pred.cpu()
    trajectory_pred = trajectory_pred.cpu()

    # Decode the predictions
    fk_pose, _ = tokenizer.decode(target_batch, rotation_pred)

    # Convert 6D rotations to quaternions
    quaternion_rotations = Quaternions(np.stack([d6_2_quat(d6) for d6 in rotation_pred]))
    
    # Process positions
    local_positions = (fk_pose - fk_pose[..., 0:1, :]).detach().numpy()
    global_positions = local_positions + trajectory_pred.detach().numpy()

    # Create reconstructed animation with proper scaling
    reconstructed_animation = Animation(
        quaternion_rotations,
        global_positions,
        target_animation.orients,
        target_animation.offsets,
        target_animation.parents,
    )

    return reconstructed_animation


def _retarget_animation_with_model(
    model: TransformerAutoEncoder,
    tokenizer: Tokenizer,
    source_animation: Animation,
    target_animation: Optional[Animation] = None,
    device: str = "cpu"
) -> Animation:
    # If no target animation is provided, use the source animation
    if target_animation is None:
        target_animation = source_animation

     # Prepare source and target animations
    source_batch, source_mask = prepare_animation_for_model(source_animation, tokenizer, device)
    target_batch, target_mask = prepare_animation_for_model(target_animation, tokenizer, device)

    with torch.inference_mode():
        # Encode the source animation into the latent space
        pose_latent, trajectory_latent = model.encoder(
            x=source_batch.x, 
            rest_pose=source_batch.pos, 
            edge_index=source_batch.edge_index, 
            mask=source_mask
        )

        # Decode the latent space to the target skeleton
        rotation_pred, trajectory_pred = model.decoder(
            pose_latent=pose_latent, 
            root_traj_latent=trajectory_latent, 
            rest_pose=target_batch.pos, 
            edge_index=target_batch.edge_index, 
            mask=target_mask
        )

    # Convert model output to animation
    retargeted_animation = convert_model_output_to_animation(
        rotation_pred, 
        trajectory_pred, 
        target_batch, 
        target_animation, 
        tokenizer
    )

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
        target_animation,
        device
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
    source_batch, source_mask = prepare_animation_for_model(animation, tokenizer, device)

    with torch.inference_mode():
        # Encode the source animation into the latent space
        pose_latent, trajectory_latent = model.encoder(
            x=source_batch.x, 
            rest_pose=source_batch.pos, 
            edge_index=source_batch.edge_index, 
            mask=source_mask
        )

    return pose_latent, trajectory_latent