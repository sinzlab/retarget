import argparse
import os
import json
from pathlib import Path

import torch
import numpy as np

from retarget.utils.Quaternions_old import d6_2_quat, Quaternions
from retarget.utils.Animation import Animation
from retarget.model import TransformerAutoEncoder, graph_to_batch
from retarget.tokenizer import Tokenizer
from retarget.utils.BVH import load, save

def load_model(model_name, device="cpu"):
    """Load a pretrained model from checkpoint."""
    print(f"LOADING PRETRAINED MODEL: {model_name}")
    checkpoint_path = f"./models/local/{model_name}_latest_checkpoint.tar"
    model_config_path = f"./models/local/{model_name}_model_config.json"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    if not os.path.exists(model_config_path):
        # raise warning and use default config
        print(f"Model config not found at {model_config_path}, using default config")
        model_config = None
    else:
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerAutoEncoder.from_pretrained(checkpoint["epoch"], checkpoint=checkpoint, model_config=model_config)
    model.to(device)
    model.eval()

    tokenizer = Tokenizer()

    return model, tokenizer

def reconstruct(model, tokenizer, src_animation, device="cpu"):
    """Reconstruct animation using the model."""
    src_data = src_animation.as_graph()
    batch, mask = tokenizer.encode(src_data)
    
    # Move data to the appropriate device
    batch.x = batch.x.to(device)
    batch.pos = batch.pos.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch.root_trajectory = batch.root_trajectory.to(device)
    mask = mask.to(device)

    with torch.inference_mode():
        # Encode the animation into the latent space
        pose_token, root_traj_token = model.encoder(batch.x, batch.pos, batch.edge_index, mask=mask)
        root_trajectory_gt = graph_to_batch(batch.root_trajectory, mask)

        # Decode the latent space back into the animation
        d6_pred, root_trajectory_pred = model.decoder(pose_token, root_traj_token, batch.pos, batch.edge_index, mask=mask)

        # Move results back to CPU for numpy operations
        d6_pred = d6_pred.cpu()
        root_trajectory_pred = root_trajectory_pred.cpu()
        
        fk_pose, _ = tokenizer.decode(batch, d6_pred)

        rotations = Quaternions(np.stack([d6_2_quat(d6) for d6 in d6_pred]))
        positions = (fk_pose - fk_pose[..., 0:1, :]).detach().numpy()
        positions += root_trajectory_pred.detach().numpy()

    # Create reconstructed animation
    recon_anim = Animation(
        rotations,
        positions * 170,
        src_animation.orients,
        src_animation.offsets * 170,
        src_animation.parents,
    )

    # Scale original animation to match reconstructed one
    src_animation.positions *= 170
    src_animation.offsets *= 170

    return recon_anim, root_trajectory_gt.cpu().numpy(), root_trajectory_pred.detach().numpy()

def get_args():
    parser = argparse.ArgumentParser(description="Visualize and reconstruct animations using a trained model")
    parser.add_argument("--model_name", type=str, default="worldly-terrain-399", help="Name of the model to load")
    parser.add_argument("--src_path", type=str, default="/user/kyang2/u12303/skip-dataset/train/Amy/Capoeira.bvh", help="Path to source BVH file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    args = parser.parse_args()

    args.src_path = Path(args.src_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    return args

if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    print(f"Using device: {device}")

    model, tokenizer = load_model(args.model_name, device)

    src_animation, names, _ = load(str(args.src_path), ground_feet=False)
    print(f"Loaded animation with {len(src_animation.rotations)} frames")

    recon_anim, gt_trajectory, pred_trajectory = reconstruct(model, tokenizer, src_animation, device)

    # Save reconstructed and ground truth animations
    output_base = os.path.join(args.output_dir, args.src_path.stem)
    save(f"{output_base}_{args.model_name}.bvh", recon_anim, names=names)
    save(f"{output_base}_gt.bvh", src_animation, names=names)
    
    print(f"Saved animations to {args.output_dir}")