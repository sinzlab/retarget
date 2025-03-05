import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict

import torch

from retarget.losses import (geodesic_loss, reconstruction_loss,
                             root_trajectory_loss)
from retarget.pipeline import retarget_animation
from retarget.utils.Animation import Animation
from retarget.utils.BVH import load, save

warnings.filterwarnings("ignore")


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Visualize and reconstruct animations using a trained model"
    )
    parser.add_argument(
        "--model_name", "-m",
        type=str,
        default="clear-monkey-418",
        help="Name of the model to load",
    )
    parser.add_argument(
        "--src_path", "-s",
        type=str,
        default="./data/testing/testing/dataset-2_run_elderly_003.bvh",
        help="Path to source BVH file",
    )
    parser.add_argument(
        "--tgt_path", "-t",
        type=str,
        default=None,
        help="Optional path to target BVH file for retargeting",
    )
    parser.add_argument(
        "--output_dir", "-o", 
        type=str, 
        default="./results/animations", 
        help="Directory to save results"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu or cuda)",
    )
    args = parser.parse_args()

    args.src_path = Path(args.src_path)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    return args

def evaluate_animation(
    recon_anim: Animation,
    src_animation: Animation,
    device: torch.device
) -> None:
    pred_rotations = torch.from_numpy(recon_anim.rotations.transforms())
    gt_rotations = torch.from_numpy(src_animation.rotations.transforms())

    gt_positions = torch.from_numpy(src_animation.positions)
    pred_positions = torch.from_numpy(recon_anim.positions)

    gt_root_trajectory = gt_positions[:, 0:1]
    pred_root_trajectory = pred_positions[:, 0:1]

    gt_positions = gt_positions - gt_root_trajectory
    pred_positions = pred_positions - pred_root_trajectory

    angle_loss = geodesic_loss(gt_rotations, pred_rotations)
    recn_loss = reconstruction_loss(gt_positions * 170, pred_positions * 170)
    rt_loss = root_trajectory_loss(gt_root_trajectory * 170, pred_root_trajectory * 170)

    return {
        "angle_loss": angle_loss,
        "recn_loss": recn_loss,
        "root_trajectory_loss": rt_loss,
    }

def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics into a pretty table format.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted table as a string
    """
    if not metrics:
        return "No metrics available"
    
    # Find the longest metric name for proper alignment
    max_key_length = max(len(k) for k in metrics.keys())
    
    # Create header
    header = f"{'Metric':<{max_key_length}} | {'Value'}"
    separator = f"{'-' * max_key_length}-+-{'-' * 10}"
    
    # Format each row
    rows = [f"{k:<{max_key_length}} | {v:.4f}" for k, v in metrics.items()]
    
    # Combine all parts
    return f"\n{header}\n{separator}\n" + "\n".join(rows) + '\n'

if __name__ == "__main__":
    args = get_args()
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    print(f"Using device: {device}")

    src_animation, source_names, _ = load(str(args.src_path), ground_feet=False)
    print(f"Loaded source animation with {len(src_animation.rotations)} frames")

    if args.tgt_path:
        tgt_animation, target_names, _ = load(str(args.tgt_path), ground_feet=False)
        retargetting = True
        print(f"Loaded target animation with {len(tgt_animation.rotations)} frames")
    else:
        tgt_animation = None
        target_names = source_names
        retargetting = False
        print("No target animation provided, using source animation as target")

    start_time = time.time()
    recon_anim = retarget_animation(
        model_name=args.model_name,
        source_animation=src_animation,
        target_animation=tgt_animation,
        device=device
    )
    end_time = time.time()
    time_taken = end_time - start_time
    frames_per_second = len(src_animation.rotations) / time_taken

    # evaluate the animation
    if not retargetting:
        metrics = evaluate_animation(recon_anim, src_animation, device)
        print(format_metrics(metrics))

    # Save reconstructed and ground truth animations
    output_base = os.path.join(args.output_dir, args.src_path.stem)
    save(f"{output_base}_{args.model_name}.bvh", recon_anim, names=target_names)
    save(f"{output_base}_gt.bvh", src_animation, names=source_names)

    print(f"Saved animations to {args.output_dir}")
    # report time taken on the device
    print(f"Retargeting time: {time_taken:.2g} seconds on {device}")
    print(f"Frames per second: {frames_per_second:.0f}")
