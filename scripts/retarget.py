import argparse
import os
from pathlib import Path

import torch

from retarget.utils.BVH import load, save

from retarget.pipeline import retarget_animation


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
        default="worldly-terrain-399",
        help="Name of the model to load",
    )
    parser.add_argument(
        "--src_path", "-s",
        type=str,
        default="/user/kyang2/u12303/skip-dataset/train/Amy/Capoeira.bvh",
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
        default="./results", 
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
        print(f"Loaded target animation with {len(tgt_animation.rotations)} frames")
    else:
        tgt_animation = None
        target_names = source_names
        print("No target animation provided, using source animation as target")

    recon_anim = retarget_animation(
        model_name=args.model_name,
        source_animation=src_animation,
        target_animation=tgt_animation,
        device=device
    )

    # Save reconstructed and ground truth animations
    output_base = os.path.join(args.output_dir, args.src_path.stem)
    save(f"{output_base}_{args.model_name}.bvh", recon_anim, names=target_names)
    save(f"{output_base}_gt.bvh", src_animation, names=source_names)

    print(f"Saved animations to {args.output_dir}")
