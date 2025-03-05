import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from retarget.metrics import (geodesic_loss, reconstruction_loss,
                              root_trajectory_loss)
from retarget.model import TransformerAutoEncoder
from retarget.pipeline import (_retarget_animation_with_model,
                               load_pretrained_model)
from retarget.tokenizer import Tokenizer
from retarget.utils.Animation import Animation
from retarget.utils.BVH import load

warnings.filterwarnings("ignore")


def calculate_animation_metrics(
    reconstructed_animation: Animation,
    source_animation: Animation,
) -> Dict[str, float]:
    """
    Calculate quality metrics between reconstructed and source animations.
    
    Args:
        reconstructed_animation: Animation reconstructed by the model
        source_animation: Original source animation (ground truth)
        
    Returns:
        Dictionary of evaluation metrics including angle loss, reconstruction loss,
        and root trajectory loss
    """
    # Convert rotations to tensors
    pred_rotations = torch.from_numpy(reconstructed_animation.rotations.transforms())
    gt_rotations = torch.from_numpy(source_animation.rotations.transforms())

    # Convert positions to tensors
    gt_positions = torch.from_numpy(source_animation.positions)
    pred_positions = torch.from_numpy(reconstructed_animation.positions)

    # Extract root trajectories
    gt_root_trajectory = gt_positions[:, 0:1]
    pred_root_trajectory = pred_positions[:, 0:1]

    # Center positions by subtracting root trajectory for better comparison
    gt_positions_centered = gt_positions - gt_root_trajectory
    pred_positions_centered = pred_positions - pred_root_trajectory

    # Calculate various loss metrics
    angle_loss = geodesic_loss(gt_rotations, pred_rotations)
    position_loss = reconstruction_loss(gt_positions_centered, pred_positions_centered)
    trajectory_loss = root_trajectory_loss(gt_root_trajectory, pred_root_trajectory)

    return {
        "angle_loss": angle_loss.item(),
        "position_loss": position_loss.item(),
        "trajectory_loss": trajectory_loss.item(),
    }


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Format metrics into a readable table format.
    
    Args:
        metrics: Dictionary of metric names and their corresponding values
        
    Returns:
        Formatted table as a string with aligned columns
    """
    if not metrics:
        return "No metrics available"
    
    # Find the longest metric name for proper alignment
    max_key_length = max(len(k) for k in metrics.keys())
    
    # Create header and separator
    header = f"{'Metric':<{max_key_length}} | {'Value'}"
    separator = f"{'-' * max_key_length}-+-{'-' * 10}"
    
    # Format each metric row with proper alignment
    rows = [f"{k:<{max_key_length}} | {v:.4f}" for k, v in metrics.items()]
    
    # Combine all parts into a formatted table
    return f"\n{header}\n{separator}\n" + "\n".join(rows) + '\n'


def create_publication_table(metrics: Dict[str, float], dataset_name: str, output_dir: Path) -> str:
    """
    Format metrics into a publication-ready table format.
    
    Args:
        metrics: Dictionary of metric names and their corresponding values
        dataset_name: Name of the dataset being evaluated
        output_dir: Directory to save the LaTeX table file
        
    Returns:
        Path to the saved LaTeX table file
    """
    if not metrics:
        return "\\begin{table}\n\\caption{No metrics available}\n\\end{table}"
    
    # Define metric names for display
    metrics_names = {
        "angle_loss": "JR (rad)",
        "position_loss": "JP (cm)",
        "trajectory_loss": "RT (cm)",
    }

    baseline_metrics = {
        "SAME": {
            "angle_loss": 0.34,
            "position_loss": 12.40,
            "trajectory_loss": 8.58,
        },
    }

    # Create caption with descriptions
    caption = f"Reconstruction results on the {dataset_name} Dataset. {metrics_names['position_loss']} represents joint positions in centimeters, {metrics_names['angle_loss']} represents joint rotations in radians, and {metrics_names['trajectory_loss']} represents root trajectory in centimeters."
    
    # Start LaTeX table
    latex_table = "\\begin{table}[h]\n"
    latex_table += f"\\caption{{{caption}}}\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{lccc}\n"
    latex_table += "\\toprule\n"
    latex_table += f"Method & {metrics_names['position_loss']} & {metrics_names['angle_loss']} & {metrics_names['trajectory_loss']} \\\\\n"
    latex_table += "\\midrule\n"
    
    # Add comparison row (placeholder for SAME method)
    for key, value in baseline_metrics.items():
        latex_table += f"{key} & {value.get('position_loss', 0):.2f} & {value.get('angle_loss', 0):.2f} & {value.get('trajectory_loss', 0):.2f} \\\\\n"
    
    # Add our method's results
    latex_table += f"Ours & {metrics.get('position_loss', 0):.2f} & {metrics.get('angle_loss', 0):.2f} & {metrics.get('trajectory_loss', 0):.2f} \\\\\n"
    
    # Close the table
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\label{tab:evaluation_results}\n"
    latex_table += "\\end{table}"
    
    # Save the table to a file
    output_path = output_dir / f"{dataset_name}_reconstruction_metrics.tex"
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"Publication-ready table saved to {output_path}")

    return output_path


def compute_average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute average metrics across multiple animation evaluations.
    
    Args:
        metrics_list: List of metric dictionaries from individual animations
        
    Returns:
        Dictionary with averaged metrics across all animations
    """
    if not metrics_list:
        return {}
        
    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m]
        if values:
            avg_metrics[key] = sum(values) / len(values)
    
    return avg_metrics

def evaluate_model(
    model: TransformerAutoEncoder,
    tokenizer: Tokenizer,
    data_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset of animations.
    
    Args:
        model: The transformer autoencoder model to evaluate
        tokenizer: Tokenizer for processing animations
        data_dir: Directory containing animation files
        device: Device to run inference on (CPU or CUDA)
        
    Returns:
        Dictionary of averaged evaluation metrics across all animations
    """
    # Initialize metrics collection
    all_metrics = []
    fps_values = []

    # Find all BVH files in the data directory
    file_list = sorted(list(data_dir.glob('*.bvh')))
    if not file_list:
        print(f"No BVH files found in {data_dir}")
        exit(1)
        
    print(f"Found {len(file_list)} BVH files to evaluate")
    
    # Process each animation file with progress bar
    progress_bar = tqdm(file_list)
    for source_path in progress_bar:
        source_animation, source_names, _ = load(str(source_path), ground_feet=False)

        # Measure reconstruction time
        start_time = time.time()
        reconstructed_animation = _retarget_animation_with_model(
            model,
            tokenizer,
            source_animation
        )
        end_time = time.time()

        # Calculate frames per second (FPS)
        processing_time = end_time - start_time
        frames_per_second = len(source_animation.rotations) / processing_time
        fps_values.append(frames_per_second)

        # Evaluate reconstruction quality
        file_metrics = calculate_animation_metrics(reconstructed_animation, source_animation)
        file_metrics["FPS"] = frames_per_second
        all_metrics.append(file_metrics)
        
        # Update progress bar with current averages
        avg_metrics = compute_average_metrics(all_metrics)
        metrics_text = " | ".join([f"{k}: {v:.3f}" for k, v in avg_metrics.items()])
        progress_bar.set_description(f"{metrics_text}")
    
    # Compute final aggregated metrics
    final_metrics = compute_average_metrics(all_metrics)

    return final_metrics


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the evaluation script.

    Returns:
        Namespace containing the parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate animation reconstruction quality using a trained model"
    )
    parser.add_argument(
        "--model_name", "-m",
        type=str,
        default="clear-monkey-418",
        help="Name of the model to load",
    )
    parser.add_argument(
        "--data_dir", "-D",
        type=str,
        default="./data/dataset/bandai-namco",
        help="Path to dataset directory containing BVH files",
    )
    parser.add_argument(
        "--output_dir", "-o", 
        type=str, 
        default="./results/tables", 
        help="Directory to save results"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (cpu or cuda)",
    )
    args = parser.parse_args()

    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    return args

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_pretrained_model(args.model_name, device)
    
    final_metrics = evaluate_model(model, tokenizer, args.data_dir, device)

    print("\nEvaluation Results:")
    print(format_metrics_table(final_metrics))

    # Create publication-ready table
    create_publication_table(final_metrics, args.data_dir.name, args.output_dir)
