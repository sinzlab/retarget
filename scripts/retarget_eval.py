import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from retarget.losses import (geodesic_loss, reconstruction_loss,
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
    position_loss = reconstruction_loss(gt_positions_centered * 170, pred_positions_centered * 170)
    trajectory_loss = root_trajectory_loss(gt_root_trajectory * 170, pred_root_trajectory * 170)

    return {
        "angle_loss": angle_loss.item(),
        "position_loss": position_loss.item(),
        "trajectory_loss": trajectory_loss.item(),
    }


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


def create_publication_table(
    intra_metrics: Dict[str, float], 
    cross_metrics: Dict[str, float], 
    output_dir: Path
) -> Path:
    """
    Create a LaTeX table comparing intra-character and cross-character retargeting metrics.
    
    Args:
        intra_metrics: Dictionary of intra-character retargeting metrics
        cross_metrics: Dictionary of cross-character retargeting metrics
        output_dir: Directory to save the LaTeX table
        
    Returns:
        Path to the saved LaTeX table file
    """
    metrics_names = {
        "position_loss": "JP (cm)",
        "angle_loss": "JR (rad)",
        "trajectory_loss": "RT (cm)"
    }

    baseline_metrics = {
        "SAME": {
            "Intra": {
                "position_loss": 2.91,
                "angle_loss": -1,
                "trajectory_loss": -1,
            },
            "Cross": {
                "position_loss": 2.47,
                "angle_loss": -1,
                "trajectory_loss": -1,
            },
        },
        'Aberman et al.': {
            "Intra": {
                "position_loss": 2.76,
                "angle_loss": -1,
                "trajectory_loss": -1,
            },
            "Cross": {
                "position_loss": 2.25,
                "angle_loss": -1,
                "trajectory_loss": -1,
            },
        }
    }
    
    # Create LaTeX table with narrower format for two-column paper
    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\caption{Animation Retargeting Evaluation Results comparing intra-character and cross-character performance metrics. JP (cm) represents joint positions in centimeters, JR (rad) represents joint rotations in radians, and RT (cm) represents root trajectory in centimeters.}\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{lcccccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "& \\multicolumn{3}{c}{Intra} & \\multicolumn{3}{c}{Cross} \\\\\n"  # Shortened headers
    latex_table += "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
    
    # Use shorter metric names
    latex_table += "Method & JP & JR & RT & JP & JR & RT \\\\\n"
    latex_table += "\\midrule\n"
    
    # Add baseline results
    for method, metrics in baseline_metrics.items():
        method_name = "Aberman" if method == "Aberman et al." else method  # Shorter method name
        intra = metrics["Intra"]
        cross = metrics["Cross"]
        latex_table += f"{method_name} & {intra['position_loss']:.2f} & "
        latex_table += f"{intra['angle_loss'] if intra['angle_loss'] >= 0 else '-'} & "
        latex_table += f"{intra['trajectory_loss'] if intra['trajectory_loss'] >= 0 else '-'} & "
        latex_table += f"{cross['position_loss']:.2f} & "
        latex_table += f"{cross['angle_loss'] if cross['angle_loss'] >= 0 else '-'} & "
        latex_table += f"{cross['trajectory_loss'] if cross['trajectory_loss'] >= 0 else '-'} \\\\\n"
    
    # Add our method's results
    latex_table += f"Ours & {intra_metrics.get('position_loss', 0):.2f} & {intra_metrics.get('angle_loss', 0):.2f} & {intra_metrics.get('trajectory_loss', 0):.2f} & {cross_metrics.get('position_loss', 0):.2f} & {cross_metrics.get('angle_loss', 0):.2f} & {cross_metrics.get('trajectory_loss', 0):.2f} \\\\\n"
    
    # Close the table
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\label{tab:retargeting_results}\n"
    latex_table += "\\end{table}"
    
    # Save the table to a file
    output_path = output_dir / "retargeting_metrics.tex"
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"Publication-ready table saved to {output_path}")

    return output_path


def load_animation_pair(
    data_dir: Path, 
    character_a: str, 
    character_b: str, 
    animation_file: str
) -> Tuple[Animation, Animation, bool]:
    """
    Load a pair of animations for retargeting evaluation.
    
    Args:
        data_dir: Base directory containing animation data
        character_a: Source character name
        character_b: Target character name
        animation_file: Animation file name
        
    Returns:
        Tuple containing source animation, target animation, and validity flag
    """
    src_path = data_dir / character_a / animation_file
    tgt_path = data_dir / character_b / animation_file
    
    src_animation, _, _ = load(str(src_path), ground_feet=False)
    tgt_animation, _, _ = load(str(tgt_path), ground_feet=False)
    
    # Check if animations have the same number of frames
    is_valid = src_animation.rotations.shape[0] == tgt_animation.rotations.shape[0]
    
    return src_animation, tgt_animation, is_valid


def evaluate_retargeting(
    model: TransformerAutoEncoder,
    tokenizer: Tokenizer,
    data_dir: Path,
    source_characters: List[str],
    target_characters: List[str],
    animation_files: List[str],
    device: torch.device,
) -> List[Dict[str, float]]:
    """
    Evaluate retargeting performance between sets of characters.
    
    Args:
        model: The transformer model for retargeting
        tokenizer: Tokenizer for processing animations
        data_dir: Base directory containing animation data
        source_characters: List of source character names
        target_characters: List of target character names
        animation_files: List of animation file names
        device: Device to run inference on
        
    Returns:
        List of metrics dictionaries for all evaluated animations
    """
    all_metrics = []
    
    for src_char in source_characters:
        for tgt_char in target_characters:
            if src_char == tgt_char:
                continue
                
            pbar = tqdm(animation_files)
            for anim_file in pbar:
                pbar.set_description(f"{src_char + '->' + tgt_char:>25} | {anim_file:>35}")
                
                # Load animations
                src_animation, tgt_animation, is_valid = load_animation_pair(
                    data_dir, src_char, tgt_char, anim_file
                )
                
                if not is_valid:
                    continue
                
                # Perform retargeting
                retargeted_animation = _retarget_animation_with_model(
                    model, tokenizer, src_animation, tgt_animation
                )
                
                # Calculate metrics
                file_metrics = calculate_animation_metrics(retargeted_animation, tgt_animation)
                all_metrics.append(file_metrics)
                
                # Update progress bar with current averages
                avg_metrics = compute_average_metrics(all_metrics)
                pbar.set_postfix(avg_metrics)
    
    return all_metrics


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the evaluation script.

    Returns:
        Namespace containing the parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate animation retargeting quality using a trained model"
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
        default="./data/mixamo",
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

    # Define character sets for evaluation
    test_characters = ['Aj', 'BigVegas', 'Kaya', 'SportyGranny', 'Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']
    characters_a = test_characters[:4]  # Training set characters
    characters_b = test_characters[4:]  # Test set characters

    # Get all animation files from the first character's directory
    animation_files = list((args.data_dir / test_characters[0]).glob('*.bvh'))
    animation_files = [file.name for file in animation_files]
    print(f"Found {len(animation_files)} animation files for evaluation")

    # Evaluate intra-character retargeting (within training set)
    print("\nEvaluating intra-skeleton retargeting...")
    intra_metrics_list = evaluate_retargeting(
        model, tokenizer, args.data_dir, 
        characters_a, characters_a, 
        animation_files, device
    )
    
    # Evaluate cross-character retargeting (training to test set)
    print("\nEvaluating cross-skeleton retargeting...")
    cross_metrics_list = evaluate_retargeting(
        model, tokenizer, args.data_dir, 
        characters_a, characters_b, 
        animation_files, device
    )

    # Compute average metrics
    intra_avg_metrics = compute_average_metrics(intra_metrics_list)
    cross_avg_metrics = compute_average_metrics(cross_metrics_list)

    # Print results
    print("\nIntra-skeleton Retargeting Results:")
    print(format_metrics_table(intra_avg_metrics))
    
    print("\nCross-skeleton Retargeting Results:")
    print(format_metrics_table(cross_avg_metrics))

    # Create publication-ready table
    create_publication_table(intra_avg_metrics, cross_avg_metrics, args.output_dir)
