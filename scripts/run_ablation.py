#!/usr/bin/env python3
"""
GVP-Conv Ablation Study Runner.

Trains and evaluates both the full model (with GVP) and the ablation model
(without GVP), then generates a comparison report.

Usage:
    python run_ablation.py --data_split outputs/existing_experiment/splits.txt
    python run_ablation.py --full_checkpoint outputs/full_model/checkpoints/best.pt
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


def run_training(config_path: str, resume: Optional[str] = None) -> str:
    """Run training and return the experiment directory."""
    cmd = ["python", "scripts/train.py", "--config", config_path]
    if resume:
        cmd.extend(["--resume", resume])
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return None
    
    return result


def run_evaluation(checkpoint: str, config: str, split: str, output: str) -> Dict:
    """Run evaluation and return results."""
    cmd = [
        "python", "scripts/evaluate.py",
        "--checkpoint", checkpoint,
        "--config", config,
        "--split", split,
        "--output", output
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Evaluation failed with return code {result.returncode}")
        return None
    
    # Load and return results
    with open(output, 'r') as f:
        return json.load(f)


def find_latest_experiment(output_dir: str, pattern: str) -> Optional[Path]:
    """Find the most recent experiment directory matching the pattern."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    matching = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and pattern in d.name],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    return matching[0] if matching else None


def generate_comparison_report(full_results: Dict, no_gvp_results: Dict) -> str:
    """Generate a markdown comparison report."""
    lines = [
        "# GVP-Conv Ablation Study Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "Comparison of model performance with and without GVP encoder.",
        "",
        "## Classification Metrics",
        "",
        "| Metric | Full Model (with GVP) | No-GVP Ablation | Δ (GVP Contribution) |",
        "|--------|----------------------|-----------------|---------------------|"
    ]
    
    # Extract key metrics
    metrics_to_compare = [
        ('accuracy', 'Accuracy'),
        ('macro_f1', 'Macro F1'),
        ('weighted_f1', 'Weighted F1'),
    ]
    
    full_cls = full_results['classification']
    no_gvp_cls = no_gvp_results['classification']
    
    for key, name in metrics_to_compare:
        full_val = full_cls.get(key, 0)
        no_gvp_val = no_gvp_cls.get(key, 0)
        delta = full_val - no_gvp_val
        sign = "+" if delta > 0 else ""
        lines.append(f"| {name} | {full_val:.4f} | {no_gvp_val:.4f} | {sign}{delta:.4f} |")
    
    # Per-class F1
    lines.append("")
    lines.append("### Per-Class F1 Scores")
    lines.append("")
    lines.append("| Class | Full Model | No-GVP | Δ |")
    lines.append("|-------|-----------|--------|---|")
    
    for cls_name in ['Contact', 'Near', 'Far']:
        full_f1 = full_cls['per_class'].get(cls_name, {}).get('f1', 0)
        no_gvp_f1 = no_gvp_cls['per_class'].get(cls_name, {}).get('f1', 0)
        delta = full_f1 - no_gvp_f1
        sign = "+" if delta > 0 else ""
        lines.append(f"| {cls_name} | {full_f1:.4f} | {no_gvp_f1:.4f} | {sign}{delta:.4f} |")
    
    # Binding site metrics
    lines.append("")
    lines.append("## Binding Site Detection")
    lines.append("")
    lines.append("| Metric | Full Model | No-GVP | Δ |")
    lines.append("|--------|-----------|--------|---|")
    
    bind_metrics = ['binary_f1', 'binary_precision', 'binary_recall', 'auroc', 'auprc']
    full_bind = full_results['binding_site']
    no_gvp_bind = no_gvp_results['binding_site']
    
    for metric in bind_metrics:
        full_val = full_bind.get(metric)
        no_gvp_val = no_gvp_bind.get(metric)
        if full_val is not None and no_gvp_val is not None:
            delta = full_val - no_gvp_val
            sign = "+" if delta > 0 else ""
            lines.append(f"| {metric} | {full_val:.4f} | {no_gvp_val:.4f} | {sign}{delta:.4f} |")
    
    # Conclusion
    lines.extend([
        "",
        "## Conclusion",
        "",
        "The GVP encoder contributes the following performance gains:",
        ""
    ])
    
    macro_f1_delta = full_cls.get('macro_f1', 0) - no_gvp_cls.get('macro_f1', 0)
    contact_f1_delta = full_cls['per_class'].get('Contact', {}).get('f1', 0) - no_gvp_cls['per_class'].get('Contact', {}).get('f1', 0)
    
    if macro_f1_delta > 0:
        lines.append(f"- **Macro F1**: +{macro_f1_delta:.4f} improvement")
    else:
        lines.append(f"- **Macro F1**: {macro_f1_delta:.4f} (no improvement)")
    
    if contact_f1_delta > 0:
        lines.append(f"- **Contact F1**: +{contact_f1_delta:.4f} improvement (binding site prediction)")
    else:
        lines.append(f"- **Contact F1**: {contact_f1_delta:.4f}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Run GVP-Conv Ablation Study')
    parser.add_argument('--full_config', type=str, default='configs/pocket_gvp_v2.yaml',
                        help='Config for full model')
    parser.add_argument('--no_gvp_config', type=str, default='configs/ablation_no_gvp.yaml',
                        help='Config for no-GVP ablation')
    parser.add_argument('--full_checkpoint', type=str, default=None,
                        help='Checkpoint for full model (skip training if provided)')
    parser.add_argument('--no_gvp_checkpoint', type=str, default=None,
                        help='Checkpoint for no-GVP model (skip training if provided)')
    parser.add_argument('--data_split', type=str, default=None,
                        help='Path to existing data split file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training, only run evaluation')
    args = parser.parse_args()
    
    print("="*60)
    print("GVP-Conv Ablation Study")
    print("="*60)
    
    # Determine checkpoints
    full_ckpt = args.full_checkpoint
    no_gvp_ckpt = args.no_gvp_checkpoint
    
    # Find or create split file path
    split_file = args.data_split
    
    if not args.skip_training:
        if not full_ckpt:
            print("\n[1/4] Training full model (with GVP)...")
            run_training(args.full_config)
            full_exp = find_latest_experiment(args.output_dir, "pocket_gvp_fusion")
            if full_exp:
                full_ckpt = str(full_exp / "checkpoints" / "best.pt")
                if not split_file:
                    split_file = str(full_exp / "splits.txt")
        
        if not no_gvp_ckpt:
            # Update no-gvp config to use same split
            if split_file:
                with open(args.no_gvp_config, 'r') as f:
                    no_gvp_config = yaml.safe_load(f)
                no_gvp_config['data']['split_file'] = split_file
                temp_config = f"/tmp/ablation_no_gvp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                with open(temp_config, 'w') as f:
                    yaml.dump(no_gvp_config, f)
                config_to_use = temp_config
            else:
                config_to_use = args.no_gvp_config
                
            print("\n[2/4] Training no-GVP ablation model...")
            run_training(config_to_use)
            no_gvp_exp = find_latest_experiment(args.output_dir, "ablation_no_gvp")
            if no_gvp_exp:
                no_gvp_ckpt = str(no_gvp_exp / "checkpoints" / "best.pt")
    
    # Verify checkpoints
    if not full_ckpt or not Path(full_ckpt).exists():
        print(f"Error: Full model checkpoint not found: {full_ckpt}")
        print("Please provide --full_checkpoint or run training first.")
        sys.exit(1)
    
    if not no_gvp_ckpt or not Path(no_gvp_ckpt).exists():
        print(f"Error: No-GVP model checkpoint not found: {no_gvp_ckpt}")
        print("Please provide --no_gvp_checkpoint or run training first.")
        sys.exit(1)
    
    # Run evaluations
    print("\n[3/4] Evaluating models...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"ablation_results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    full_results_path = results_dir / "full_model_results.json"
    full_results = run_evaluation(
        full_ckpt, args.full_config, "test", str(full_results_path)
    )
    
    no_gvp_results_path = results_dir / "no_gvp_results.json"
    no_gvp_results = run_evaluation(
        no_gvp_ckpt, args.no_gvp_config, "test", str(no_gvp_results_path)
    )
    
    if not full_results or not no_gvp_results:
        print("Error: Evaluation failed")
        sys.exit(1)
    
    # Generate comparison report
    print("\n[4/4] Generating comparison report...")
    report = generate_comparison_report(full_results, no_gvp_results)
    
    report_path = results_dir / "ablation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_dir}")
    print(f"  - Full model results: {full_results_path}")
    print(f"  - No-GVP results: {no_gvp_results_path}")
    print(f"  - Comparison report: {report_path}")
    
    # Print quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    full_macro = full_results['classification']['macro_f1']
    no_gvp_macro = no_gvp_results['classification']['macro_f1']
    delta = full_macro - no_gvp_macro
    print(f"Full Model Macro F1:  {full_macro:.4f}")
    print(f"No-GVP Macro F1:      {no_gvp_macro:.4f}")
    print(f"GVP Contribution:     {'+' if delta > 0 else ''}{delta:.4f}")


if __name__ == '__main__':
    main()
