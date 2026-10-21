#!/usr/bin/env python3
"""
Ablation study analysis: compare metrics across experiments.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_wandb_metrics(output_dir: Path) -> Dict:
    """Load metrics from WandB logs (placeholder - requires WandB API)."""
    # TODO: Implement WandB API integration to pull metrics
    pass


def parse_checkpoint_logs(exp_dir: Path) -> Dict:
    """Parse checkpoint directories and extract metadata."""
    metrics = {
        'exp_name': exp_dir.name,
        'checkpoints': [],
    }

    checkpoints_dir = exp_dir / 'checkpoints'
    if checkpoints_dir.exists():
        for ckpt in sorted(checkpoints_dir.glob('step*.ckpt')):
            step = int(ckpt.stem.replace('step', '').lstrip('0') or '0')
            metrics['checkpoints'].append({
                'path': str(ckpt),
                'step': step,
            })

    return metrics


def compare_experiments(exp_dirs: List[Path]) -> None:
    """Compare multiple ablation experiments."""
    print("=" * 80)
    print("ABLATION STUDY COMPARISON: ChexWorld + patch_only mode")
    print("=" * 80)
    print()

    experiments = {}
    for exp_dir in exp_dirs:
        if exp_dir.exists():
            metrics = parse_checkpoint_logs(exp_dir)
            experiments[exp_dir.name] = metrics
            print(f"✓ {exp_dir.name}")
            print(f"  Checkpoints: {len(metrics['checkpoints'])}")
        else:
            print(f"✗ {exp_dir.name} (not found)")

    print()
    print("=" * 80)
    print("EXPERIMENT CONFIGURATIONS")
    print("=" * 80)
    print()

    config = {
        'exp1_baseline_chexworld': {
            'description': 'Baseline - Full architecture',
            'conv_proj': '✓ Yes (Conv 3×3)',
            'long_skip': '✓ Yes',
            'expected': 'Reference performance',
        },
        'exp2_no_conv_chexworld': {
            'description': 'Remove Conv 3×3 projections',
            'conv_proj': '✗ No (Linear only)',
            'long_skip': '✓ Yes',
            'expected': 'May reduce spatial awareness',
        },
        'exp3_no_skip_chexworld': {
            'description': 'Remove long-skip connections',
            'conv_proj': '✓ Yes (Conv 3×3)',
            'long_skip': '✗ No',
            'expected': 'May affect gradient flow',
        },
        'exp4_no_conv_no_skip_chexworld': {
            'description': 'Remove both Conv & long-skip (simplest)',
            'conv_proj': '✗ No (Linear only)',
            'long_skip': '✗ No',
            'expected': 'Simplest but may struggle with training',
        },
    }

    for exp_name, info in config.items():
        if exp_name in experiments:
            print(f"[{exp_name}]")
            print(f"  Description: {info['description']}")
            print(f"  Conv projections: {info['conv_proj']}")
            print(f"  Long-skip connections: {info['long_skip']}")
            print(f"  Expected impact: {info['expected']}")
            print()


def create_summary_template(output_root: Path) -> None:
    """Create a template for manual result entry."""
    summary_file = output_root / 'ablation_summary.md'

    template = """# Ablation Study Results

**Date**: [YYYY-MM-DD]
**Dataset**: MIMIC-CXR + NIH + CheXpert
**Backbone**: ChexWorld (ViT-Base/16)
**Conditioning**: patch_only (dense cross-attention)

---

## Convergence Comparison

| Exp | Name | Loss@50K | Loss@150K | Loss@300K | Notes |
|-----|------|----------|-----------|-----------|-------|
| 1   | Baseline | ?        | ?         | ?         | Reference (Conv + Skip) |
| 2   | No Conv  | ?        | ?         | ?         | Linear projections only |
| 3   | No Skip  | ?        | ?         | ?         | Sequential blocks |
| 4   | No Conv + Skip | ?   | ?         | ?         | Simplest architecture |

## Sample Quality

### 50K steps
- **Exp 1 (Baseline)**: [visual assessment]
- **Exp 2 (No Conv)**: [visual assessment]
- **Exp 3 (No Skip)**: [visual assessment]
- **Exp 4 (No Conv + Skip)**: [visual assessment]

### 150K steps
- **Exp 1 (Baseline)**: [visual assessment]
- **Exp 2 (No Conv)**: [visual assessment]
- **Exp 3 (No Skip)**: [visual assessment]
- **Exp 4 (No Conv + Skip)**: [visual assessment]

### 300K steps (convergence)
- **Exp 1 (Baseline)**: [visual assessment]
- **Exp 2 (No Conv)**: [visual assessment]
- **Exp 3 (No Skip)**: [visual assessment]
- **Exp 4 (No Conv + Skip)**: [visual assessment]

## Training Stability

- **Gradient norms**: Check `train/grad_norm` in WandB
- **Parameter norms**: Check `train/param_norm` in WandB
- **Loss smoothness**: Smoother curves indicate better training dynamics

## Key Findings

1. **Conv 3×3 impact** (Exp 1 vs 2, 4): [Findings on spatial projection importance]
   - Does removing Conv 3×3 significantly hurt convergence?
   - Can linear projections alone handle dense cross-attention?

2. **Long-skip impact** (Exp 1 vs 3, 4): [Findings on gradient flow]
   - Is residual connection necessary for stability?
   - Does training diverge without skip connections?

3. **Combined effect** (Exp 4): [Findings on minimal architecture]
   - Can simplest design still converge?
   - What's the performance gap between full vs minimal?

4. **Recommendations**: [Which components are critical for medical imaging]

## Conclusion

[Summary of which components are essential vs optional for ChexWorld-conditioned RCT]
- **Essential**: ?
- **Optional**: ?
- **Surprising**: ?
"""

    with open(summary_file, 'w') as f:
        f.write(template)

    print(f"✓ Created template: {summary_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Ablation study analysis')
    parser.add_argument('--output_root', type=Path, default=Path('/home/yspark/RCT/output'),
                       help='Root directory containing all experiment outputs')
    args = parser.parse_args()

    # Experiment directories
    exp_dirs = [
        args.output_root / 'ablation_exp1_baseline_chexworld',
        args.output_root / 'ablation_exp2_no_conv_chexworld',
        args.output_root / 'ablation_exp3_no_skip_chexworld',
        args.output_root / 'ablation_exp4_no_conv_no_skip_chexworld',
    ]

    # Run analysis
    compare_experiments(exp_dirs)

    # Create summary template
    create_summary_template(args.output_root)
