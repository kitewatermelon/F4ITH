#!/usr/bin/env python3
"""
Download miniSD checkpoint and prepare FeatInv models for both ConvNeXt and ChexWorld backbones.
"""

import os
import sys
from pathlib import Path

# Add code directory to path
code_dir = Path(__file__).parent / "code"
sys.path.insert(0, str(code_dir))

def download_minisd(output_dir=None):
    """Download miniSD from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    if output_dir is None:
        output_dir = Path(__file__).parent / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "miniSD.ckpt"

    if output_path.exists():
        print(f"✓ miniSD already exists at {output_path}")
        return str(output_path.absolute())

    print("Downloading miniSD from HuggingFace Hub...")
    print("Repository: justinpinkney/miniSD")

    try:
        ckpt_path = hf_hub_download(
            repo_id="justinpinkney/miniSD",
            filename="miniSD.ckpt",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            force_download=False,
        )
        print(f"✓ Downloaded to {ckpt_path}")
        return str(Path(ckpt_path).absolute())
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nAlternative: Download manually from https://huggingface.co/justinpinkney/miniSD")
        sys.exit(1)


OUTPUT_DIR = Path(__file__).parent / "output"


def create_control_featinv(minisd_path, output_name, backbone="convnext", config_yaml=None):
    """Create FeatInv control model using tool_add_control_featinv.py."""
    import subprocess

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name

    if output_path.exists():
        print(f"✓ {output_name} already exists")
        return str(output_path)

    print(f"\nCreating FeatInv control model ({backbone})...")
    print(f"  Merging {minisd_path} + ControlNet → {output_path}")

    # Run tool_add_control_featinv.py via command line
    tool_path = code_dir / "tool_add_control_featinv.py"
    cmd = [sys.executable, str(tool_path), str(minisd_path), str(output_path)]
    if config_yaml is not None:
        cmd.append(str(config_yaml))

    try:
        result = subprocess.run(cmd, cwd=str(code_dir), check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✓ Created {output_name}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        print(f"Error creating {output_name}:")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("FeatInv Model Setup: miniSD + ControlNet")
    print("=" * 60)

    # Change to FeatInv directory
    featinv_dir = Path(__file__).parent
    os.chdir(featinv_dir)

    # Step 1: Download miniSD
    print("\n[Step 1/3] Downloading miniSD checkpoint...")
    minisd_path = download_minisd("./miniSD.ckpt")

    models_dir = code_dir / "models"

    # Step 2: Create ConvNeXt version
    print("\n[Step 2/3] Creating FeatInv control model (ConvNeXt)...")
    convnext_path = create_control_featinv(
        minisd_path, "control_featinv_convnext_init.ckpt", "convnext",
        config_yaml=models_dir / "cldm_featinv_convnext.yaml",
    )

    # Step 3: Create ChexWorld version (same miniSD, ChexWorld ControlNet config)
    print("\n[Step 3/3] Creating FeatInv control model (ChexWorld)...")
    chexworld_path = create_control_featinv(
        minisd_path, "control_featinv_chexworld_init.ckpt", "chexworld",
        config_yaml=models_dir / "cldm_featinv_chexworld.yaml",
    )

    print("\n" + "=" * 60)
    print("✓ All models ready!")
    print("=" * 60)
    print(f"\nGenerated checkpoints:")
    print(f"  • {convnext_path}")
    print(f"  • {chexworld_path}")
    print(f"\nNext step:")
    print(f"  python code/tutorial_train_featinv.py")


if __name__ == "__main__":
    main()
