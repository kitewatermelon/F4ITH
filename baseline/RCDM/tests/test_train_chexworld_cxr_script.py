import os
import subprocess
import sys

RCDM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_train_chexworld_cxr_help_lists_new_data_args():
    # Set up the environment with PYTHONPATH so the script can import guided_diffusion_rcdm
    env = os.environ.copy()
    env['PYTHONPATH'] = RCDM_ROOT
    result = subprocess.run(
        [sys.executable, "scripts/train_chexworld_cxr.py", "--help"],
        cwd=RCDM_ROOT, capture_output=True, text=True, timeout=30, env=env,
    )
    assert result.returncode == 0
    assert "--mimic_dir" in result.stdout
    assert "--nih_dir" in result.stdout
    assert "--chexpert_dir" in result.stdout
    assert "--data_dir" not in result.stdout


def test_launch_script_has_expected_content():
    script_path = os.path.join(RCDM_ROOT, "scripts", "train_chexworld_cxr.sh")
    assert os.path.isfile(script_path)
    content = open(script_path).read()
    assert "image_size 128" in content
    assert "--type_model chexworld" in content
    assert "--feat_cond" in content
    assert "torchrun" in content
