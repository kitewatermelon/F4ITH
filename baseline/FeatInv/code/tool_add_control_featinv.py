import os
import sys
from pathlib import Path

assert len(sys.argv) in (3, 4), 'Args: <input> <output> [config_yaml]'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), f'Input model does not exist: {input_path}'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from cldm.model import create_model

# PyTorch 2.6 compatibility - allow safe globals for checkpoint loading
torch.serialization.add_safe_globals([
    __import__('pytorch_lightning.callbacks.model_checkpoint', fromlist=['ModelCheckpoint']).ModelCheckpoint
])


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

# Use relative path from this script
script_dir = Path(__file__).parent
default_cfg = script_dir / 'models' / 'cldm_featinv_convnext.yaml'
config_path = Path(sys.argv[3]) if len(sys.argv) == 4 else default_cfg
model = create_model(config_path=str(config_path))

pretrained_weights = torch.load(input_path, weights_only=False)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
