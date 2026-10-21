import os
from omegaconf import OmegaConf

CFG = os.path.join(os.path.dirname(__file__), "..", "models", "cldm_featinv_chexworld.yaml")


def test_chexworld_config_hint_channels_is_768():
    cfg = OmegaConf.load(CFG)
    assert cfg.model.params.control_stage_config.params.hint_channels == 768
    # Everything else must match the convnext baseline (miniSD, CLIP context).
    assert cfg.model.params.control_stage_config.params.context_dim == 768
    assert cfg.model.target == "cldm.cldm.ControlLDM"
    assert cfg.model.params.first_stage_config.params.ddconfig.resolution == 256
