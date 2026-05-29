from .fl_model import FluorescenceModel
from .he_model import HEModel

def get_model(modality, cfg):
    # 根据输入的 modality 和配置 cfg 返回对应的模型实例
    if modality == 'fluorescence':
        return FluorescenceModel(
            in_channels=cfg.get('in_channels', 1),
            base=cfg.get('base_channels', 64)
        )
    elif modality == 'he':
        return HEModel(
            he_classes=cfg.get('he_classes', 4),
            base=cfg.get('base_channels', 64)
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")