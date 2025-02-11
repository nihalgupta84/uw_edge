#File: models/__init__.py

from .model import Model
from .edge_model_v1 import EdgeModel_V1
from .edge_model_v2 import EdgeModel_V2
from .edge_model_v3 import EdgeModel_V3
from .wavelet_model_v1 import WaveletModel_V1
from .wavelet_model_v2 import WaveletModel_V2
from .wavelet_model_v3 import WaveletModel_V3

def create_model(opt):
    """
    Create the model specified by opt.MODEL.NAME and pass
    config-based parameters for ablation (edge module, etc.).
    """
    model_name = opt.MODEL.NAME.lower()

    if model_name == "model":
        # Example model that takes no extra kwargs
        return Model()

    elif model_name == "edge_v1":
        return EdgeModel_V1(
            in_channels          = opt.MODEL.INPUT_CHANNELS,
            base_channels        = opt.MODEL.BASE_CHANNELS,
            use_edge_module      = opt.MODEL.EDGE_MODULE,
            use_attention_module = opt.MODEL.ATTENTION_MODULE,
            use_ck               = opt.MODEL.EDGE_CK,
            use_hk               = opt.MODEL.EDGE_HK,
            use_vk               = opt.MODEL.EDGE_VK,
            init_weights         = opt.MODEL.INIT_WEIGHTS
        )

    elif model_name == "edge_v2":
        # You can pass any relevant config flags similarly
        return EdgeModel_V2()

    elif model_name == "edge_v3":
        return EdgeModel_V3()

    elif model_name == "wavelet_v1":
        return WaveletModel_V1()

    elif model_name == "wavelet_v2":
        return WaveletModel_V2()

    elif model_name == "wavelet_v3":
        return WaveletModel_V3()

    else:
        raise ValueError(f"Unknown model type: {opt.MODEL.NAME}")
