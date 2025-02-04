#File: models/__init__.py

from .model import Model
from .edge_model_v1 import EdgeModel_V1
from .edge_model_v2 import EdgeModel_V2
from .edge_model_v3 import EdgeModel_V3
from .wavelet_model_v1 import WaveletModel_V1
from .wavelet_model_v2 import WaveletModel_V2
from .wavelet_model_v3 import WaveletModel_V3


# (hint) You can set MODEL.NAME in your config file to "version3", "wavelet", or "edge".
def create_model(model_name: str):
    """Factory function to create models based on name.
    
    Args:
        model_name: Name of the model to create (case insensitive)
        
    Returns:
        Instantiated model
    """

    if model_name == "model":
        return Model()
    elif model_name == "edge_v1":
        return EdgeModel_V1()
    elif model_name == "edge_v2":
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
        raise ValueError(f"Unknown model type: {model_name}")