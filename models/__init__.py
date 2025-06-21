from .model import Model
from .edge_model import EdgeModel
from .lied import LIED 
def get_model(name, config):
    models = {
        "edge": EdgeModel,
        "new": LIED 
    }
    return models[name.lower()](config)