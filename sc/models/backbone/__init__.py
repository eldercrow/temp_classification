import torchvision.models as models

__all__ = ['get_backbone']

BACKBONES = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
