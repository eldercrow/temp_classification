import torchvision.models as models
# from sc.models.backbone.resnet_cifar10 import *

__all__ = ['get_backbone']

# original torchvision models
BACKBONES = { name: models.__dict__[name] for name in models.__dict__ 
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
}

# for resnet_cifar
# BACKBONES = {'resnet18': resnet18, 'resnet50': resnet50}


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
