import torch

from ..model_definitions.upgrade_model import MyViT

if __name__ == '__main__':

    model = MyViT(
        chw = (3,32,32),
        n_patches = 8
    )
    print(type(model))

    x = torch.randn(7, 3, 32, 32)
    print(model(x).shape)