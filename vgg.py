import torch
import torch.nn as nn
from torchvision.models import vgg16


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        self.layers = nn.ModuleList(features).eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        results = []
        layers_of_interest = {3, 8, 15, 22}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in layers_of_interest:
                results.append(x)

        return results
