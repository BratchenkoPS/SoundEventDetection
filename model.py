from torchvision.models import resnet18

import torch.nn as nn


class Resnet18Multi(nn.Module):
    """
    Class for modification of resnet18 architecture for multilabel classification of audio
    """

    def __init__(self, n_classes: int) -> None:
        """
        Initialises the model with changed conv1 layer (for 1 channel audio) and fully-connected layer for correct
         number of classes
        Args:
            n_classes: number of classes for output
        """
        super().__init__()
        resnet = resnet18(pretrained=True)
        resnet.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model = resnet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.base_model(x))
