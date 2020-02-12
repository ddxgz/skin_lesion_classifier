import torch.nn as nn
from torchvision import models


class DenseNet(nn.Module):
    def __init__(self, num_classes: int = 7, net_choice: str = 'densenet161',
                 pretrained: bool = False):
        super(DenseNet, self).__init__()
        net = models.densenet161(pretrained=pretrained)

        modules = list(net.children())[:-1]
        self.net = nn.Sequential(*modules)

        n_features = net.classifier.in_features
        self.classifier = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.net(x)

        x = nn.ReLU()(x)
        x = nn.AvgPool2d(kernel_size=7)(x).view(x.size(0), -1)
        x = self.classifier(x)

        return x
