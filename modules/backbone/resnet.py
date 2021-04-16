'''
    extract the ImageNet trained backbone of ResNet
    adapted acoording to https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
'''
from types import MethodType

import torch
import torchvision.models as models


class ResNetBackbone(torch.nn.Module):
    def __init__(self, resnet):
        super().__init__()
        assert isinstance(resnet, models.ResNet)
        self.resnet = resnet
        self.resnet._forward_impl = MethodType(ResNetBackbone._part_forward_impl, self.resnet)

    def forward(self, x):
        return self.resnet(x)

    @staticmethod
    def _part_forward_impl(resnet, x):
        output = []

        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)

        # output intermediate layers
        output.append(x)

        x = resnet.layer1(x)
        output.append(x)

        x = resnet.layer2(x)
        output.append(x)

        x = resnet.layer3(x)
        output.append(x)

        x = resnet.layer4(x)
        output.append(x)

        x = resnet.avgpool(x)
        x = torch.flatten(x, 1)
        output.append(x)

        # remove classfier
        # x = self.fc(x)

        return output


def test_resnet_backbone():
    backbone = ResNetBackbone(models.resnet101())
    output = backbone(torch.rand(2, 3, 416, 416))
    assert len(output) == 6
    assert output[-1].shape[0] == 2 and len(output[-2].shape) == 4
