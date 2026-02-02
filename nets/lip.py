import torch.nn as nn
import numpy as np
import torch
from torchvision import models


class SimpleLipNet(nn.Module):
    """
    Implementation of Lipschitz regularized network
    """

    def __init__(self, backbone, input_sz, output_sz, hidden_layers=[]):
        super(SimpleLipNet, self).__init__()

        self.LipNet = nn.Sequential()
        # self.LipNet.append(backbone)
        # self.LipNet.append(nn.utils.spectral_norm(nn.Linear(input_sz, hidden_layers[0] if hidden_layers else output_sz)))

        # for i, in_sz in enumerate(hidden_layers):
        #     out_sz = output_sz if i >= len(hidden_layers) - 1 else hidden_layers[i+1]
        #     self.LipNet.append(nn.utils.spectral_norm(nn.Linear(in_sz, out_sz)))

        self.backbone = backbone
        self.lip_layer = nn.utils.spectral_norm(nn.Linear(input_sz, hidden_layers[0]))
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(hidden_layers[0])
        self.fc = nn.Linear(hidden_layers[0], output_sz)

    def forward(self, inputs):  # [N,C,H,W]
        x = self.backbone(inputs)
        lip_out = torch.flatten(x, 1)
        lip_out = self.norm(self.lip_layer(lip_out))
        lip_out = self.relu(lip_out)
        out = self.fc(lip_out)
        return lip_out, out


if __name__ == "__main__":
    resnet = models.resnet18(pretrained=False, num_classes=512)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    print(resnet)
    simple_lip_net = SimpleLipNet(resnet, 512, 10, [512])
    data = torch.tensor(np.random.randn(8, 3, 32, 32).astype(np.float32))
    label = torch.tensor(np.arange(0, 8))
    print(simple_lip_net)
