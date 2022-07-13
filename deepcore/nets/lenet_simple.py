import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder


# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

class LeNetAB(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        super(LeNetAB, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel == 1 else 0),
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ) ## AB: this looks like the LeNet-5 architecture
        # self.fc_1 = nn.Linear(16 * 53 * 53 if im_size[0] == im_size[1] == 224 else 16 * 5 * 5, 120)
        # self.fc_2 = nn.Linear(120, 84)
        # self.fc_3 = nn.Linear(84, num_classes)

        ## Amrita: more simple FC layers:

        self.fc_model = nn.Sequential(
            nn.Linear(256, 120),
            nn.Tanh(),
            nn.Linear(120,84),
            nn.Tanh(),
            nn.Linear(84,10)
        )
       
        # self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc_3

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc_model(x)
            # x = F.tanh(self.fc_1(x))
            # x = F.tanh(self.fc_2(x))
            # x = self.embedding_recorder(x)
            # x = self.fc_3(x)
        return x
