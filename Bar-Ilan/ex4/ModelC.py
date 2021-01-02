from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class ModelC(nn.Module):
    def __init__(self, image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.drop0 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(100, 50)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.drop0(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def getOpt(self, params, lr):
        opt = optim.SGD(params, lr)
        return opt

    @staticmethod
    def name():
        return "ModelC"