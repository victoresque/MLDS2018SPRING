from base.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as f


class DeepModel(BaseModel):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.fc = None
        self.build_model()

    def build_model(self):
        self.fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class ShallowModel(BaseModel):
    def __init__(self):
        super(ShallowModel, self).__init__()
        self.fc = None
        self.build_model()

    def build_model(self):
        self.fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = None
        self.fc = None
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            # 28x28
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(8),
            # 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(16)
            # 7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return f.log_softmax(output, dim=1)
