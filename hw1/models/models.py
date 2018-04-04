from base.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as f


class DeepFC(BaseModel):
    def __init__(self):
        super(DeepFC, self).__init__()
        self.fc = None
        self.build_model()

    def build_model(self):
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class MiddleFC(BaseModel):
    def __init__(self):
        super(MiddleFC, self).__init__()
        self.fc = None
        self.build_model()

    def build_model(self):
        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 17),
            nn.ReLU(inplace=True),
            nn.Linear(17, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class ShallowFC(BaseModel):
    def __init__(self):
        super(ShallowFC, self).__init__()
        self.fc = None
        self.build_model()

    def build_model(self):
        self.fc = nn.Sequential(
            nn.Linear(1, 210),
            nn.ReLU(inplace=True),
            nn.Linear(210, 1)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class DeeperMnistCNN(BaseModel):
    def __init__(self, hidden_size):
        super(DeeperMnistCNN, self).__init__()
        self.cnn = None
        self.fc = None
        self.hidden_size = hidden_size
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.hidden_size // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.hidden_size // 4, self.hidden_size // 2, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.hidden_size // 2, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 3 * 3, self.hidden_size * 3 * 3),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size * 3 * 3, 10),
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)


class DeepMnistCNN(BaseModel):
    def __init__(self):
        super(DeepMnistCNN, self).__init__()
        self.cnn = None
        self.fc = None
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            # 28x28
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 14x14
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 6x6
            nn.Conv2d(16, 18, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 3x3
        )
        self.fc = nn.Sequential(
            nn.Linear(18 * 3 * 3, 10),
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)


class MiddleMnistCNN(BaseModel):
    def __init__(self):
        super(MiddleMnistCNN, self).__init__()
        self.cnn = None
        self.fc = None
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            # 28x28
            nn.Conv2d(1, 7, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 14x14
            nn.Conv2d(7, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(12 * 7 * 7, 10),
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)


class ShallowMnistCNN(BaseModel):
    def __init__(self):
        super(ShallowMnistCNN, self).__init__()
        self.cnn = None
        self.fc = None
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            # 28x28
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 13x13
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 13 * 13, 10),
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)


class DeeperCifarCNN(BaseModel):
    def __init__(self, hidden_size):
        super(DeeperCifarCNN, self).__init__()
        self.cnn = None
        self.fc = None
        self.hidden_size = hidden_size
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            # 32x32
            nn.Conv2d(3, self.hidden_size // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 16x16
            nn.Conv2d(self.hidden_size // 4, self.hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 8x8
            nn.Conv2d(self.hidden_size // 2, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 4 * 4, 10)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)


class DeepCifarCNN(BaseModel):
    def __init__(self):
        super(DeepCifarCNN, self).__init__()
        self.cnn = None
        self.fc = None
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            # 32x32
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 8x8
            nn.Conv2d(16, 25, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(25 * 4 * 4, 10)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)


class MiddleCifarCNN(BaseModel):
    def __init__(self):
        super(MiddleCifarCNN, self).__init__()
        self.cnn = None
        self.fc = None
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            # 32x32
            nn.Conv2d(3, 7, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 16x16
            nn.Conv2d(7, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 10)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)


class ShallowCifarCNN(BaseModel):
    def __init__(self):
        super(ShallowCifarCNN, self).__init__()
        self.cnn = None
        self.fc = None
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            # 32x32
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 15x15
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 15 * 15, 10)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)
