from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

# avoid sparse gradient (tip 05)
relu = None
relu_config = None


class GAN(BaseModel):
    def __init__(self, config):
        super(GAN, self).__init__(config)

        global relu, relu_config
        relu, relu_config = (getattr(nn, config['tips']['05']['type']), config['tips']['05']['config']) \
            if config['tips']['05']['enabled'] else (nn.ReLU, {'inplace': True})

        self.generator = self.Generator(config)
        self.discriminator = self.Discriminator(config)

    def forward(self, noise):
        return self.generator(noise)

    class Generator(nn.Module):
        def __init__(self, config):
            super(GAN.Generator, self).__init__()
            self.config = config
            self.noise_dim = config['model']['noise_dim']

            global relu, relu_config
            self.conv_trans = nn.Sequential(
                nn.ConvTranspose2d(self.noise_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1024),
                relu(**relu_config),
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                relu(**relu_config),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                relu(**relu_config),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                relu(**relu_config),
                nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
            )

            for m in self.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()

        def forward(self, noise):
            output_img = self.conv_trans(noise)
            return output_img

    class Discriminator(nn.Module):
        def __init__(self, config):
            super(GAN.Discriminator, self).__init__()

            global relu, relu_config
            self.conv = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                relu(**relu_config),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                relu(**relu_config),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                relu(**relu_config),
                nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                relu(**relu_config),
                nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            )

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()

        def forward(self, input_img):
            output = self.conv(input_img)
            output = output.squeeze()
            return output
