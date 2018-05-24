from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

# batch normalization (tip 04)
batch_norm_g = True
batch_norm_d = True
# avoid sparse gradient (tip 05)
relu_g = None
relu_d = None
relu_g_config = None
relu_d_config = None
# dropouts in generator (tip 17)
dropout_g = False
dropout_g_rate = 0


class DCGAN(BaseModel):
    def __init__(self, config):
        super(DCGAN, self).__init__(config)

        global batch_norm_g, batch_norm_d, relu_g, relu_g_config, relu_d, relu_d_config, dropout_g, dropout_g_rate
        batch_norm_g = config['tips']['04']['generator']['enabled']
        batch_norm_d = config['tips']['04']['discriminator']['enabled']
        relu_g, relu_g_config, relu_d, relu_d_config = \
            (getattr(nn, config['tips']['05']['generator']['type']),
             config['tips']['05']['generator']['config'],
             getattr(nn, config['tips']['05']['discriminator']['type']),
             config['tips']['05']['discriminator']['config']) \
            if config['tips']['05']['enabled'] else (nn.ReLU, {'inplace': True}, nn.ReLU, {'inplace': True})
        dropout_g = config['tips']['17']['enabled']
        dropout_g_rate = config['tips']['17']['rate'] if dropout_g else 0

        self.generator = self.Generator(config)
        self.discriminator = self.Discriminator(config)

    def forward(self, noise):
        return self.generator(noise)

    class Generator(nn.Module):
        def __init__(self, config):
            super(DCGAN.Generator, self).__init__()
            self.config = config
            self.noise_dim = config['model']['noise_dim']

            global relu_g, relu_g_config, batch_norm_g, dropout_g, dropout_g_rate
            self.conv_trans = nn.Sequential(
                nn.ConvTranspose2d(self.noise_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
                *([nn.BatchNorm2d(1024)] if batch_norm_g else []),
                *([nn.Dropout2d(dropout_g_rate, inplace=True)] if dropout_g else []),
                relu_g(**relu_g_config),
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
                *([nn.BatchNorm2d(512)] if batch_norm_g else []),
                *([nn.Dropout2d(dropout_g_rate, inplace=True)] if dropout_g else []),
                relu_g(**relu_g_config),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                *([nn.BatchNorm2d(256)] if batch_norm_g else []),
                *([nn.Dropout2d(dropout_g_rate, inplace=True)] if dropout_g else []),
                relu_g(**relu_g_config),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                *([nn.BatchNorm2d(128)] if batch_norm_g else []),
                *([nn.Dropout2d(dropout_g_rate, inplace=True)] if dropout_g else []),
                relu_g(**relu_g_config),
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
            super(DCGAN.Discriminator, self).__init__()

            global relu_d, relu_d_config, batch_norm_d
            self.conv = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
                *([nn.BatchNorm2d(128)] if batch_norm_d else []),
                relu_d(**relu_d_config),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                *([nn.BatchNorm2d(256)] if batch_norm_d else []),
                relu_d(**relu_d_config),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                *([nn.BatchNorm2d(512)] if batch_norm_d else []),
                relu_d(**relu_d_config),
                nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
                *([nn.BatchNorm2d(1024)] if batch_norm_d else []),
                relu_d(**relu_d_config),
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
