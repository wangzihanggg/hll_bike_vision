# -*- coding:utf-8 _*-

from collections import OrderedDict
import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class UNetBig(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, WithActivateLast=True, ActivateFunLast=None):
        super(UNetBig, self).__init__()
        features = init_features
        self.WithActivateLast = WithActivateLast
        self.ActivateFunLast = ActivateFunLast
        self.encoder1 = UNetBig._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetBig._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetBig._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetBig._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetBig._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNetBig._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNetBig._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNetBig._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNetBig._block(features * 2, features, name="dec1")
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.cbam = CBAMLayer(8)
        self.sa = SpatialAttention()
        self.sa_maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_sa = self.sa(enc1)
        enc1_sa_pool = self.sa_maxpool(enc1_sa)
        enc2 = self.encoder2(self.pool1(enc1))
        enc2_sa = self.sa(enc2)
        enc2_sa_pool = self.sa_maxpool(enc2_sa)
        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.relu(enc3 * enc1_sa_pool)
        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = self.relu(enc4 * enc2_sa_pool)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.cbam(dec1)

        if self.WithActivateLast:
            # return torch.sigmoid(self.conv(dec1))  # BS*1*256*256
            return self.ActivateFunLast(self.conv(dec1))
        else:
            return self.sigmoid(self.conv(dec1))   # BS*1*256*256

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # 用字典的形式进行网络定义，字典key即为网络每一层的名称
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (
                        name + "conv3",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm3", nn.BatchNorm2d(num_features=features)),
                    (name + "relu3", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == '__main__':
    net = UNetBig(in_channels=3, out_channels=3, WithActivateLast=False)
    input = torch.randn(2, 3, 640, 480)
    output = net(input)
    print(output.shape)
