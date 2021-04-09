import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ThumbNet(nn.Module):
    def __init__(self, num_block):
        super().__init__()
        block = Block
        self.in_channels = 64

        self.thumb_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.thumb_conv2 = self._make_layer(block, 64, num_block[0], 1)
        self.thumb_conv3 = self._make_layer(block, 128, num_block[1], 2)
        self.thumb_avg = nn.AdaptiveAvgPool2d((1, 1))

        self.in_channels = 64
        self.original_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.original_conv2 = self._make_layer(block, 64, num_block[0], 1)
        self.original_conv3 = self._make_layer(block, 128, num_block[1], 2)
        self.original_conv4 = self._make_layer(block, 256, num_block[2], 2)
        self.original_conv5 = self._make_layer(block, 512, num_block[3], 2)
        self.original_avg = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_1 = nn.Linear(640, 64)
        self.fc_2 = nn.Linear(64, 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, original, thumb):
        thumb = self.thumb_conv1(thumb)
        thumb = self.thumb_conv2(thumb)
        thumb = self.thumb_conv3(thumb)
        thumb = self.thumb_avg(thumb)
        thumb = thumb.view(thumb.size(0), -1)
        # print(thumb.shape)

        original = self.original_conv1(original)
        original = self.original_conv2(original)
        original = self.original_conv3(original)
        original = self.original_conv4(original)
        original = self.original_conv5(original)
        original = self.original_avg(original)
        original = original.view(original.size(0), -1)
        # print(original.shape)

        x = torch.cat((thumb, original), dim=1)
        # print(x.shape)
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x


if __name__ == '__main__':
    model = ThumbNet([2, 2, 2, 2])
    model(torch.ones((1, 3, 600, 600)), torch.ones((1, 3, 100, 100)))
