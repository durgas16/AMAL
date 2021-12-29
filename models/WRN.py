import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)  # , momentum = 0.001
        self.relu1 = nn.LeakyReLU(inplace=False)  # , negative_slope = 0.02)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)  # , momentum = 0.001
        self.relu2 = nn.LeakyReLU(inplace=False)  # , negative_slope = 0.02)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Flatten(nn.Module):
    def __init__(self, d):
        super(Flatten, self).__init__()
        self.d = d

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class WideResNet(nn.Module):
    def __init__(self, input_shape, depth, num_classes, widen_factor=1, dropRate=0.0, repeat=3, bias=True):
        super(WideResNet, self).__init__()
        nChannels = [16]
        if widen_factor > 20:
            for ii in range(repeat):
                nChannels.append(2 ** ii * widen_factor)
        else:
            for ii in range(repeat):
                nChannels.append(2 ** ii * 16 * widen_factor)
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(input_shape[1], nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.blocks = [NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)]
        for ii in range(repeat - 1):
            self.blocks.append(NetworkBlock(n, nChannels[ii + 1], nChannels[ii + 2], block, 2, dropRate))
        self.blocks = nn.ModuleList(self.blocks)
        self.bn1 = nn.BatchNorm2d(nChannels[-1])
        self.relu = nn.LeakyReLU(inplace=True)
        self.flatten = Flatten(nChannels[-1])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, num_classes, bias=bias)
        self.nChannels = nChannels[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if bias:
                    m.bias.data.zero_()

        self.embDim = self.feature_size

    def _forward_conv(self, x):
        out = self.conv1(x)
        for i, blk in enumerate(self.blocks):
            out = blk(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, output_size=1)
        return out

    def forward(self, x, last=False, freeze=False):

        if freeze:
            with torch.no_grad():
                x = self._forward_conv(x)
                outfea = self.flatten(x)

        else:
            x = self._forward_conv(x)
            outfea = self.flatten(x)

        x = self.fc(outfea)

        if last:
            return x, outfea
        else:
            return x

    def get_embedding_dim(self):
        return self.embDim


def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model

