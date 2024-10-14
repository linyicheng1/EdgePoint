import torch
from torch import nn
from typing import Optional, Callable
import torch.nn.functional as F
from torchvision.models import resnet
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class EdgePoint(nn.Module):
    def __init__(self, param=None, trainable=False):
        super().__init__()
        if param is None:
            c1 = 32
            c2 = 64
            c3 = 128
            c4 = 128
            dim = 128
        else:
            c1 = param['c1']
            c2 = param['c2']
            c3 = param['c3']
            c4 = param['c4']
            dim = param['dim']
        single_head: bool = True

        self.trainable = trainable
        self.gate = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)

        self.block2 = ResBlock(inplanes=c1, planes=c2, stride=1,
                               downsample=nn.Conv2d(c1, c2, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block3 = ResBlock(inplanes=c2, planes=c3, stride=1,
                               downsample=nn.Conv2d(c2, c3, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block4 = ResBlock(inplanes=c3, planes=c4, stride=1,
                               downsample=nn.Conv2d(c3, c4, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)

        # ================================== feature aggregation
        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        # ================================== detector and descriptor head
        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = resnet.conv1x1(dim, dim)
        self.convhead2 = resnet.conv1x1(dim, dim)

        self.conv_score = nn.Conv2d(dim // 4, 1, 1)
        self.conv_transpose_4 = nn.ConvTranspose2d(dim // 4, dim // 4, 4, stride=4, padding=0)
        self.conv_4 = nn.Conv2d(dim // 4, dim // 4, 1, stride=4, padding=0)
        self.conv_8 = nn.Conv2d(dim // 4, dim // 4, 1, stride=8, padding=0)


    def forward(self, image):
        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32

        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32

        x1_desc = self.conv_8(x1)  # B x dim//4 x H//8 x W//8
        x2_desc = self.conv_4(x2)  # B x dim//4 x H//8 x W//8
        x4_desc = self.conv_transpose_4(x4)  # B x dim//4 x H//8 x W//8
        x1234 = torch.cat([x1_desc, x2_desc, x3, x4_desc], dim=1)
        # x2_ = self.upsample2(x2)
        # x3_ = self.upsample4(x3)
        # x4_ = self.upsample8(x4)
        # x1234 = torch.cat([x1, x2_, x3_, x4_], dim=1)  # B x dim x H x W

        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        descriptor_map = self.convhead2(x1234)  # B x dim+1 x H x W
        scores_map = self.conv_score(x1)

        return scores_map, descriptor_map


    def extract_dense_map(self, image, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
        if h_ != h:
            h_padding = torch.zeros(b, c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        scores_map, descriptor_map = self.forward(image)

        # ====================================================
        if h_ != h or w_ != w:
            descriptor_map = descriptor_map[:, :, :h, :w]
            scores_map = scores_map[:, :, :h, :w]  # Bx1xHxW
        # ====================================================
        if self.trainable:
            hc = h // 8
            wc = w // 8
            heatmap = torch.reshape(scores_map, [b, hc, 8, wc, 8]) # BxHx8xWx8
            heatmap = heatmap.permute(0, 1, 3, 2, 4) # BxHxWx8x8
            nodust = torch.reshape(heatmap, [b, hc, wc, 64]) # BxHxWx64
            nodust = nodust.permute(0, 3, 1, 2) # Bx64xHxW
            dust = torch.full((b, 1, hc, wc), 0.01, device=nodust.device, requires_grad=False) # Bx1xHxW
            scores_map = torch.cat([nodust, dust], dim=1) # Bx65xHxW

        # BxCxHxW
        descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)

        if ret_dict:
            return {'descriptor_map': descriptor_map, 'scores_map': scores_map, }
        else:
            return descriptor_map, scores_map


if __name__ == '__main__':
    from thop import profile

    param = {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64}
    net = EdgePoint(param)
    weight = torch.load('../weights/EdgePoint.pt')
    net.load_state_dict(weight)
    image = torch.randn(1, 3, 512, 512)
    flops, params = profile(net, inputs=(image,))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))
