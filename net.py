import torch.nn as nn
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        # out = self.dropout(out)
        return out


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,
                 identity=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.stride = stride
        self.identity = identity
        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()
        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if identity == True else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        # inputs = self.reflection_pad(inputs)
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        inputs = self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
        return inputs

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


# 最主要的网络结构
class GetLS_RepVGG(nn.Module):
    def __init__(self, s, n, channel, stride, deploy=False, use_se=False, use_checkpoint=False):
        super(GetLS_RepVGG, self).__init__()
        self.s = s  # filter size
        self.stride = stride
        self.Channel = channel
        self.n = n
        width_multiplier = [128, 256, 256, 256]
        self.deploy = deploy
        override_groups_map = None
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint
        num_blocks = [1, 1, 1, 1]
        self.in_planes = 128
        self.stage0 = RepVGGBlock(in_channels=self.Channel, out_channels=self.in_planes, kernel_size=self.s,
                                  stride=stride,
                                  padding=1, deploy=self.deploy, use_se=self.use_se, identity=False)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(width_multiplier[0], num_blocks[0], stride=stride, identity=True)
        self.stage2 = self._make_stage(width_multiplier[1], num_blocks[1], stride=stride, identity=False)
        self.stage3 = self._make_stage(width_multiplier[2], num_blocks[2], stride=stride, identity=True)
        # self.stage4 = self._make_stage(width_multiplier[3], num_blocks[3], stride=stride)
        self.conv = ConvLayer(width_multiplier[2], n * 2, 1, stride)

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
            if stage == self.stage3:
                out = self.conv(out)
        L = out[:, :224, :, :]
        H = out[:, 224: 257, :, :]
        return L, H

    def _make_stage(self, planes, num_block, stride, identity):
        strides = [stride] + [1] * (num_block - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      use_se=self.use_se, identity=identity))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)


class Decoder(nn.Module):
    def __init__(self, s, n, channel, stride, fusion_type):
        super(Decoder, self).__init__()
        self.type = fusion_type
        self.conv_ReLx = ConvLayer(224, channel, s, stride)
        self.conv_ReHx = ConvLayer(32, channel, s, stride)
        self.conv_ReLy = ConvLayer(224, channel, s, stride)
        self.conv_ReHy = ConvLayer(32, channel, s, stride)

        if self.type.__contains__('cat'):
            # cat
            self.conv_ReL = ConvLayer(2 * channel, channel, s, stride)
            self.conv_ReH = ConvLayer(2 * channel, channel, s, stride)
        else:
            # add
            self.conv_ReL = ConvLayer(channel, channel, s, stride)
            self.conv_ReH = ConvLayer(channel, channel, s, stride)

    def forward(self, Lx, Hx, Ly, Hy):
        # get loww parts and sparse parts
        x_l = self.conv_ReLx(Lx)
        x_h = self.conv_ReHx(Hx)
        y_l = self.conv_ReLy(Ly)
        y_h = self.conv_ReHy(Hy)
        # reconstructure
        if self.type.__contains__('cat'):
            # cat
            low = self.conv_ReL(torch.cat([x_l, y_l], 1))
            high = self.conv_ReH(torch.cat([x_h, y_h], 1))
        else:
            # add
            low = self.conv_ReL(x_l + y_l)
            high = self.conv_ReH(x_h + y_h)
        out = low + high
        return out


# ========================  用于RepVGG的训练和测试代码
# RepVGGFuse network
class RepVGGFuse_net(nn.Module):
    def __init__(self, s, n, channel, stride, deploy=False, use_se=False):
        super(RepVGGFuse_net, self).__init__()
        self.deploy = deploy
        self.use_se = use_se
        self.fusion_type = 'cat'  # cat, add
        self.get_ls1 = GetLS_RepVGG(s, n, channel, stride, deploy=False, use_se=False)
        self.get_ls2 = GetLS_RepVGG(s, n, channel, stride, deploy=False, use_se=False)
        self.decoder = Decoder(s, n, channel, stride, self.fusion_type)

    def forward(self, x, y):
        fea_x_l, fea_x_H = self.get_ls1(x)
        fea_y_l, fea_y_H = self.get_ls2(y)
        out = self.decoder(fea_x_l, fea_x_H, fea_y_l, fea_y_H)
        return out


class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        b, c, h, w = X.shape
        if c == 1:
            X = X.repeat(1, 3, 1, 1)

        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.relu(self.conv4_4(h))
        relu4_3 = h
        # [relu1_2, relu2_2, relu3_3, relu4_3]
        return [relu1_2, relu2_2, relu3_3, relu4_3]
