import torch.nn as nn
import numpy as np
import torch
from model.se_block import SEBlock
import copy

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1,groups = 1, padding_mode = 'zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        padding_11 = padding - kernel_size // 2
        if padding_11 < 0:
            padding_11 = 0


        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs)+ self.rbr_1x1(inputs) + id_out))

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
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGG(nn.Module):
    def __init__(self, num_blocks, num_classes = 1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.num_class = num_classes
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        stage0_blocks = []
        stage0_blocks.append(RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=1, padding=100, deploy=self.deploy, use_se=self.use_se))
        stage0_blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=self.in_planes, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se))
        stage0_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage1 = nn.ModuleList(stage0_blocks)
        self.cur_layer_idx = 1
        self.stage2 = self._make_stage(int(128 * width_multiplier[0]), num_blocks[0], stride=1, number=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[1]), num_blocks[1], stride=1, number=3)
        self.stage4 = self._make_stage(int(512 * width_multiplier[2]), num_blocks[2], stride=1, number=4)
        self.stage5 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=1, number=5)
        self.stage6 = self._make_stage(4096, num_blocks[4],   stride=2, number=6)

        # stage5_blocks = []
        # stage5_blocks.append(RepVGGBlock(in_channels=int(512 * width_multiplier[3]), out_channels=4096, kernel_size=3, stride=2, padding=2, deploy=self.deploy, use_se=self.use_se))
        # stage5_blocks.append(RepVGGBlock(in_channels=4096, out_channels=4096, kernel_size=1, stride=1, deploy=self.deploy, use_se=self.use_se))
        # stage5_blocks.append(RepVGGBlock(in_channels=4096, out_channels=num_classes, kernel_size=1, stride=1, deploy=self.deploy, use_se=self.use_se))
        # self.stage6 = nn.ModuleList(stage5_blocks)
        # self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

        self.stage3_1 = nn.Conv2d(256, 21, 1) #conv3.shape = [1, 192, 16, 16]
        self.stage4_1 = nn.Conv2d(512, 21, 1) #conv4.shape = [1, 1280, 8, 8]

        self.upscale6 = nn.ConvTranspose2d(21, 21, 4, 2)
        self.upscale4 = nn.ConvTranspose2d(21, 21, 4, 2)
        self.upscale = nn.ConvTranspose2d(21, 21, 16, 8)

    def _make_stage(self, planes, num_blocks, stride, number):
        strides = [stride] + [1] *(num_blocks - 1)
        blocks = []
        for idx, stride in enumerate(strides):
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            if number == 6:
                if idx == 0:
                    blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride,
                                          padding=2 , groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
                elif idx == len(strides)-1:
                    planes = self.num_class
                    blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=1, stride=stride,
                                              padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
                else:
                    blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=1, stride=stride, padding=0, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))

            else:
                blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride,
                                          padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        if number != 6:
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.ModuleList(blocks)

    def forward(self, x):
        conv = []
        idx = 0
        #
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6):
            for block in stage:
                if idx == 0:
                    out = block(x)
                    idx +=1
                else:
                    out = block(out)
            conv.append(out)
        #
        conv1 = conv[0]  #(227, 227)
        conv2 = conv[1]  #(113, 113)
        conv3 = conv[2]  #(56, 56)
        conv4 = conv[3]  #(28, 28)
        conv5 = conv[4]  #(14, 14)
        conv6 = conv[5]  #(8, 8)

        # upscale6
        upscale6 = self.upscale6(conv6)
        # conv4 1x1 conv & upscale4
        scale4 = self.stage4_1(conv4)
        scale4 = scale4[:, :, 5:5+upscale6.size()[2], 5:5+upscale6.size()[3]].contiguous()
        # conv6 + conv4
        scale4 += upscale6

        # conv3 1x1 conv
        scale3 = self.stage3_1(conv3)
        # upscale (conv4+ conv6)
        upscale4 = self.upscale4(scale4)
        scale3 = scale3[:, :, 9:9 + upscale4.size()[2], 9:9 + upscale4.size()[3]].contiguous()

        scale3 += upscale4

        output = self.upscale(scale3)
        output = output[:, :, 31:31+x.size()[2], 31:31+x.size()[3]].contiguous()

        return output

    def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
        if do_copy:
            model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        if save_path is not None:
            torch.save(model.state_dict(), save_path)

        return model
    def repvgg_model_convert2(model:torch.nn.Module, save_path=None, do_copy=True):
        if do_copy:
            model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        if save_path is not None:
            torch.save(model.state_dict(), save_path)

        return model




if __name__=='__main__':
    x = torch.randn(1, 3, 256, 256)
    my_model = RepVGG(num_blocks=[2, 3, 3, 3, 3], num_classes=21, width_multiplier=[1, 1, 1, 1], override_groups_map=None, deploy=False)

    # from fvcore.nn import FlopCountAnalysis
    #
    # flops = FlopCountAnalysis(model_a0, x)
    out = my_model(x)
