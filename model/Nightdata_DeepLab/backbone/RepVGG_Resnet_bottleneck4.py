import torch.nn as nn
import numpy as np
import torch
import copy
import math
import torch.nn as nn
from difflib import get_close_matches
import torch.utils.model_zoo as model_zoo
from backbone.se_block import SEBlock
from math import log
from model.DA_ECA_module import DA_ECA


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, groups, deploy, use_se, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.groups = groups
        self.use_se = use_se
        self.deploy = deploy
        self.in_planes = inplanes
        #
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = self._make_stage(planes, 1, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        # self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def _make_stage(self, planes, num_blocks, kernel_size, stride, padding , bias, dilation=1):
        strides = [stride] + [1] *(num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(in_channels=planes, out_channels=planes, kernel_size=kernel_size, dilation=dilation, stride=stride,  padding=dilation, groups=self.groups, deploy=self.deploy, use_se=self.use_se))

        return nn.ModuleList(blocks)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        for stage in self.conv2:
            out = stage(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



def resnet50(BatchNorm, deploy, model_name):
    return RepVGG_ResNet([3, 4, 6, 3], num_classes = 21, width_multiplier=[1, 1, 1, 1], override_groups_map=None, deploy=deploy, use_se=False, model_name=model_name)

def resnet101(BatchNorm, deploy, model_name):
    model = RepVGG_ResNet([3, 4, 23, 3], num_classes = 21, width_multiplier=[1, 1, 1, 1], override_groups_map=None, deploy=deploy, use_se=False, model_name=model_name)
    return model



def build_backbone(backbone, BatchNorm, deploy):
    if backbone == 'resnet50':
        return resnet50(BatchNorm, deploy, 'resnet50')
    elif backbone == 'resnet101':
        return resnet101(BatchNorm, deploy, 'resnet101')

    else:
        NotImplementedError


def conv_bn(in_channels, out_channels, kernel_size, dilation, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, dilation=dilation,padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result

def conv(in_channels, out_channels, kernel_size, dilation, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, dilation=dilation,padding=padding, groups=groups, bias=False))

    return result

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1,groups = 1, padding_mode = 'zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        # assert kernel_size == 3
        # assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,stride=stride, padding=padding, groups=groups)
            if dilation > 1:
                self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, dilation=dilation, stride=stride, padding=0, groups=groups)
            else:
                self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                    dilation=dilation, stride=stride, padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.se(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.se(self.rbr_dense(inputs)+ self.rbr_1x1(inputs) + id_out)

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


class RepVGG_ResNet(nn.Module):
    def __init__(self, num_blocks, model_name, num_classes = 1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super(RepVGG_ResNet, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.num_block = num_blocks
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.in_planes = min(3, int(64 * width_multiplier[0]))
        self.cur_layer_idx = 1
        #ResNet
        blocks = [1, 2, 4]
        self.block = Bottleneck
        self.BatchNorm = nn.BatchNorm2d
        self.strides = [1, 2, 2, 1]
        self.dilations = [1, 1, 1, 2]
        self.backbone_model_name = model_name
        #
        self.layer0 = self._make_stage(64, 3, 3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        self.in_planes = 64
        self.layer1 = self._make_layer(self.block, 64, self.num_block[0], stride=self.strides[0], dilation=self.dilations[0], BatchNorm=self.BatchNorm)
        self.layer2 = self._make_layer(self.block, 128, self.num_block[1], stride=self.strides[1], dilation=self.dilations[1], BatchNorm=self.BatchNorm)
        self.layer3 = self._make_layer(self.block, 256, self.num_block[2], stride=self.strides[2], dilation=self.dilations[2], BatchNorm=self.BatchNorm)
        self.layer4 = self._make_MG_unit(self.block, 512, blocks, stride=self.strides[3], dilation=self.dilations[3], BatchNorm=self.BatchNorm)
        #
        self.t1 = int (abs((log(256, 2) + 1) / 2))
        self.k1 = self.t1 if self.t1 % 2 else self.t1 + 1
        self.da_eca_module1 = DA_ECA(self.k1)

        self.t2 = int (abs((log(512, 2) + 1) / 2))
        self.k2 = self.t2 if self.t2 % 2 else self.t2 + 1
        self.da_eca_module2 = DA_ECA(self.k2)
                        
        self.t3 = int (abs((log(1024, 2) + 1) / 2))
        self.k3 = self.t3 if self.t3 % 2 else self.t3 + 1
        self.da_eca_module3 = DA_ECA(self.k3)
        #
        self._load_pretrained_model()
    def _make_stage(self, planes, num_blocks, kernel_size, stride, padding):
        strides = [stride] + [1] *(num_blocks - 1)
        # change channel
        if self.in_planes==3 and planes==64:
            planes = [64, 64, 64]

        blocks = []
        for idx, stride in enumerate(strides):
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes[idx], kernel_size=kernel_size, stride=stride, padding=padding, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes[idx]
            self.cur_layer_idx += 1

        return nn.ModuleList(blocks)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
        layers.append(block(self.in_planes, planes, cur_groups, self.deploy, self.use_se, stride=stride, dilation=dilation, downsample=downsample, BatchNorm=BatchNorm))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, cur_groups, self.deploy, self.use_se, stride=1, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
        layers.append(block(self.in_planes, planes, cur_groups, self.deploy, self.use_se, stride=1, dilation=blocks[0] * dilation, downsample=downsample, BatchNorm=BatchNorm))
        self.in_planes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.in_planes, planes, cur_groups, self.deploy, self.use_se, stride=1, dilation=blocks[i] * dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        conv = []
        idx = 0
        #
        for stage in (self.layer0):
            if idx == 0:
                out = stage(x)
                idx += 1
            else:
                out = stage(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        low_level_feat = out
        out = self.da_eca_module1(out)        
        out = self.layer2(out)
        out = self.da_eca_module2(out)        
        out = self.layer3(out)
        out = self.da_eca_module3(out)        
        out = self.layer4(out)        
        return out, low_level_feat

    def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
        if do_copy:
            model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        if save_path is not None:
            torch.save(model.state_dict(), save_path)

        return model

    def _load_pretrained_model(self):
        # if you try 7X7 & RepVGG_bottleneck you should change this code
        if self.deploy == True:
            pass
        else:
            if self.backbone_model_name == 'resnet50':
                pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-0676ba61.pth')
                print("ResNet50_weight")
            elif self.backbone_model_name == 'resnet101':
                pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
                print("ResNet101_weight")
            #
            state_dict = self.state_dict()
            pretrain_keys = list(pretrain_dict.keys())
            state_dict_keys = list(state_dict.keys())
            del pretrain_keys[0:5]
            del state_dict_keys[0:41]
            #
            index = []
            for i in range(len(state_dict_keys)):
                if 'rbr_identity' in state_dict_keys[i] or 'rbr_1x1' in state_dict_keys[i]:
                    index.append(i)
            state_dict_match_keys = []
            for i in range(len(state_dict_keys)):
                if i in index:
                    pass
                else:
                    state_dict_match_keys.append(state_dict_keys[i])
            #
            model_dict = {}
            for i in pretrain_keys:
                if i in state_dict_match_keys:
                    model_dict[i] = pretrain_dict[i]

                else:
                    if '.weight' in i:
                        a, b = i.split('.weight')
                        if a + '.0.rbr_dense.conv' + '.weight' in state_dict_match_keys:
                            model_dict[a + '.0.rbr_dense.conv' + '.weight'] =  pretrain_dict[i]

                    elif '.running_mean' in i:
                        a, b = i.split('.running_mean')
                        if a + '.0.rbr_dense.conv' + '.running_mean' in state_dict_match_keys:
                            model_dict[a + '.0.rbr_dense.conv' + '.running_mean'] =  pretrain_dict[i]

                    elif '.running_var' in i:
                        a, b = i.split('.running_var')
                        if a + '.0.rbr_dense.conv' + '.running_var' in state_dict_match_keys:
                            model_dict[a + '.0.rbr_dense.conv' + '.running_var'] =  pretrain_dict[i]


            try:
                state_dict.update(model_dict)
                self.load_state_dict(state_dict)
                print("Success Load weight !!")
            except:
                raise



if __name__=='__main__':

    x = torch.randn(1, 3, 256, 256)
    my_model = RepVGG_ResNet([3, 4, 6, 3], 'resnet101',21, [1,1,1,1], None, True)
    out, low_level_feature = my_model(x)
