import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Function, Variable

from quantization import conv2d_quantize_fn

K=32

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Pact(Function):
    @staticmethod
    def forward(ctx, x, alpha, k):
        ctx.save_for_backward(x, alpha)
        # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        y = torch.clamp(x, min=0, max=alpha.item())
        scale = (2 ** k - 1) / alpha
        y_q = torch.round(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        x, alpha, = ctx.saved_tensors
        lower_bound = x < 0
        upper_bound = x > alpha
        x_range = ~(lower_bound | upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
        return dLdy_q * x_range.float(), grad_alpha, None


def clip_relu(x, clip_value=10):
    return F.relu(x).clip(max=clip_value)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', name=None, mask=None, inner=None):
        assert name is not None, 'name is empty'
        assert mask is not None, 'mask is empty'
        assert inner is not None, f'inner is empty'

        super(BasicBlock, self).__init__()
        conv2d = conv2d_quantize_fn(mask[f'{name}.{inner}.conv1.weight'])
        self.conv1 = conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        # batch_norm = bn_quantize_fn(mask[f'{name}.{inner}.bn1.weight'])
        # self.bn1 = batch_norm(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.alpha1 = nn.Parameter(torch.tensor(10.))

        conv2d = conv2d_quantize_fn(mask[f'{name}.{inner}.conv2.weight'])
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # batch_norm = bn_quantize_fn(mask[f'{name}.{inner}.bn2.weight'])
        # self.bn2 = batch_norm(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.alpha2 = nn.Parameter(torch.tensor(10.))

        self.pact = Pact.apply
        self.full_quant=False


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            # elif option == 'B':
            #     self.shortcut = nn.Sequential(
            #          nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            #          nn.BatchNorm2d(self.expansion * planes)
            #     )

    def forward(self, x):
        out = self.pact(self.bn1(self.conv1(x,self.full_quant)),self.alpha1,K)
        out = self.bn2(self.conv2(out,self.full_quant))
        out += self.shortcut(x)
        out = self.pact(out,self.alpha2,K)
        return out


class ResNets(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, mask=None):
        assert mask is not None, 'mask is empty'

        super(ResNets, self).__init__()
        self.in_planes = 16

        conv2d = conv2d_quantize_fn(mask['conv1.weight'])

        self.conv1 = conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # batch_norm2d = bn_quantize_fn(mask['bn1.weight'])

        # self.bn1 = batch_norm2d(16)
        self.bn1 = nn.BatchNorm2d(16)
        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.pact = Pact.apply
        self.full_quant=False

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, mask=mask, name="layer1")
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, mask=mask, name="layer2")
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, mask=mask, name="layer3")

        # linear = linear_quantize_fn(mask['fc.weight'])
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, name, mask=None):
        assert name is not None, 'name is empty'
        assert mask is not None, 'mask is empty'

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for inner_idx, stride in enumerate(strides):
            layers.append(BasicBlock(self.in_planes, planes, stride, name=name, inner=inner_idx, mask=mask))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pact(self.bn1(self.conv1(x,self.full_quant)),self.alpha1,K)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet20(mask, num_classes=10):
    assert mask is not None, 'mask is empty'
    return ResNets(BasicBlock, [3, 3, 3], num_classes=num_classes, mask=mask)


def resnet56(mask, num_classes=10):
    assert mask is not None, 'mask is empty'
    return ResNets(BasicBlock, [9, 9, 9], num_classes=num_classes)


if __name__ == '__main__':
    from lth import lth

    model = resnet20()
    lt = lth(model=model, percentage=10)

    pruned_model, mask = lt.generate_new_mask(model)

    model = resnet20(mask=mask)

    names = [n for n in mask.keys() if 'weight' in n]
    print(names)
