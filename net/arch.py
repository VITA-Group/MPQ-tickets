import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

torch.set_default_dtype(torch.float64)


# Function for Initialization
def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def imgnet_transform(is_training=True):
    if is_training:
        transform_list = transforms.Compose([transforms.RandomResizedCrop(64),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.5,
                                                                    contrast=0.5,
                                                                    saturation=0.3),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    else:
        transform_list = transforms.Compose([transforms.Resize(73),
                                             transforms.CenterCrop(64), #similar to 256/224 ratio
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    return transform_list


def cifar10_loaders(batch_size: int, num_workers=8):
    train_transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Pad(padding=4, padding_mode='reflect'),
                        transforms.RandomCrop(32, padding=0),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    test_transform=transform=transforms.Compose(
                                        [
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])
    train_dataset = Subset(datasets.CIFAR10('~/data', train=True, download=True, transform=train_transform),list(range(45000)))
    val_dataset = Subset(datasets.CIFAR10('~/data', train=True, download=True, transform=test_transform),list(range(45000,50000)))
    test_dataset = datasets.CIFAR10('~/data', train=False,download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader, test_loader


def cifar100_loaders(batch_size: int,num_workers=4):
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5088964127604166, 0.48739301317401956, 0.44194221124387256),
                             (0.2682515741720801, 0.2573637364478126, 0.2770957707973042))
    ])

    train_dataset = Subset(datasets.CIFAR100('~/data', train=True, download=True, transform=train_transform),list(range(45000)))
    val_dataset = Subset(datasets.CIFAR100('~/data', train=True, download=True, transform=test_transform),list(range(45000,50000)))
    test_dataset = datasets.CIFAR100('~/data', train=False,download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader, test_loader

def tiny_imagenet_dataloaders(batch_size: int,num_workers=4):
    traindir = os.path.join('~/data/tiny-imagenet-200', 'train')
    valdir = os.path.join('~/data/tiny-imagenet-200', 'val')
    train_dataset = datasets.ImageFolder(traindir, imgnet_transform(is_training=True))
    train_loader = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers,
                               pin_memory=True)

    test_dataset = datasets.ImageFolder(valdir, imgnet_transform(is_training=False))
    test_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)

    return train_loader, test_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, criterion, train_loader):
    EPS = 1e-6

    model.train()
    for i,(imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs.double())
        train_loss = criterion(output, targets)


        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data
                grad_tensor = p.grad
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor
        optimizer.step()
    return train_loss.item()


def test(model, test_loader,full_quant=False):
    model.eval()
    test_loss = 0
    correct = 0

    if full_quant:
        for m in model.modules():
            if hasattr(m,'full_quant'):
                m.full_quant=True

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.double())
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)


    if full_quant:
        for m in model.modules():
            if hasattr(m,'full_quant'):
                m.full_quant=False

    return accuracy

