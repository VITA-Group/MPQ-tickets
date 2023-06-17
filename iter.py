import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from log import setup_file_logger, log
from lth import lth
from net.arch import cifar10_loaders, cifar100_loaders, tiny_imagenet_dataloaders, weight_init, test, train
from utils import safe_save, stats_mask
import torch.backends.cudnn as cudnn
import numpy as np
from setproctitle import setproctitle
setproctitle(os.getcwd().split('/')[-1])


cudnn.benchmark = True


torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sparsity(model,keys):
    all_value = [param.data.cpu().numpy() for name, param in model.named_parameters() if name in keys]
    alive = np.concatenate([e.flatten() for e in [tensor[np.nonzero(tensor)] for tensor in all_value]])
    total = np.concatenate([e.flatten() for e in all_value])
    return len(alive)/len(total)



def writer_log(key, value, key1=None, value1=None, global_step=0):
    writer.add_scalar(key, value, global_step=global_step)
    if key1!=None:
        writer.add_scalar(key1, value1, global_step=global_step)
        log(f'({key}, {value}, {key1}, {value1}, global_step={global_step})')

    else:
        log(f'({key}, {value}, global_step={global_step})')


def get_model_and_mask(num_classes):
    current_iter = args.current_iter
    save_dir = args.save_dir
    if current_iter == 0:
        # if first time, generate full precision mask, init weight, and save state_dict.
        if args.model == 'resnet20':
            from model.resnet.resnet import resnet20 as model_origin
            from model.resnet.resnet_q import resnet20 as model_q
        elif args.model == 'resnet56':
            from model.resnet.resnet import resnet56 as model_origin
            from model.resnet.resnet_q import resnet56 as model_q
        elif args.model == 'vgg16_bn':
            from model.vgg.vgg import vgg16_bn as model_origin
            from model.vgg.vgg_q import vgg16_bn as model_q
        else:
            print('model not supported')
            sys.exit(-1)

        mask = lth.init_mask(model_origin(num_classes=num_classes).to(device))
        if args.model == 'vgg16_bn':
            from model.vgg.vgg_q import get_vgg_mask
            mask = get_vgg_mask(model_origin(num_classes).to(device))

        model = model_q(num_classes=num_classes, mask=mask).to(device)
        print(model)



        safe_save.torch_save(model.state_dict(), os.path.join(save_dir, 'model', 'model_state_dict.pth'), log)
    else:
        # if not first time, load model, mask and state_dict.
        mask = safe_save.np_load(
            os.path.join(args.save_dir, 'mask', f'mask_{args.current_iter}_{args.down_percentage}.npy'))

        if args.model == 'resnet20':
            from model.resnet.resnet_q import resnet20 as model_q
        elif args.model == 'resnet56':
            from model.resnet.resnet_q import resnet56 as model_q
        elif args.model == 'vgg16_bn':
            from model.vgg.vgg_q import vgg16_bn as model_q
        else:
            print('model not supported')
            sys.exit(-1)

        model = model_q(mask=mask, num_classes=num_classes).to(device)

        if args.reinit == 'lt':
            init_state = torch.load(os.path.join(save_dir, 'model', 'model_state_dict.pth'))
            model.load_state_dict(init_state)
        else:
            # model has been initialized after created.
            pass

    for name, param in model.named_parameters():
        if name in mask.keys():
            param.data = (mask[name] != 0).int() * param.data

    return model, mask


def main():
    if args.datasets == 'cifar10':
        train_loader, val_loader, test_loader = cifar10_loaders(batch_size=args.batch_size, )
        num_classes = 10
    elif args.datasets == 'cifar100':
        train_loader, val_loader, test_loader = cifar100_loaders(batch_size=args.batch_size, )
        num_classes = 100
    elif args.datasets == 'tiny_imagenet':
        train_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size=args.batch_size, )
        num_classes = 200
    else:
        raise ValueError('datasets should be cifar10, cifar100 or imagenet')

    current_iter = args.current_iter
    model, mask = get_model_and_mask(num_classes)
    stats_mask(mask)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_schedu = torch.optim.lr_scheduler.MultiStepLR(optimizer, [91, 136], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    lt = lth(model=model, mask=mask, percentage=args.down_percentage)
    avg_bit = lth.get_avg_bit_of_mask(mask)

    writer_log(f'avg_bit', avg_bit, global_step=current_iter)

    best_v_accuracy = 0.0
    best_accuracy = 0.0


    print('===>before training sparsity:',get_sparsity(model,lt.mask.keys()))

    for epoch_iter in range(args.epoch_iter):
        loss = train(model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader)
        writer_log(f'{args.current_iter}_loss', loss, global_step=epoch_iter)
        v_accuracy = test(model, val_loader)
#        fq_v_accuracy = test(model, val_loader, full_quant=True)
        if args.datasets=='tiny_imagenet':
            accuracy = v_accuracy
        else:
            accuracy = test(model, test_loader)

#        fq_accuracy = test(model, test_loader, full_quant=True)


        lr_schedu.step()
        writer_log(f'{args.current_iter}_val_accuracy', v_accuracy, global_step=epoch_iter)
        writer_log(f'{args.current_iter}_accuracy', accuracy, f'{args.current_iter}_best_accuracy', best_accuracy, global_step=epoch_iter)
        if current_iter==0 and epoch_iter+1==args.rewind_epoch:
            safe_save.torch_save(model.state_dict(), os.path.join(args.save_dir, 'model', 'model_state_dict.pth'), log)
        if v_accuracy > best_v_accuracy:
            best_accuracy = accuracy
            best_v_accuracy = v_accuracy

            writer_log('best_acc', accuracy, global_step=args.current_iter)
            safe_save.torch_save(model.state_dict(),
                                 os.path.join(os.getcwd(),
                                              args.save_dir,
                                              'model',
                                              f'{args.current_iter}_{args.down_percentage}_state.pth.tar'),
                                 log=log)


    if args.mask_type == 'minimal':
        mask, bit = lt.generate_new_mask()
    elif args.mask_type == 'random':
        mask, bit = lt.generate_random_mask()
    else:
        raise ValueError('mask_type should be minimal or random')

    print('===>after pruning sparsity:',get_sparsity(model,lt.mask.keys()))

    stats_mask(mask)
    safe_save.np_save(data=mask, path=os.path.join(args.save_dir, 'mask',
                                                   f'mask_{args.current_iter + 1}_{args.down_percentage}.npy'),
                      log=log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse for iteratively quantization')
    parser.add_argument('--datasets', type=str, default='cifar10', help='cifar10 | cifar100 | tiny_imagenet')
    parser.add_argument('--model', type=str, default='resnet20', help='resnet20 | resnet56 | vgg16_bn')
    parser.add_argument('--current_iter', type=int, default=-1, help="current iteration, START FROM 0")
    parser.add_argument('--down_percentage', type=int, default=30, help='how much percent you want to quantization')
    parser.add_argument('--epoch_iter', type=int, default=182, help='how many epoch you want to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--mask_type', type=str, default='minimal', help='minimal | random')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--reinit', type=str, default='lt', help='how to reinitialize')
    parser.add_argument('--rewind_epoch', type=int, default=4,help='which epoch to rewind weights')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--save_dir', type=str, default='save', help='model save dir')
    parser.add_argument('--tensor_save_dir', type=str, default='tensorlog', help='tensorboard log save dir')
    parser.add_argument('--log_file', type=str, default='out.log', help='log file path')
    args = parser.parse_args()

    assert args.current_iter >= 0, 'current_iter is required, and PLEASE ENSURE it starts from 0'

    setup_file_logger(args.log_file)
    writer = SummaryWriter(log_dir=args.tensor_save_dir)

    main()
