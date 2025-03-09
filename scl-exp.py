import random
import numpy as np
import time
from datetime import datetime
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from utils.utils_data import *
from utils.utils_algo import *
from models.preactresnet import *
import torchvision
import wandb

parser = argparse.ArgumentParser(description='Learning from Multiple Complementary Labels(LOG loss)')

parser.add_argument('-dataset', type=str, choices=['mnist', 'kmnist', 'fmnist', 'tinyimagenet', 'cifar10', 'cifar100', 'svhn', 'stl'], required=True)
parser.add_argument('-distr', type=int, help='usage of uniform distribution',required=True)
parser.add_argument('-nc', default=1, type=int, help="number of complementary labels")
parser.add_argument('-me', default='SCL-EXP', type=str)

parser.add_argument('-eps', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-bs', default=64, type=int)
parser.add_argument('-lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('-wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('-lam', default=1, type=float)
parser.add_argument('-gpu', default=0, type=int, required=True)
parser.add_argument('-model', default='preact', type=str)
parser.add_argument('-data-dir', default='./data/', type=str)
parser.add_argument('-seed', default=0, type=int)

args = parser.parse_args()
print(args)
wandb.init(project='Compared Methods', config=args, name=f'{args.me}_{args.dataset}_distr{args.distr}_nc{args.nc}')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.cuda.set_device(args.gpu)

best_prec1 = 0

class AverageMeter(object):
    """
        Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
        Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def comp_train(train_loader, model, optimizer, epoch):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.train()
    for i, (x_aug0, x_aug1, x_aug2, comp_y, _, __, ___) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, comp_y = x_aug0.cuda(), comp_y.float().cuda()
        outputs = model(images)
        loss = scl_exp_loss(outputs=outputs, comp_y=comp_y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('[{time}] - '
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return losses.avg

def test(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print('[{time}] - '
                      'Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(test_loader), batch_time=batch_time, loss=losses,
                          top1=top1, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    print('[{time}] - * Prec@1 {top1.avg:.3f}'.format(top1=top1, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return top1.avg, losses.avg


def scl_exp_training():
    global args, best_prec1

    # load data
    train_loader, _, test_loader = My_dataloaders(args)
    train_dataset = train_loader.dataset
    num_classes = train_dataset.k
    # load model
    if args.model == 'preact':
        if args.dataset in ['stl', 'tinyimagenet']:
            model = torchvision.models.resnet18(num_classes=num_classes)
        elif args.dataset in ['kmnist', 'fmnist']:
            model = ResNet18_mnist(num_classes=num_classes)
        else:
            model = PreActResNet18(num_classes=num_classes)
        
    model = model.cuda()
    
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
    
    #  cuDNN find best suitable cnn automatically
    cudnn.benchmark = True

    # Train loop
    for epoch in range(0, args.eps):

        print('[{time}] - current lr {:.5e}'.format(optimizer.param_groups[0]['lr'], time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # training
        trainloss = comp_train(train_loader, model, optimizer, epoch)
        # lr_step
        scheduler.step()
        # evaluate on validation set
        valacc, valloss = test(test_loader, model, criterion, epoch)
        wandb.log({"Accuracy":valacc})


if __name__ == '__main__':
    scl_exp_training()


