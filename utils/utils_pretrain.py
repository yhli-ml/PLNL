import os
import random
import math
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
from utils.utils_log import *
from models.preactresnet import *

best_Acc1 = 0

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
        Computes the Accuracy@k for the specified values of k
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

def pretrain(args, train_loader, model, optimizer, epoch, test_loader):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    model.train()
    for i, (each_img, each_img_aug1, each_img_aug2, each_comp_label, 
            each_confidence, each_confident_true_label, index) in enumerate(train_loader):
        each_comp_label = each_comp_label.cuda()
        each_img = each_img.cuda()
        each_output = model(each_img)
        comp_loss = scl_log_loss(outputs=each_output, comp_y=each_comp_label)
        optimizer.zero_grad()
        comp_loss.backward()
        optimizer.step()
        losses.update(comp_loss.item(), each_img.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 50 == 0:
            print('[{time}] - '
                  'Pretrain Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))) 
    if epoch == args.pre_eps:
        if args.distr==0:
            model_path = './model_path/{}_distr{}_nc{}_pretrain.pth'.format(
                args.dataset, args.distr, args.nc, args.pre_eps)
        else:
            model_path = './model_path/{}_distr{}_pretrain.pth'.format(
                args.dataset, args.distr, args.pre_eps)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    """
    Run one test epoch
    """
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
            _, output = model(input_var)
            Acc1 = accuracy(output.data, target)[0]
            top1.update(Acc1.item(), input.size(0))
            # measure elapsed time              
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 50 == 0:
                print('[{time}] - '
                    'Pretrain Test: Epoch[{2}] [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(test_loader), epoch, batch_time=batch_time, loss=losses,
                        top1=top1, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print('[{time}] - * Acc@1 {top1.avg:.3f}'.format(top1=top1, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
