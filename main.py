import os
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
from utils.utils_log import *
from utils.utils_pretrain import *
from models.preactresnet import *
from copy import deepcopy
import faiss
import wandb

parser = argparse.ArgumentParser(
	prog='complementary label learning',
	usage='learning with complementary labels',
	description='My method version 26',
	epilog='end',
	add_help=True)

parser.add_argument('-dataset', type=str, choices=['mnist', 'kmnist', 'fmnist', 'tinyimagenet', 'cifar10', 'cifar100', 'svhn', 'stl'], required=True)
parser.add_argument('-distr', type=int, help='usage of uniform distribution', required=True)
parser.add_argument('-nc', default=1, type=int, help="number of complementary labels")

parser.add_argument('-lam', default=0.99, type=float, help='confidence threshold for plg')
parser.add_argument('-lam2', default=0.90, type=float, help='confidence threshold for nle')
parser.add_argument('-k', default=500, type=int, help='the number of nearest neighbours')
parser.add_argument('-t', default=0.1, type=float, help='the extent of enhancement for each sample')
parser.add_argument('-alpha', default=0.9, type=float)

parser.add_argument('-eps', default=200, type=int, metavar='N', help='number of training epochs')
parser.add_argument('-weps', default=50, type=int, metavar='N', help='number of training epochs')
parser.add_argument('-neps', default=150, type=int, metavar='N', help='number of training epochs')
parser.add_argument('-bs', default=64, type=int, help='size of one mini-batch')
parser.add_argument('-lr', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('-wd', default=1e-4, type=float, help='weight decay')

parser.add_argument('-model', default='resnet', type=str, help='backbone model')
parser.add_argument('-gpu', default=0, type=int, help='one gpu available', required=True)
parser.add_argument('-seed', default=0, type=int, help='random seed')
parser.add_argument('-data-dir', default='./data', type=str)

args = parser.parse_args()
print(args)
if args.dataset=='tinyimagenet':
    args.bs=128

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.cuda.set_device(args.gpu)

file_path = __file__
raw_file_name = os.path.basename(file_path)
file_name = os.path.splitext(raw_file_name)[0]

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

def train(train_loader, model, model1, optimizer, optimizer1, epoch, threshold_w, threshold_s, comp_labels):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    model.train()
    outputs = torch.zeros(args.num_ins, args.num_cls)
    outputs1 = torch.zeros(args.num_ins, args.num_cls)
    threshold_w = threshold_w.cuda()
    threshold_s = threshold_s.cuda()
    hc_num=0
    sc_num=0
    uc_num=0
    for i, (each_img_w, each_img_s, each_comp_label, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        each_comp_label = each_comp_label.float().cuda()
        each_img_w = each_img_w.cuda()
        each_img_s = each_img_s.cuda()
        each_feature, each_output = model.extract_features(each_img_w)
        each_feature1, each_output1 = model1.extract_features(each_img_s)
        prob, _ = torch.max(F.softmax(each_output, dim=1), dim=1)
        prob1, _ = torch.max(F.softmax(each_output1, dim=1), dim=1)
        if epoch >= args.neps:
            with torch.no_grad():
                for j in range(each_img_w.size(0)):
                    outputs[index[j]] = each_output[j]
                    outputs1[index[j]] = each_output1[j]
        threshold_w[index] = args.alpha * threshold_w[index] + (1 - args.alpha) * prob
        threshold_s[index] = args.alpha * threshold_s[index] + (1 - args.alpha) * prob1

        hc_ind = torch.nonzero((prob >= threshold_w[index]) & (prob1 >= threshold_s[index])).squeeze(1)
        sc_ind = torch.nonzero(((prob >= threshold_w[index]) & (prob1 < threshold_s[index])) | ((prob < threshold_w[index]) & (prob1 >= threshold_s[index]))).squeeze(1)
        uc_ind = torch.nonzero((prob < threshold_w[index]) & (prob1 < threshold_s[index])).squeeze(1)

        hc_num+=hc_ind.numel()
        sc_num+=sc_ind.numel()
        uc_num+=uc_ind.numel()
        # Positive Label Guessing 
        hc_target = torch.argmax(each_output[hc_ind], dim=1)

        criterion = nn.CrossEntropyLoss()

        sup_loss = 0
        sup_loss1 = 0
        if epoch >= args.weps:
            sup_loss = criterion(each_output[hc_ind], hc_target)
            sup_loss1 = criterion(each_output1[hc_ind], hc_target)
        # complementary loss
        comp_loss = scl_log_loss(outputs=each_output, comp_y=each_comp_label)
        comp_loss1 = scl_log_loss(outputs=each_output1, comp_y=each_comp_label)
        
        loss = sup_loss + comp_loss
        loss1 = sup_loss1 + comp_loss1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        losses.update(loss.item(), each_img_w.size(0))
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
        # NLE
    print(f"hc_num:{hc_num}\n"
          f"sc_num:{hc_num}\n"
          f"uc_num:{uc_num}")
    if epoch >= args.neps:
        with torch.no_grad():
            # negative label enhancement
            cur_k_indices = similarity(outputs)
            cur_k_indices1 = similarity(outputs1)
            # computation of complementary labels intensity
            intensity_matrix=0.5*(torch.sum(comp_labels[cur_k_indices], dim=1) + torch.sum(comp_labels[cur_k_indices1], dim=1))
            # intensity_matrix=0.1*torch.sum(oclm[cur_k_indices], dim=1)+0.9*pre_epoch_intensity
            # pre_epoch_intensity=intensity_matrix.clone()
            # find number of expansion for each sample
            num_non_comp_labels = (args.num_cls - 1) - torch.sum(comp_labels, dim=1)
            num_exp = torch.ceil(args.t * num_non_comp_labels).long()
            # find top (num_exp) class for each sample
            enhanced_comp_labels_matrix = deepcopy(comp_labels)
            for i in range(intensity_matrix.size(0)):
                # enhance only for unconfident samples but not too unconfident
                # if train_loader.dataset.confidence[i]==0 and torch.max(probs[i])>args.lam2:
                if train_loader.dataset.confidence[i]==0:
                    k = num_exp[i].item()
                    _, top_indices_row = torch.topk(intensity_matrix[i], k=k)
                    enhanced_comp_labels_matrix[i][top_indices_row] = 1
            train_loader.dataset.comp_labels=enhanced_comp_labels_matrix
            


def test(test_loader, model, model1, criterion, epoch):
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
            # compute output
            _, output = model.extract_features(input_var)
            _, output1 = model1.extract_features(input_var)
            output = 0.5*(output+output1)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            Acc1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(Acc1.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 50 == 0:
                print('[{time}] - '
                      'Test: Epoch[{2}] [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(test_loader), epoch, batch_time=batch_time, loss=losses,
                          top1=top1, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print('[{time}] - * Acc@1 {top1.avg:.3f}'.format(top1=top1, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return top1.avg


def similarity(logits):
    logits = F.normalize(logits, dim=1)
    index = faiss.IndexFlatL2(logits.shape[1])
    index.add(logits)
    V, I = index.search(logits, args.k+1)
    k_indices = torch.from_numpy(I[:, 1:args.k+1])
    return k_indices


def training_pipeline():
    """
        run one complete training pipeline
    """
    global args, best_Acc1
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
    # load data
    train_loader, test_loader = My_dataloaders(args)
    train_dataset = train_loader.dataset
    args.num_ins = train_dataset.n
    args.num_cls = train_dataset.k
    comp_labels = deepcopy(train_dataset.comp_labels)
    
    if args.dataset == 'stl':
        args.lr = 0.01
    # cuDNN find best suitable cnn automatically
    cudnn.benchmark = True
    # load model
    if args.model == 'resnet':
        if args.dataset in ['cifar10', 'cifar100', 'svhn']:
            model = PreActResNet18(num_classes=args.num_cls)
        elif args.dataset in ['stl', 'tinyimagenet']: # 96*96 || 224*224 large image
            model = ResNet18_feature(num_classes=args.num_cls)
        elif args.dataset in ['fmnist', 'kmnist', 'mnist']:
            model = ResNet18_feature_mnist(num_classes=args.num_cls)
    model1 = deepcopy(model)
    model = model.cuda()
    model1 = model1.cuda()
        
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # learning rate scheduler -- cosine annealing   
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch= -1)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[100, 150], last_epoch= -1)

    threshold_w = torch.zeros(args.num_ins)
    threshold_s = torch.zeros(args.num_ins)
        
    for epoch in range(args.eps):
        print('[{time}] - Epoch: [{}]\tcurrent model lr {:.5e}\t'.format(epoch, optimizer.param_groups[0]['lr'], time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # training
        train(train_loader, model, model1, optimizer, optimizer1, epoch, threshold_w, threshold_s, comp_labels)
        # lr_step
        scheduler.step()
        # evaluate on test dataset
        test(test_loader, model, model1, criterion, epoch)


if __name__ == '__main__':
    training_pipeline()