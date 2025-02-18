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
	epilog='end',
	add_help=True)

parser.add_argument('-dataset', type=str, choices=['mnist', 'kmnist', 'fmnist', 'tinyimagenet', 'cifar10', 'cifar100', 'svhn', 'stl10', 'clcifar10', 'clcifar20', '20ng'], required=True)
parser.add_argument('-distr', type=int, help='usage of uniform distribution', required=True)
parser.add_argument('-nc', default=1, type=int, help="number of complementary labels")

parser.add_argument('-lam', default=0.99, type=float, help='confidence threshold for plg')
parser.add_argument('-lam2', default=0.90, type=float, help='confidence threshold for nle')
parser.add_argument('-k', default=500, type=int, help='the number of nearest neighbours')
parser.add_argument('-t', default=0.1, type=float, help='the extent of enhancement for each sample')
parser.add_argument('-qs', default=3, type=int, help='queue size for sample selection')

parser.add_argument('-eps', default=200, type=int, metavar='N', help='number of training epochs')
parser.add_argument('-weps', default=20, type=int, metavar='N', help='number of warm up epochs for nle')
parser.add_argument('-bs', default=64, type=int, help='size of one mini-batch')
parser.add_argument('-lr', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('-wd', default=1e-4, type=float, help='weight decay')

parser.add_argument('-model', default='resnet', type=str, help='backbone model')
parser.add_argument('-pretrained', type=str, help='pretrain model directory')
parser.add_argument('-gpu', default=0, type=int, help='one gpu available', required=True)
parser.add_argument('-seed', default=0, type=int, help='random seed')
parser.add_argument('-data-dir', default='/nas/datasets', type=str)

args = parser.parse_args()
print(args)

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="CLL",

#     name=f'{args.dataset}_distr{args.distr}',
#     # track hyperparameters and run metadata
#     config={
#     "dataset": args.dataset,
#     "number of CLs": args.distr,
#     }
# )


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


def train(train_loader, model, optimizer, epoch, consistency_criterion, flag, ori_comp_labels_matrix):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    model.train()
    num_samples = train_loader.dataset.n
    num_classes = train_loader.dataset.k
    logits = torch.zeros(num_samples, num_classes)
    comp_labels_matrix = train_loader.dataset.comp_labels
    true_labels_matrix = train_loader.dataset.true_labels
    # construct output queue based on reverse flags(dual network cross selection)
    if flag == 's':
        reversed_flag = 't'
    else:
        reversed_flag = 's'
    previous_output_logits_queue = []
    if epoch >= args.weps:
        for ep in range(epoch - args.qs, epoch):
            if args.distr==0:
                file_path = './logits/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.pt'.format(
                    args.dataset, args.distr, args.nc, args.lam, args.qs, args.k, args.t, ep, reversed_flag)
            else:
                file_path = './logits/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.pt'.format(
                    args.dataset, args.distr, args.lam, args.qs, args.k, args.t, ep, reversed_flag)
            os.makedirs(file_path, exist_ok=False)
            previous_logits = torch.load(file_path)
            previous_output_logits_queue.append(previous_logits.clone())
    for i, (each_img, each_img_aug1, each_img_aug2, each_comp_label, 
            each_confidence, each_confident_true_label, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        each_comp_label = each_comp_label.float().cuda()
        each_confidence = each_confidence.cuda()
        each_confident_true_label = each_confident_true_label.float().cuda()
        # original image
        each_img = each_img.cuda()
        num_samples_batch = each_img.size(0)
        each_output = model(each_img)
        # complementary loss
        comp_loss = scl_log_loss(outputs=each_output, comp_y=each_comp_label)
        with torch.no_grad():
            for j in range(num_samples_batch):
                logits[index[j]] = each_output[j]
            # Positive Label Guessing
            if epoch >= args.weps:
                previous_output_prob_queue = [F.softmax(previous_output_logits_queue[i], dim=1) for i in range(args.qs)]
                previous_output_prob_queue_batch = [previous_output_prob_queue[i][index] for i in range(args.qs)]
                confident_samples = torch.ones(num_samples_batch)
                confident_samples = confident_samples.cuda()
                # criteria1
                mean = torch.mean(torch.stack(previous_output_prob_queue_batch),dim=0)
                mean_max_indices = torch.argmax(mean, dim=1)
                criteria1 = 1-each_comp_label[torch.arange(num_samples_batch), mean_max_indices]
                confident_samples *= criteria1
                # criteria2
                max_val, _ = torch.max(mean, dim=1)
                criteria2 = (max_val > args.lam).float().cuda()
                confident_samples *= criteria2
                # criteria3
                max_indices = [torch.argmax(matrix, dim=1) for matrix in previous_output_prob_queue_batch]
                all_same = torch.all(torch.stack([max_indices[0] == idx for idx in max_indices[1:]]), dim=0)
                criteria3 = all_same.int().cuda()
                confident_samples *= criteria3
                confident_true_labels = torch.zeros_like(each_comp_label)
                confident_true_labels[torch.arange(num_samples_batch), mean_max_indices] = 1
                masks = torch.unsqueeze(confident_samples, dim=1).repeat(1, each_comp_label.size(1))
                confident_true_labels = confident_true_labels * masks
                confident_true_labels = confident_true_labels.cuda()
                # update positive label information
                for j in range(num_samples_batch):
                    train_loader.dataset.confidence[index[j]] = confident_samples[j]
                    train_loader.dataset.confident_true_labels[index[j]] = confident_true_labels[j]
        confident_indices = torch.nonzero(each_confidence).squeeze(1)
        each_confident_output = each_output[confident_indices]
        each_unconfident_output = each_output[~each_confidence.bool()]
        # supervised loss
        each_confident_true_label = each_confident_true_label[confident_indices]
        sup_loss = supervised_loss(outputs=each_confident_output, targets=each_confident_true_label) # supervised training
        # weak augmentation
        each_img_aug1 = each_img_aug1.cuda()
        y_pred_aug1 = model(each_img_aug1)
        each_unconfident_y_pred_aug1 = y_pred_aug1[~each_confidence.bool()]
        # strong augmentation
        each_img_aug2 = each_img_aug2.cuda()
        y_pred_aug2 = model(each_img_aug2)
        each_confident_y_pred_aug2 = y_pred_aug2[confident_indices]
        each_unconfident_y_pred_aug2 = y_pred_aug2[~each_confidence.bool()]
        # preprocess for consistency loss
        y_pred_aug0_probas = torch.softmax(each_output, dim=-1)
        y_pred_aug1_probas_log = torch.log_softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas_log = torch.log_softmax(y_pred_aug2, dim=-1)
        each_unconfident_comp_label = each_comp_label[~each_confidence.bool()]
        revisedY0 = (1 - each_comp_label).clone()
        revisedY0 = revisedY0 * y_pred_aug0_probas
        revisedY0_sum = revisedY0.sum(dim=1)
        revisedY0_sum[revisedY0_sum == 0] = 1  # make sure the denominator not equal zero
        revisedY0 = revisedY0 / revisedY0_sum.unsqueeze(1)
        soft_positive_label0 = revisedY0.detach()
        # # Consistency loss
        # consist_loss1 = consistency_criterion(y_pred_aug1_probas_log, soft_positive_label0 + 1e-8) # strong -> weak
        # consist_loss2 = consistency_criterion(y_pred_aug2_probas_log, soft_positive_label0 + 1e-8) # strong -> weak
        # Unified loss
        balance_para1 = 1
        balance_para2 = min((epoch/100), 1)
        final_loss = sup_loss + balance_para1 * comp_loss # + balance_para2 * (consist_loss1 + consist_loss2)
        
        # back propagation
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        losses.update(final_loss.item(), each_img.size(0))
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
    # negative label enhancement
    if epoch >= args.weps:
        k_indices = similarity(logits)
        probs = F.softmax(logits, dim=1)
        max_value, _ = torch.max(probs, dim=1)
        mean_max_value = torch.mean(max_value)
        aboveAveFlag = (max_value > mean_max_value).float()
        # t = 0.1+(args.t-0.1) * (epoch / args.eps)
        enhanced_comp_labels_matrix = NLE(ori_comp_labels_matrix, k_indices, args.t, aboveAveFlag)
        train_loader.dataset.comp_labels = enhanced_comp_labels_matrix
    if args.distr==0:
        file_path = './logits/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.pt'.format(
            args.dataset, args.distr, args.nc, args.lam, args.qs, args.k, args.t, epoch, flag)
    else:
        file_path = './logits/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.pt'.format(
            args.dataset, args.distr, args.lam, args.qs, args.k, args.t, epoch, flag)
    os.makedirs(file_path, exist_ok=False)
    torch.save(logits, file_path)
    if args.distr == 0:
        model_path = './resume_model_path/{}_distr{}_nc{}_train_ep{}.pth'.format(
            args.dataset, args.distr, args.nc, epoch)
    else:
        model_path = './resume_model_path/{}_distr{}_train_ep{}.pth'.format(
            args.dataset, args.distr, epoch)


def test(test_loader, model_t, model_s, criterion, epoch):
    """
        Run one test epoch
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    model_t.eval()
    model_s.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()
            # compute output
            output1 = model_t(input_var)
            output2 = model_s(input_var)

            loss1 = criterion(output1, target_var)
            loss2 = criterion(output2, target_var)

            output = 0.5*(output1+output2).float()
            loss = 0.5*(loss1+loss2).float()
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


# def NLE(comp_labels_matrix, k_indices, t, num_classes): # true label was only used for evaluation not training
#     # computation of complementary labels intensity
#     intensity_matrix = torch.sum(comp_labels_matrix[k_indices], dim=1)
#     # find number of expansion for each sample
#     num_non_comp_labels = (num_classes - 1) - torch.sum(comp_labels_matrix, dim=1)
#     num_exp = torch.ceil(t * num_non_comp_labels).long()
#     # find top (num_exp) class for each sample
#     enhanced_comp_labels_matrix = torch.zeros_like(comp_labels_matrix) + comp_labels_matrix
#     for i in range(intensity_matrix.size(0)):
#         k = num_exp[i].item()
#         _, top_indices_row = torch.topk(intensity_matrix[i], k=k)
#         enhanced_comp_labels_matrix[i][top_indices_row] = 1
#     return enhanced_comp_labels_matrix, intensity_matrix


def NLE(comp_labels_matrix, k_indices, t, flag): # true label was only used for evaluation not training
    # computation of complementary labels intensity(cli)
    intensity_matrix = torch.sum(comp_labels_matrix[k_indices], dim=1)
    # find number of expansion for each sample
    num_non_comp_labels = torch.sum(1 - comp_labels_matrix, dim=1) - 1
    num_exp = torch.ceil(t * num_non_comp_labels).to(torch.int)
    # find top (num_exp) class for each sample
    enhanced_comp_labels_matrix = torch.zeros_like(comp_labels_matrix) + comp_labels_matrix
    for i in range(intensity_matrix.size(0)):
        if flag[i] == 1:
            k = num_exp[i].item()
        else:
            k = 1
        _, top_indices_col = torch.topk(intensity_matrix[i], k=k)
        enhanced_comp_labels_matrix[i][top_indices_col] = 1
    return enhanced_comp_labels_matrix


def training_pipeline():
    """
        run one complete training pipeline
    """
    global args, best_Acc1
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
    # load data
    train_loader_t, train_loader_s, test_loader = My_dataloaders(args)
    train_dataset = train_loader_t.dataset
    num_samples = train_dataset.n
    num_classes = train_dataset.k
    ori_comp_labels_matrix = deepcopy(train_loader_t.dataset.comp_labels)
    true_labels_matrix = train_dataset.true_labels
    
    if args.dataset == 'stl10':
        args.lr = 0.01
    # cuDNN find best suitable cnn automatically
    cudnn.benchmark = True
    # load model
    if args.model == 'resnet':
        if args.dataset in ['cifar10', 'cifar100', 'svhn', 'clcifar10', 'clcifar20']:
            model_t, model_s = PreActResNet18(num_classes=num_classes), PreActResNet18(num_classes=num_classes)
        elif args.dataset in ['stl10', 'tinyimagenet', '20ng']: # 96*96 || 224*224 large image
            model_t, model_s = models.resnet18(num_classes=num_classes), models.resnet18(num_classes=num_classes)
        elif args.dataset in ['fmnist', 'kmnist', 'mnist']:
            model_t, model_s = ResNet18_feature_mnist(num_classes=num_classes),ResNet18_feature_mnist(num_classes=num_classes)
    model_t, model_s = model_t.cuda(), model_s.cuda()
    if args.distr == 0:
        model_path = './model_path/{}_distr{}_nc{}_ep200.pth'.format(
            args.dataset, args.distr, args.nc)
    else:
        model_path = './model_path/{}_distr{}_ep200.pth'.format(
            args.dataset, args.distr)
    if not os.path.exists(model_path):
        model = deepcopy(model_t)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
        for epoch in range(200):
            pretrain(args, train_loader_t, model, optimizer, epoch, test_loader)
            scheduler.step()
    model_t.load_state_dict(torch.load(model_path, map_location='cpu'))
    model_s.load_state_dict(torch.load(model_path, map_location='cpu'))

    # optimizer
    optimizer_t = torch.optim.SGD(model_t.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd);optimizer_s = torch.optim.SGD(model_s.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # learning rate scheduler -- cosine annealing
    scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=200, eta_min=1e-3);scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=200, eta_min=1e-3)
    
    for epoch in range(args.eps):
        print('[{time}] - Epoch: [{}]\tcurrent model lr {:.5e}\t'.format(
            epoch, optimizer_t.param_groups[0]['lr'], time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        ## training
        train(train_loader_t, model_t, optimizer_t, epoch, consistency_criterion, 't', ori_comp_labels_matrix)
        train(train_loader_s, model_s, optimizer_s, epoch, consistency_criterion, 's', ori_comp_labels_matrix)
        
        # samples selections precision computation
        if epoch >= args.weps:
            selected_ratio, positive_precision, confident_samples, valid_samples = EvaluatePLGPrecision(train_loader_t.dataset, train_loader_s.dataset, num_samples, num_classes, epoch)
            negative_precision, ave_negative_enhancement, total_negative_enhancement = EvaluateNLEPrecision(train_loader_t, train_loader_s, ori_comp_labels_matrix, true_labels_matrix, epoch)
            # wandb.log({"Selected Ratio":selected_ratio, "Positive Precision":positive_precision, "Confident Samples":confident_samples, "Valid Samples":valid_samples, 
            #            "Negative Precision":negative_precision, "Average Enhancement Number":ave_negative_enhancement, "Total Enhancement Number":total_negative_enhancement})
        ## lr_step
        scheduler_t.step();scheduler_s.step()
        ## evaluate on test dataset
        accuracy_t = test(test_loader, model_t, model_s, criterion, epoch)
        #wandb.log({"Accuracy":accuracy})
    #wandb.finish()


if __name__ == '__main__':
    training_pipeline()