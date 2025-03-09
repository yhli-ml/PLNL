import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def assump_free_loss(f, K, labels, ccp):
    """Assumption free loss (based on Thm 1) is equivalent to non_negative_loss if the max operator's threshold is negative inf."""
    return non_negative_loss(f=f, K=K, labels=labels, ccp=ccp, beta=np.inf)

def non_negative_loss(f, K, labels, ccp, beta):
    labels = torch.argmax(labels, dim=1)
    ccp = torch.from_numpy(ccp).float().cuda()
    neglog = -F.log_softmax(f, dim=1)
    loss_vector = torch.zeros(K, requires_grad=True).cuda()
    temp_loss_vector = torch.zeros(K).cuda()
    for k in range(K):
        idx = labels == k
        if torch.sum(idx).item() > 0:
            idxs = idx.byte().view(-1,1).repeat(1,K)
            neglog_k = torch.masked_select(neglog, idxs).view(-1,K)
            temp_loss_vector[k] = -(K-1) * ccp[k] * torch.mean(neglog_k, dim=0)[k]  # average of k-th class loss for k-th comp class samples
            loss_vector = loss_vector + torch.mul(ccp[k], torch.mean(neglog_k, dim=0))  # only k-th in the summation of the second term inside max 
    loss_vector = loss_vector + temp_loss_vector
    count = np.bincount(labels.data.cpu()).astype('float')
    while len(count) < K:
        count = np.append(count, 0) # when largest label is below K, bincount will not take care of them
    loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).cuda()-beta), 1)
    max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
    final_loss = torch.sum(max_loss_vector)
    return final_loss, torch.mul(torch.from_numpy(count).float().cuda(), loss_vector)

def forward_loss(f, K, labels):
    Q = torch.ones(K,K) * 1/(K-1)
    Q = Q.cuda()
    for k in range(K):
        Q[k,k] = 0
    q = torch.mm(F.softmax(f, 1), Q)
    return -torch.sum(torch.log(q) * labels) / len(labels)

def pc_loss(f, K, labels):
    sigmoid = nn.Sigmoid()
    n = f.shape[0]
    con_num = labels.sum(dim=1).view(-1, 1)
    max_con_labels = int(max(con_num).item())
    loss_matrix = torch.zeros_like(f)
    f_con = labels * f
    f_con_value_list = []
    for row in f_con:
        non_zero_indices = torch.nonzero(row).squeeze(1)
        non_zero_values = row[non_zero_indices]
        f_con_value_list.append(non_zero_values)
    f_con_value = torch.nn.utils.rnn.pad_sequence(f_con_value_list, batch_first=True)
    f_con_value_padded = torch.nn.functional.pad(f_con_value, (0, 10 - f_con_value.shape[1]))
    for i in range(max_con_labels):
        fbar = f_con_value_padded[:, i].view(-1, 1).repeat(1, K)
        zero_indices = (fbar == 0) # 0 denotes this instance's cl run out
        fbar_ = fbar.clone()
        fbar_[zero_indices] = float('inf')
        loss_matrix += sigmoid(-1.*(f - fbar_)) # multiply -1 for "complementary"
    M1, M2 = K*(K-1)/2, K-1
    l_m, c_n = loss_matrix.clone(), con_num.repeat(1, K)
    loss_matrix = l_m / c_n
    pc_loss = -(K-1) * torch.sum(loss_matrix)/len(labels) + M1
    return pc_loss

# upper bounded exp loss
def ub_exp_loss(outputs, comp_y):
    comp_y = 1 - comp_y
    k = comp_y.shape[1]
    can_num = comp_y.sum(dim=1).float() # the number of non-complementary label
    
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * comp_y

    average_loss = ((k-1)/(k-can_num) * torch.exp(-final_outputs.sum(dim=1))).mean() # k-can_num denotes the number of complementary label
    return average_loss

# upper bounded log loss
def ub_log_loss(outputs, comp_y):
    comp_y = 1 - comp_y # note that this comp_y need to be reversed
    k = comp_y.shape[1]
    can_num = comp_y.sum(dim=1).float() # n
    
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * comp_y + 1e-8
    
    average_loss = - ((k-1)/(k-can_num) * torch.log(final_outputs.sum(dim=1))).mean()
    return average_loss

# supervised loss
def supervised_loss(outputs, targets):
    if outputs.size(0) == 0:
        return 0
    loss = F.binary_cross_entropy_with_logits(outputs, targets)
    return loss

# surrogate complementary exp loss
def scl_exp_loss(outputs, comp_y):
    probs = F.softmax(outputs, dim=1)
    exp_probs = torch.exp(probs)
    comp_exp_probs_sum = torch.sum(exp_probs * comp_y, dim=1)
    final_loss = torch.mean(comp_exp_probs_sum)
    return final_loss

# surrogate complementary log loss
def scl_log_loss(outputs, comp_y):
    probs = F.softmax(outputs, dim=1)
    log_probs = torch.log(1.0-probs+1e-8) # This is modified cross entropy. Use 1 minus probs for "complementary".
    comp_log_probs_sum = torch.sum(log_probs * comp_y, dim=1)
    final_loss = -torch.mean(comp_log_probs_sum)
    return final_loss
    
def EvaluatePLGPrecision(train_dataset_t, train_dataset_s, num_samples, num_classes, epoch):
    confidence_t = train_dataset_t.confidence
    confidence_s = train_dataset_s.confidence
    true_labels_t = train_dataset_t.true_labels
    true_labels_s = train_dataset_s.true_labels
    empirical_confident_true_labels_t = train_dataset_t.confident_true_labels
    empirical_confident_true_labels_s = train_dataset_s.confident_true_labels
    num_confident_samples_t = torch.sum(confidence_t).int()
    num_confident_samples_s = torch.sum(confidence_s).int()
    masks_t = torch.unsqueeze(confidence_t, dim=1).repeat(1, num_classes)
    masks_s = torch.unsqueeze(confidence_s, dim=1).repeat(1, num_classes)
    real_confident_true_labels_t = true_labels_t * masks_t
    real_confident_true_labels_s = true_labels_s * masks_s
    equal_rows_t = torch.all(torch.eq(empirical_confident_true_labels_t, real_confident_true_labels_t), dim=1).float() * confidence_t
    equal_rows_s = torch.all(torch.eq(empirical_confident_true_labels_s, real_confident_true_labels_s), dim=1).float() * confidence_s
    num_equal_rows_t = torch.sum(equal_rows_t)
    num_equal_rows_s = torch.sum(equal_rows_s)
    precision_rate_t = num_equal_rows_t / num_confident_samples_t
    precision_rate_s = num_equal_rows_s / num_confident_samples_s
    positive_precision = (precision_rate_t + precision_rate_s) / 2
    num_confident_samples = torch.floor((num_confident_samples_t+num_confident_samples_s)/2)
    print('Epoch: [{epoch}]\tmodel: positive precision:{:.2%}\t\tconfident samples :{}'.format(positive_precision, num_confident_samples, epoch=epoch))
    selected_ratio_t = torch.sum(train_dataset_t.confidence) / num_samples
    selected_ratio_s = torch.sum(train_dataset_s.confidence) / num_samples
    selected_ratio = (selected_ratio_t + selected_ratio_s) / 2
    valid_samples = num_confident_samples * positive_precision
    return selected_ratio, positive_precision, num_confident_samples, valid_samples

def EvaluateNLEPrecision(train_loader_t, train_loader_s, ori_comp_labels_matrix, true_labels_matrix, epoch):
    enhanced_comp_labels_matrix_t = train_loader_t.dataset.comp_labels
    enhanced_comp_labels_matrix_s = train_loader_s.dataset.comp_labels
    enhanced_part_t = enhanced_comp_labels_matrix_t - ori_comp_labels_matrix
    enhanced_part_s = enhanced_comp_labels_matrix_s - ori_comp_labels_matrix
    total_enhance_num = torch.floor((torch.sum(enhanced_part_t)+torch.sum(enhanced_part_s))/2)
    ave_num_exp_t = torch.sum(enhanced_part_t) / ori_comp_labels_matrix.size(0)
    ave_num_exp_s = torch.sum(enhanced_part_s) / ori_comp_labels_matrix.size(0)
    error_rate_t = torch.nonzero(torch.sum(enhanced_part_t * true_labels_matrix, dim=1) != 0).shape[0] / ori_comp_labels_matrix.size(0)
    error_rate_s = torch.nonzero(torch.sum(enhanced_part_s * true_labels_matrix, dim=1) != 0).shape[0] / ori_comp_labels_matrix.size(0)
    negative_precision_t = 1 - error_rate_t
    negative_precision_s = 1 - error_rate_s
    negative_precision = (negative_precision_t + negative_precision_s) / 2
    ave_num_exp = (ave_num_exp_t + ave_num_exp_s) / 2
    print('Epoch: [{epoch}]\tmodel: negative precision:{:.2%}\taverage number of enhancement:{:.5f}'.format(negative_precision, ave_num_exp, epoch=epoch))
    return negative_precision, ave_num_exp, total_enhance_num

# def EvaluatePLGPrecision(train_dataset, num_samples, num_classes, epoch):
#     confidence = train_dataset.confidence
#     true_labels = train_dataset.true_labels
#     empirical_confident_true_labels = train_dataset.confident_true_labels
#     num_confident_samples = torch.sum(confidence).int()
#     masks = torch.unsqueeze(confidence, dim=1).repeat(1, num_classes)
#     real_confident_true_labels = true_labels * masks
#     equal_rows = torch.all(torch.eq(empirical_confident_true_labels, real_confident_true_labels), dim=1).float() * confidence
#     num_equal_rows = torch.sum(equal_rows)
#     precision_rate = num_equal_rows / num_confident_samples
#     print('Epoch: [{epoch}]\tpositive precision:{:.2%}\t\tconfident samples :{}'.format(precision_rate, num_confident_samples, epoch=epoch))
#     selected_ratio = torch.sum(train_dataset.confidence) / num_samples # dataset selected by model_s
#     return selected_ratio, precision_rate


# def EvaluateNLEPrecision(train_loader, ori_comp_labels_matrix, true_labels_matrix, epoch):
#     enhanced_comp_labels_matrix = train_loader.dataset.comp_labels
#     enhanced_part = enhanced_comp_labels_matrix - ori_comp_labels_matrix
#     ave_num_exp = torch.sum(enhanced_part) / ori_comp_labels_matrix.size(0)
#     error_rate = torch.nonzero(torch.sum(enhanced_part * true_labels_matrix, dim=1) != 0).shape[0] / ori_comp_labels_matrix.size(0)
#     negative_precision = 1 - error_rate
#     print('Epoch: [{epoch}]\tmodel: negative precision:{:.2%}\taverage number of enhancement:{:.5f}\n'.format(negative_precision, ave_num_exp, epoch=epoch))
#     return negative_precision