import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        attn_weights = F.softmax(self.W(x), dim=-1)
        output = torch.matmul(attn_weights.transpose(0, 1), x)
        return output
