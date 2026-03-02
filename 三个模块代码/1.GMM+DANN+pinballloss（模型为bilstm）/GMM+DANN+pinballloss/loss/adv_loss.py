import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


class LambdaSheduler(nn.Module):
    def __init__(self, gamma=10.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, domain_num=2):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        multlayer = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, domain_num),
            nn.BatchNorm1d(domain_num),
            nn.Softmax(dim=1)
        ]
        self.layers = torch.nn.Sequential(*multlayer)

    def forward(self, x):
        return self.layers(x)


class AdversarialLoss(nn.Module):
    '''
    Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
    '''

    def __init__(self, gamma=1.0, max_iter=1000, domain_num=3, use_lambda_scheduler=True, smooth=False, **kwargs):
        super(AdversarialLoss, self).__init__()
        self.domain_num = domain_num
        self.domain_classifier = Discriminator(domain_num=domain_num)
        
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)
        self.lambd = 1.0
        self.smooth_mode = smooth

    def forward(self, sfc, domain_label, epoch_ratio=None):
        if epoch_ratio is not None and self.smooth_mode is not False:
            smooth_param = 1.0 - (domain_label.shape[-1] - 1) / domain_label.shape[-1] * epoch_ratio
            smooth_label = self.get_smooth_label(domain_label, smooth_param)
        else:
            smooth_label = domain_label
            
        loss_adv, _ = self.cal_loss(sfc, smooth_label)
        return loss_adv

    def cal_loss(self, fc, domain_label):
        x = ReverseLayerF.apply(fc, self.lambd)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        loss_fn = nn.CrossEntropyLoss()
        loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv, domain_pred

    def get_smooth_label(self, domain_label, smooth_param):
        smooth_label = torch.clone(domain_label)
        smooth_label[domain_label == 1] = smooth_param
        smooth_label[domain_label == 0] = (1 - smooth_param) / (domain_label.shape[-1] - 1)
        return smooth_label
