import torch
import torch.nn.functional as F

class TwoWaySigmoidLoss(torch.nn.Module):
    def __init__(self, prior, margin=1.0, dist='softplus', tmpr=1):
        super(TwoWaySigmoidLoss, self).__init__()
        self.prior = prior
        self.margin = margin

        tmpr = 1.0
        print('tmpr of 2-way sigmoid: ', tmpr)
        self.slope_P_left = tmpr # 1.0
        self.slope_P_right = tmpr #1.0
        self.slope_N_left = tmpr #1.0
        self.slope_N_right = tmpr #1.0

        self.dist = None
        if dist == 'softplus':
            def softplus_loss(x1, x2, reduction='mean'):
                x_diff = x1 - x2
                return torch.mean(F.softplus(x_diff, tmpr) + F.softplus(-x_diff, tmpr))
            self.dist = softplus_loss
        else:
            raise NotImplementedError("The distance function: {} is not defined!".format(dist))

    def twoWaySigmoidLossForPositive(self, outputs):
        return torch.sigmoid(self.slope_P_left*outputs) * torch.sigmoid(-self.slope_P_right*(outputs-self.margin))

    def twoWaySigmoidLossForNegative(self, outputs):
        return torch.sigmoid(self.slope_N_left*(outputs+self.margin)) * torch.sigmoid(-self.slope_N_right*outputs)

    def forward(self, outputs, labels):
        labels = labels.view(-1,1)
        outputs = outputs.view_as(labels)

        outputs_P = outputs[labels==1].view(-1,1)
        outputs_U = outputs[labels!=1].view(-1,1)

        if outputs_P.numel() > 0:
            C_P_plus = self.twoWaySigmoidLossForPositive(outputs_P).mean()
            C_P_minus = self.twoWaySigmoidLossForNegative(outputs_P).mean()
        else:
            C_P_plus = torch.tensor(0., device=labels.device)
            C_P_minus = torch.tensor(0., device=labels.device)
        if outputs_U.numel() > 0:
            C_U_minus = self.twoWaySigmoidLossForNegative(outputs_U).mean()
        else:
            C_U_minus = torch.tensor(0., device=labels.device)

        return self.prior*C_P_plus + self.dist(C_U_minus, self.prior*C_P_minus, reduction='mean')

class TwoWaySigmoidLossWithEMA(TwoWaySigmoidLoss):
    def __init__(self, prior, margin=1.0, dist='softplus', tmpr=1, alpha_=0.9):
        TwoWaySigmoidLoss.__init__(self, prior, margin, dist, tmpr)

        self.alpha_ = alpha_
        self.one_minus_alpha_ = 1-alpha_

        self.C_P_minus_ema = None
        self.C_U_minus_ema = None
    
    def forward(self, outputs, labels):
        labels = labels.view(-1,1)
        outputs = outputs.view_as(labels)

        outputs_P = outputs[labels==1].view(-1,1)
        outputs_U = outputs[labels!=1].view(-1,1)

        if outputs_P.numel() > 0:
            C_P_plus = self.twoWaySigmoidLossForPositive(outputs_P).mean()
            C_P_minus = self.twoWaySigmoidLossForNegative(outputs_P).mean()
            self.C_P_minus_ema = C_P_minus.detach()
        else:
            C_P_plus = torch.tensor(0., device=labels.device)
            C_P_minus = torch.tensor(0., device=labels.device)
        if outputs_U.numel() > 0:
            C_U_minus = self.twoWaySigmoidLossForNegative(outputs_U).mean()
            self.C_U_minus_ema = C_U_minus.detach()
        else:
            C_U_minus = torch.tensor(0., device=labels.device)

        if self.C_P_minus_ema != None and self.C_U_minus_ema != None:
            self.forward = self.second_forward

        return self.prior*C_P_plus + self.dist(C_U_minus, self.prior*C_P_minus, reduction='mean')
    
    def second_forward(self, outputs, labels):
        labels = labels.view(-1,1)
        outputs = outputs.view_as(labels)

        outputs_P = outputs[labels==1].view(-1,1)
        outputs_U = outputs[labels!=1].view(-1,1)

        if outputs_P.numel() > 0:
            C_P_plus = self.twoWaySigmoidLossForPositive(outputs_P).mean()
            C_P_minus = self.alpha_*self.C_P_minus_ema + self.one_minus_alpha_*self.twoWaySigmoidLossForNegative(outputs_P).mean()
            self.C_P_minus_ema = C_P_minus.detach()
        else:
            C_P_plus = torch.tensor(0., device=labels.device)
            C_P_minus = self.C_P_minus_ema
        if outputs_U.numel() > 0:
            C_U_minus = self.alpha_*self.C_U_minus_ema + self.one_minus_alpha_*self.twoWaySigmoidLossForNegative(outputs_U).mean()
            self.C_U_minus_ema = C_U_minus.detach()
        else:
            C_U_minus = self.C_U_minus_ema

        return self.prior*C_P_plus + self.dist(C_U_minus, self.prior*C_P_minus, reduction='mean')/self.one_minus_alpha_
