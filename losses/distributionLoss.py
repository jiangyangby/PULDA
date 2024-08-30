import torch
import torch.nn.functional as F

class LabelDistributionLoss(torch.nn.Module):
    def __init__(self, prior, device, dist='softplus', tmpr=1):
        super(LabelDistributionLoss, self).__init__()
        self.prior = prior
        self.two_times_prior = 2*prior
        self.exp_y_P = torch.tensor(1, dtype=torch.float, requires_grad=False).to(device)
        self.exp_y_U = torch.tensor(prior, dtype=torch.float, requires_grad=False).to(device)

        self.dist = None
        if dist == 'softplus':
            def softplus_loss(x1, x2, reduction='mean'):
                x_diff = x1 - x2
                return torch.mean(F.softplus(x_diff, tmpr) + F.softplus(-x_diff, tmpr))
            self.dist = softplus_loss
        else:
            raise NotImplementedError("The distance function: {} is not defined!".format(dist))

        print('#### Label Distribution Alignment ####')
        print('Expectation of Labels from Labeled Positive data: ')
        print(self.exp_y_P)
        print('Expectation of Labels from Unlabeled data: ')
        print(self.exp_y_U)
        print('Distance Measure Function: ')
        print(dist)

    def forward(self, outputs, labels):
        scores=torch.sigmoid(outputs)
        labels=labels.view(-1,1)

        scores = scores.view_as(labels)
        scores_P = scores[labels==1].view(-1,1)
        scores_U = scores[labels==0].view(-1,1)

        R_L = 0
        R_U = 0
        if scores_P.numel() > 0:
            exp_y_hat_P = scores_P.mean()
            R_L = torch.mean(self.exp_y_P - exp_y_hat_P)
        if scores_U.numel() > 0:
            exp_y_hat_U = scores_U.mean()
            R_U = self.dist(exp_y_hat_U, self.exp_y_U, reduction='mean')

        return self.two_times_prior*R_L + R_U

class LabelDistributionLossWithEMA(LabelDistributionLoss):
    def __init__(self, prior, device, dist='softplus', tmpr=1, alpha_U=0.9):
        LabelDistributionLoss.__init__(self, prior, device, dist, tmpr)
        
        self.alpha_U = alpha_U
        self.one_minus_alpha_U = 1-alpha_U

        self.exp_y_hat_U_ema = None
    
    def forward(self, outputs, labels):
        scores=torch.sigmoid(outputs)
        labels=labels.view(-1,1)

        scores = scores.view_as(labels)
        scores_P = scores[labels==1].view(-1,1)
        scores_U = scores[labels==0].view(-1,1)

        R_L = 0
        R_U = 0
        if scores_P.numel() > 0:
            exp_y_hat_P = scores_P.mean()
            R_L = torch.mean(self.exp_y_P - exp_y_hat_P)
        if scores_U.numel() > 0:
            exp_y_hat_U = scores_U.mean()
            R_U = self.dist(exp_y_hat_U, self.exp_y_U, reduction='mean')
            self.exp_y_hat_U_ema = exp_y_hat_U.detach()

        if scores_U.numel() > 0: 
            self.forward = self.second_forward

        return self.two_times_prior*R_L + R_U

    def second_forward(self, outputs, labels):
        scores=torch.sigmoid(outputs)
        labels=labels.view(-1,1)

        scores = scores.view_as(labels)
        scores_P = scores[labels==1].view(-1,1)
        scores_U = scores[labels==0].view(-1,1)

        R_L = 0
        R_U = 0
        if scores_P.numel() > 0:
            exp_y_hat_P = scores_P.mean()
            R_L = torch.mean(self.exp_y_P - exp_y_hat_P)
        if scores_U.numel() > 0:
            exp_y_hat_U = self.alpha_U*self.exp_y_hat_U_ema + self.one_minus_alpha_U*scores_U.mean()
            R_U = self.dist(exp_y_hat_U, self.exp_y_U, reduction='mean')
            self.exp_y_hat_U_ema = exp_y_hat_U.detach()

        return self.two_times_prior*R_L + R_U/self.one_minus_alpha_U