from .distributionLoss import *
from .entropyMinimization import *
from .twoWaySigmoidLoss import *

CLASS_PRIOR = {
    'cifar-10': 0.4,
}

def create_loss(args):
    prior = CLASS_PRIOR[args.dataset]
    print('prior: {}'.format(prior))
    
    if args.loss == 'labPU':
        if args.EMA == 1:
            base_loss = LabelDistributionLossWithEMA(prior=prior, 
            device=args.device, dist=args.dist, tmpr=args.tmpr, alpha_U=args.alpha_U)
        else:
            base_loss = LabelDistributionLoss(prior=prior, device=args.device, dist=args.dist, tmpr=args.tmpr)
    else:
        raise NotImplementedError("The loss: {} is not defined!".format(args.loss))
    
    if args.two_way == 1:
        if args.EMA == 1:
            two_loss = TwoWaySigmoidLossWithEMA(prior, args.margin, args.dist, args.tmpr, args.alpha_CN)
        else:
            two_loss = TwoWaySigmoidLoss(prior, args.margin, args.dist, args.tmpr)
        def loss_fn_2way(outputs, labels):
            return base_loss(outputs, labels) + two_loss(outputs, labels)
        return loss_fn_2way

    return base_loss