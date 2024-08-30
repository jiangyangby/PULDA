import torch
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score, average_precision_score

class Dict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__
  
def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst=Dict()
    for k,v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst

def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.deterministic = True

def get_checkpt_path(args):
    return './checkpt/{}/{}_{}.pth'.format(args.dataset, args.dist, args.random_seed)

def train(data_loader, model, device, loss_fn, optimizer, schedular):
    model.train()
    loss_total = 0
    for _, (index, Xs, Ys) in enumerate(data_loader):
        Xs = Xs.to(device)
        Ys = Ys.to(device)
        outputs = model(Xs).squeeze()
        loss = loss_fn(outputs, Ys.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
    
    schedular.step()
    return loss_total / len(data_loader)

def train_w_temperature(data_loader, model, device, loss_fn, optimizer, schedular, tmpr = 1.):
    model.train()
    loss_total = 0
    for _, (index, Xs, Ys) in enumerate(data_loader):
        Xs = Xs.to(device)
        Ys = Ys.to(device)
        outputs = model(Xs).squeeze() / tmpr
        loss = loss_fn(outputs, Ys.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
    
    schedular.step()
    return loss_total / len(data_loader)

def get_pred_scores_labels(data_loader, model, device):
    model.eval()
    predicted_scores = []
    labels = []
    with torch.no_grad():
        for _, (index, Xs, Ys) in enumerate(data_loader):
            Xs = Xs.to(device)
            Ys = Ys.to(device)
            outputs = model(Xs).view_as(Ys)
            outputs = torch.sigmoid(outputs)
            predicted_scores.append(outputs.cpu())
            labels.append(Ys.cpu())

    predicted_scores = torch.cat(predicted_scores).numpy()
    labels = torch.cat(labels).numpy()
    return predicted_scores, labels

def validate(data_loader, model, device, loss_fn):
    model.eval()
    predicted_scores = []
    labels = []
    indexes = []
    loss = 0
    with torch.no_grad():
        for _, (index, Xs, Ys) in enumerate(data_loader):
            Xs = Xs.to(device)
            Ys = Ys.to(device)
            outputs = model(Xs).view_as(Ys)

            loss = loss + loss_fn(outputs, Ys.float()).item()

            outputs = torch.sigmoid(outputs)
            predicted_scores.append(outputs.cpu())
            labels.append(Ys.cpu())
            indexes.append(index.squeeze().cpu())

    predicted_scores = torch.cat(predicted_scores).numpy()
    labels = torch.cat(labels).numpy()
    indexes = torch.cat(indexes).numpy()

    threshold = 0.5
    preds = np.zeros_like(predicted_scores, dtype=int)
    preds[predicted_scores>=threshold] = 1
    corrects = np.sum(preds==labels)
    TP = np.sum(preds[labels==1]==1)
    FP = np.sum(labels[preds==1]==0)
    FN = np.sum(labels[preds==0]==1)

    acc = float(corrects) / len(predicted_scores)
    precision = float(TP) / (TP+FP+1e-8)
    recall = float(TP) / (TP+FN+1e-8)
    f1 = 2 * precision * recall / (precision+recall+1e-8)

    auc = roc_auc_score(labels, predicted_scores)
    ap = average_precision_score(labels, predicted_scores)

    # class_prior = np.sum(predicted_scores>=0.5)/len(predicted_scores)
    # s_mean = np.mean(predicted_scores)

    return loss/len(data_loader), acc, precision, recall, f1, auc, ap

def mkdirs(path):
    dirs = os.path.split(path)[0]
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def save_chekpt(path, epoch, model, optimizer, schedular):
    mkdirs(path)
    all_states = {'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': schedular.state_dict()}
    torch.save(obj=all_states, f=path)