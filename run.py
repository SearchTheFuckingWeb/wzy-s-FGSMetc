#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/21 

import json
from argparse import ArgumentParser
from data import *
from attackers import fgsm as my_fgsm
from attackers import pgd  as my_pgd
from attackers import uap as my_uap
from universal_pert import universal_perturbation
import torchattacks
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import grad as mygrad
import tensorflow as tf
from tensorflow.python.platform import gfile

import matplotlib.pyplot as plt

BASE_PATH = ''

def run_one(args):
  #print(args.way, args.method, args.steps, args.eps, args.alpha)
  #print("one")
  dataloader=get_dataloader()
  X, Y = dataloader.dataset.__getitem__(args.i)
  model = resnet18(pretrained=True).eval()
  X = torch.tensor(X)
  Y = torch.tensor([Y])
  X = X.unsqueeze(dim=0)
  pred_X = model(X).argmax(dim=-1)
  #print(pred_X)
  if(args.way == 1):
    if (args.method == 'FGSM'):
      attack = my_fgsm
    else:
      attack = my_pgd
    AX = attack(model, X, Y, args.eps, args.alpha, args.steps)
  else:
    if (args.method == 'FGSM'):
      attack = torchattacks.FGSM(model, args.eps)
    elif (args.method == 'MIFGSM'):
      attack = torchattacks.MIFGSM(model, args.eps, args.alpha, args.steps)
    else:
      attack = torchattacks.PGD(model, args.eps, args.alpha, args.steps)
    AX = attack(X, Y)
  AX_im = transforms.ToPILImage()(AX[0])
  AX_im.show()
  pred_AX = model(AX).argmax(dim=-1)
  DX = AX - X
  DX_im = transforms.ToPILImage()(DX[0])
  DX_im.show()
  Linf = (AX - X).abs().max()
  L1 = (AX - X).abs().sum()
  L2 = ((AX.flatten() - X.flatten())**2).sum().sqrt()
  print("Y       = ", int(Y))
  print("pred_X  = ", int(pred_X))
  print("pred_AX = ", int(pred_AX))
  print("Linf    = ", float(Linf))
  print("L1      = ", float(L1))
  print("L2      = ", float(L2))

  #print(Linf, L1, L2)
  # plt.savefig([X, AX, minmax_norm(DX)])
  #
  # json.save({
  #   cmd: ' '.join(sys.args),
  #   args: vars(args),
  #   dt: ...   // run start date-time (str)
  #   ts: ...   // run time in seconds (float)
  #   pred: {
  #     truth: Y
  #     raw: pred_X,
  #     adv: pred_AX,
  #   },
  #   metric: {
  #     Linf: ...
  #     L1: ...
  #     L2: ...
  #   },
  # })
  pass

def grad(model, AX:Tensor, Y:Tensor):
  AX.requires_grad = True
  logits = model(normalize(AX))
  loss = F.cross_entropy(logits, Y, reduction='none')   # [B]
  g = mygrad(loss, AX, loss)[0]
  return g

def jacobian(y_flat, x, inds):
    n = 10 # Not really necessary, just a quick fix.
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < n,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars)
    return jacobian.stack()

def run_all(args):
  #print("all")
  dataloader=get_dataloader()
  model = resnet18(pretrained=True).eval()
  len = dataloader.dataset.__len__()
  #len = 10
  print(args.way, args.batch_size)
  if(args.way == 1):
    if (args.method == 'FGSM'):
      attack = my_fgsm
    elif (args.method == 'UAP'):
      print("niubi")
      attack = my_uap
    else:
      print("shabi")
      attack = my_pgd
  else:
    print("6")
    if (args.method == 'FGSM'):
      attack = torchattacks.FGSM(model, args.eps)
    elif (args.method == 'MIFGSM'):
      attack = torchattacks.MIFGSM(model, args.eps, args.alpha, args.steps)
    else:
      attack = torchattacks.PGD(model, args.eps, args.alpha, args.steps)
  if(attack != my_uap):
    print("gaoji")
    acc_cnt = 0
    racc_cnt = 0
    psr_cnt = 0
    sasr_cnt = 0
    for i in range(0, len):
      X, Y = dataloader.dataset.__getitem__(i)
      X = torch.tensor(X)
      Y = torch.tensor([Y])
      X = X.unsqueeze(dim=0)
      pred_X = model(normalize(X)).argmax(dim=-1)
      if(args.way == 1):
        AX = attack(model, X, Y, args.eps, args.alpha, args.steps)
      else:
        AX = attack(X, Y)
      pred_AX = model(AX).argmax(dim=-1)
      #print(pred_X, pred_AX, Y)
      if(pred_X == Y): 
        acc_cnt += 1
        if(pred_AX != Y): sasr_cnt += 1
      if(pred_X == pred_AX): psr_cnt += 1
      if(pred_AX == Y): racc_cnt += 1
    print("acc  = ", float(acc_cnt / len))
    print("racc = ", float(racc_cnt / len))
    print("asr  = ", float((len - racc_cnt) / len))
    print("psr  = ", float(psr_cnt / len))
    print("sasr = ", float(sasr_cnt / len))
  else:
    print("gongxi")
    acc_cnt = 0
    racc_cnt = 0
    psr_cnt = 0
    sasr_cnt = 0
    dataset = np.empty([len, 224, 224, 3], dtype=np.uint8)
    for i in range(0, len):
      X, Y = dataloader.dataset.__getitem__(i)
      X = np.asarray(X, dtype=np.uint8).transpose(1, 2, 0)   # [C, H, W]
      dataset[i] = X
    print("official")
    def grads(AX:Tensor, Y:Tensor): return grad(model, AX, Y)
    v = universal_perturbation(dataset, model, grads)
    DX = transforms.ToTensor()(v[0]) * 255
    print(DX)
    DX_im = transforms.ToPILImage()(DX[0].to(torch.float))
    DX_im.show()
    for i in range(0, len):
      X, Y = dataloader.dataset.__getitem__(i)
      X = torch.tensor(X)
      Y = torch.tensor([Y])
      X = X.unsqueeze(dim=0)
      pred_X = model(X).argmax(dim=-1)
      AX = X + 4 * DX
      pred_AX = model(AX.to(torch.float)).argmax(dim=-1)
      if(pred_X == Y): 
        acc_cnt += 1
        if(pred_AX != Y): sasr_cnt += 1
      if(pred_X == pred_AX): psr_cnt += 1
      if(pred_AX == Y): racc_cnt += 1
    print("acc  = ", float(acc_cnt / len))
    print("racc = ", float(racc_cnt / len))
    print("asr  = ", float((len - racc_cnt) / len))
    print("psr  = ", float(psr_cnt / len))
    print("sasr = ", float(sasr_cnt / len))

  # acc, racc, asr, psr, sasr = ...
  #
  # for X, Y in dataloader:
  #   AX = attack(X, Y)
  #   ...
  #
  # json.save({
  #   cmd: ' '.join(sys.args),
  #   args: vars(args),
  #   dt: ...   // run start date-time (str)
  #   ts: ...   // run time in seconds (float)
  #   metric: {
  #     acc: ...
  #     racc: ...
  #     asr: ...
  #     psr: ...
  #     sasr: ...
  #   },
  # })
  pass



if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-I', '--i', type=int, help='image index of the dataset')
  parser.add_argument('-M', '--method', default='PGD', choices=['FGSM', 'PGD', 'MIFGSM', 'UAP'])
  parser.add_argument('-B', '--batch_size', default=16, type=int)
  parser.add_argument('-F', '--way', default=1, type=int)
  parser.add_argument('--steps', default=10,    type=int)
  parser.add_argument('--alpha', default=1/255, type=eval)
  parser.add_argument('--eps',   default=8/255, type=eval)
  args = parser.parse_args()

  if args.i:
    run_one(args)
  else:
    run_all(args)