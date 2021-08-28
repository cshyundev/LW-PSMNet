from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *

from lightmodels.bounded_stackhourglass import *
from lightmodels.channel_compression import *
from lightmodels.disparity_compression import *
from lightmodels.disparity_expansion_v1 import *
from lightmodels.disparity_expansion_v2 import *
from lightmodels.tapered_compression import *
from lightmodels.uniform_compression import *
from lightmodels.weight_sharing import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--mindisp', type=int, default=0,
                    help='minimum disparity')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='./',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--masking', type=bool, default=True,
                    help='use masking with minimum/maximum disparity')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
    args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=4, shuffle=True, num_workers=4, drop_last=False)
# batch_size=12, shuffle=True, num_workers=8, drop_last=False)


start_epoch = 0

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'bounded_stackhourglass':
    model = bounded_stackhourglass(args.mindisp, args.maxdisp)
elif args.model == 'channel_compression':
    model = channel_compression(args.maxdisp)
elif args.model == 'disparity_compression':
    model = disparity_compression(args.maxdisp)
elif args.model == 'disparity_expansion_v1':
    model = disparity_expansion_v1(args.maxdisp)
elif args.model == 'disparity_expansion_v2':
    model = disparity_expansion_v2(args.maxdisp)
elif args.model == 'tapered_compression':
    model = tapered_compression(args.maxdisp)
elif args.model == 'uniform_compression':
    model = uniform_compression(args.maxdisp)
elif args.model == 'weight_sharing':
    model = weight_sharing(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])
    optimizer.load_state_dict(pretrain_dict["optimizer"])
    start_epoch = pretrain_dict['epoch'] + 1

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

def train(imgL, imgR, disp_L):
    model.train()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    if args.masking:
        # [min, max)로 mask 씌워서 학습
        mask = torch.logical_and(disp_true < args.maxdisp, disp_true >= args.mindisp)
    else:
        # [min, max) 바깥은 min 또는 max로 학습
        mask = disp_true > args.maxdisp
        disp_true[mask] = args.maxdisp
        mask = disp_true <= args.maxdisp
        mask2 = disp_true < args.mindisp
        disp_true[mask2] = args.mindisp
        
    mask.detach_()
    # ----
    optimizer.zero_grad()

    output1, output2, output3 = model(imgL, imgR)
    output1 = torch.squeeze(output1, 1)
    output2 = torch.squeeze(output2, 1)
    output3 = torch.squeeze(output3, 1)
    loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(
        output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    start_full_time = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('This is %d-th epoch' % (epoch))
        print(str(epoch) + '-th epoch start time:', time.strftime('%y-%m-%d %H:%M:%S'))
        epoch_start_time = time.time()
    
        total_train_loss = 0
        # adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            if batch_idx % 40 == 0:
                print(
                    "Iter %d training loss = %.3f , time = %.2f"
                    % (batch_idx, loss, time.time() - start_time)
                )
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %
              (epoch, total_train_loss/len(TrainImgLoader)))

        # SAVE
        savefilename = args.savemodel+'/checkpoint_%03d.tar' % epoch
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

        print(str(epoch) + '-th epoch finish time:', time.strftime('%y-%m-%d %H:%M:%S'))
        print(str(epoch) + '-th epoch training time:', ((time.time() - epoch_start_time)/3600))
        
    print('full training time = %.2f HR' %
          ((time.time() - start_full_time)/3600))


if __name__ == '__main__':
    main()
