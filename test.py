from __future__ import print_function
import argparse
import os
import os.path
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
parser.add_argument("--start", type=int, default=0, help="number of epochs started to test")
parser.add_argument("--end", type=int, default=0, help="number of epochs ended to test")
parser.add_argument('--datapath', default='./',
                    help='datapath')
parser.add_argument('--loadmodels', default=None,
                    help='load models')
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

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=4, shuffle=False, num_workers=4, drop_last=False)
# batch_size=8, shuffle=False, num_workers=4, drop_last=False)

if args.model == 'stackhourglass_org':
    model = stackhourglass_org(args.maxdisp)
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


model = nn.DataParallel(model)
model.cuda()

def test(imgL, imgR, disp_true):

    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    # ---------
    if args.masking:
        # [min, max)로 mask 씌워서 테스트
        mask = torch.logical_and(disp_true < args.maxdisp, disp_true >= args.mindisp)
    else:
        # [min, max) 바깥은 min 또는 max로 테스트
        mask = disp_true > args.maxdisp
        disp_true[mask] = args.maxdisp
        mask = disp_true <= args.maxdisp
        mask2 = disp_true < args.mindisp
        disp_true[mask2] = args.mindisp
    # ----

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16
        top_pad = (times+1)*16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3]//16
        right_pad = (times+1)*16-imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3)

    if top_pad != 0:
        img = output3[:, top_pad:, :]
    else:
        img = output3

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(img[mask] - disp_true[mask]))  # end-point-error
        # loss = F.l1_loss(img[mask], disp_true[mask])

    return loss.data.cpu()

def main():
    # ------------- TEST ------------------------------------------------------------
    for epoch in range(args.start, args.end):
        print(str(epoch) + ": test start")
        checkpoint = torch.load(args.loadmodels+"/checkpoint_%03d.tar" % epoch)
        model.load_state_dict(checkpoint["state_dict"])

        print("complete to load model")
        total_test_loss = 0
        for _, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(imgL, imgR, disp_L)
            # print("Iter %d test loss = %.3f" % (batch_idx, test_loss))
            total_test_loss += test_loss

        print("total test loss = %.3f" % (total_test_loss / len(TestImgLoader)))
    # ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
