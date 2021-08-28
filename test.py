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

parser = argparse.ArgumentParser(description="PSMNet")
parser.add_argument("--maxdisp", type=int, default=192, help="maxium disparity")
parser.add_argument(
    "--start", type=int, default=0, help="number of epochs started to test"
)
parser.add_argument("--datapath", default="./", help="datapath")
parser.add_argument("--end", type=int, default=0, help="number of epochs ended to test")
parser.add_argument("--savedpath", default="./", help="save model")


args = parser.parse_args()

(
    all_left_img,
    all_right_img,
    all_left_disp,
    test_left_img,
    test_right_img,
    test_left_disp,
) = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=4,
    shuffle=True,
    num_workers=4,
    drop_last=False,
)
# batch_size=12, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=4,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)
# batch_size=8, shuffle=False, num_workers=4, drop_last=False)


model = stackhourglass(args.maxdisp)

model = nn.DataParallel(model)
model.cuda()


def test(imgL, imgR, disp_true):

    model.eval()

    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    # ---------
    mask = disp_true < args.maxdisp
    # ----

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        top_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3] // 16
        right_pad = (times + 1) * 16 - imgL.shape[3]
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
        checkpoint = torch.load(args.savedpath + "checkpoint_" + str(epoch) + ".tar")
        model.load_state_dict(checkpoint["state_dict"])

        print("complete to load model")
        total_test_loss = 0
        for _, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(imgL, imgR, disp_L)
            # print("Iter %d test loss = %.3f" % (batch_idx, test_loss))
            total_test_loss += test_loss

        print("total test loss = %.3f" % (total_test_loss / len(TestImgLoader)))
    # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
