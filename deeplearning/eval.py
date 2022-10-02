import sys
import argparse
import os
import time
import torch
sys.path.append('BD_23Net.py')
from PIL import Image

from django.conf import settings
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import random
import cv2
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
from math import sqrt
import torch.nn.functional as F
#BDNet
class ARM(nn.Module):

    def __init__(self, input_h, input_w, channels):
        super().__init__()
        self.pool = nn.AvgPool2d( (input_h, input_w) )
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        feature_map = x

        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)
        x = torch.sigmoid(x)

        return x.expand_as(feature_map) * feature_map
class Segment(nn.Module):
    def __init__(self):
        super(Segment,self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 32, 3, stride=2),
            ConvBNReLU(32, 32, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(32, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )
        self.S4 = nn.Sequential(  # feat 64
            GELayerS2(128, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(  # feat 128
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

        self.head = nn.Sequential(
            ConvBNReLU(320,1024,3,stride=1),
            nn.Dropout(0.1),

            nn.Conv2d(1024, 256*4*4, 1, 1, 0),
            nn.PixelShuffle(4)
        )
        self.x8_arm = ARM(512 // 8, 512 // 8, 128)
        self.x16_arm = ARM(512 // 16, 512 // 16, 64)
        self.x32_arm = ARM(512 // 32, 512 // 32, 128)
        self.global_pool = nn.AvgPool2d((512 // 32, 512 // 32))
        self.up=UpSample(n_chan=256)
        self.conv2 = nn.Conv2d(320, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(256,256, 1, 1, 0)
        # self.up2 = nn.PixelShuffle(8)


    def forward(self, x):
        feat1=self.S1(x)
        feat2 = self.S2(feat1)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        # feature_x16 =feat4
        feat5 = self.S5_4(feat4)
        # feat5 = feature_x32
        # feat5_5 = self.S5_5(feature_x32)
        center = self.global_pool(feat5)
        feature_x8 = self.x8_arm(feat3)

        feature_x16 = self.x16_arm(feat4)
        feature_x32 = self.x32_arm(feat5)
        up_feature_x32 = F.upsample(center, size=(512 // 32, 512 // 32), mode='bilinear',
                                    align_corners=False)
        ensemble_feature_x32 = feature_x32 + up_feature_x32
        up_feature_x16 = F.upsample(ensemble_feature_x32, scale_factor=2, mode='bilinear', align_corners=False)
        ensemble_feature_x16 = torch.cat((feature_x16, up_feature_x16), dim=1)

        #print(ensemble_feature_x16.shape)
        up_feature_x8 = F.upsample(ensemble_feature_x16, scale_factor=2, mode='bilinear', align_corners=False)
        ensemble_feature_x8 = torch.cat((feature_x8, up_feature_x8), dim=1)





        feat = self.conv2(ensemble_feature_x8)
        feat=self.bn2(feat)
        feat = self.relu(feat)
        feat=self.conv1(feat)


        return feat,feat1,feat2,feat3,feat4,feat5





    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)
class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)  #3
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat
class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):

        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)

        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)

        return feat
class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat
class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class Detail(nn.Module):

    def __init__(self):
        super(Detail, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 32, 3, stride=2),
            ConvBNReLU(32, 32, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(32, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )
        self.up1 = nn.Sequential(
            UpSample(n_chan=64),
        )

        self.up2 = nn.Sequential(
            UpSample(n_chan=128, factor=2),
            UpSample(n_chan=128),
        )

        self.up3 = nn.Sequential(
            UpSample(n_chan=64),
        )
        self.conv1 = nn.Conv2d(288, 256, 1, 1, 0)

    def forward(self, x,x1,x2):


        x_up1 = self.up1(x1)
        x_1 = torch.cat([x, x_up1], 1)

        # print(x2.shape)
        x_up2 = self.up2(x2)
        # print(x_up2.shape)

        x_up3 = self.up3(x1)
        x_2 = torch.cat([x_1, x_up2, x_up3], 1)

        x_2 = self.conv1(x_2)

        return x_2

class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                256, 256, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                256, 256, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out
class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes * up_factor * up_factor
        if aux:
            self.conv_out = nn.Sequential(
                ConvBNReLU(mid_chan, up_factor * up_factor, 3, stride=1),
                nn.Conv2d(up_factor * up_factor, out_chan, 1, 1, 0),
                nn.PixelShuffle(up_factor)
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_chan, out_chan, 1, 1, 0),
                nn.PixelShuffle(up_factor)
            )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat

class BDNet(nn.Module):
    def __init__(self):
        super(BDNet,self).__init__()
        self.detail = Detail()
        self.seg=Segment()
        self.caafm=BGALayer()
        self.pred = nn.Conv2d(256,6,kernel_size=1,stride=1)
        self.aux2 = SegmentHead(64, 128, n_classes=6, up_factor=4)
        self.aux3 = SegmentHead(128, 128, n_classes=6, up_factor=8)
        self.aux4 = SegmentHead(64, 128, n_classes=6, up_factor=16)
        self.aux5_4 = SegmentHead(128, 128, n_classes=6, up_factor=32)

    def forward(self,x):

        feat_s,feat1,feat2,feat3,feat4,feat5=self.seg(x)

        feat_d = self.detail(feat1,feat2,feat3)
        # print(feat_d.shape)
        # print(feat_s.shape)

        feat = self.caafm(feat_d,feat_s)
        feat = self.pred(feat)
        feat = F.upsample(feat,x.size()[2:],mode='bilinear',align_corners=False)
        #logits_aux2 = self.aux2(feat2)
        #logits_aux3 = self.aux3(feat3)
        #logits_aux4 = self.aux4(feat4)
        #logits_aux5_4 = self.aux5_4(feat5)
        return feat#, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)
def get_label_info():
    return np.array([[255,255,255],[0,0,255],[0,255,255],
                [0,255,0],[255,255,0],[255,0,0]])
def train(args):

    model = BDNet()
    #model = model.cuda()
    # args.save_path=settings.IMG_UPLOAD    #保存的地址



    state_dict = torch.load(args.model_path,map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    test(args,model)
def crop_eval(args,model,dataloader):
    with torch.no_grad():
        model.eval()
        image = dataloader

        #image, label = image.cuda(), label.cuda().squeeze()
        h, w = image.size()[-2:]

        list1 = model(image)
        predict= F.softmax(list1[0], dim=1).squeeze(0)

        predict = predict.argmax(dim=0)
        save_image =get_label_info()[predict.cpu().numpy()].astype(np.uint8)
        save_image=save_image[:,:,::-1]
        cv2.imwrite(os.path.join(args.save_path, 'predict.png'), save_image)


def test(args,model):
    print('---------test begin-------------')
    data_transforms = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = args.dataset_path
    img = Image.open(image)
    image=data_transforms(img).unsqueeze(0)

    crop_eval(args,model,image)


def pred():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='stdc', help='Architecture')
    parser.add_argument('--dataset', nargs='?', type=str, default='vaihingen', help='Dataset to use ')
    parser.add_argument('--dataset_path', nargs='?', type=str, default='/Users/sir_zhang/Desktop/system/django_system/static/uploads/test.png', help='Dataset path ')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1000, help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=2.5e-2, help='Learning Rate')
    parser.add_argument('--num_classes', type=int, default=6, help='num of object classes (with void)')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')

    parser.add_argument('--load_model', type=bool, default=True, help='whether to load model ')
    parser.add_argument('--save_path', type=str, default='/Users/sir_zhang/Desktop/system/django_system/static/uploads/', help='whether to load model ')
    parser.add_argument('--model_path', type=str, default='/Users/sir_zhang/Desktop/system/django_system/best_model.pt',
                        help='pre-train model path')
    args = parser.parse_args(args=[])
    train(args)
    #CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #../../guang/dataset/Potsdam
