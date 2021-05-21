#!/usr/bin/env python3
import torch
import torch.nn as nn
from ResNet import Resnet, conv1x1
import numpy as np
import cv2
from carafe import CARAFEPack
n_class = 2

class BarcodeNet(nn.Module):

    def __init__(self, n_class=2, pretrained=True):
        super(BarcodeNet, self).__init__()
        self.n_class = n_class
        resnet = Resnet(model='resnet50', pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.Sequential(
            conv1x1(128, 64),
            CARAFEPack(channels=64, scale_factor=2)
        )
        # self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score

def predict_mask(img, net):
    
    means = np.array([103.939, 116.779, 123.68]) / 255.
    img = img[:, 160:1120]
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
    img = img / 255.
    img[:, :, 0] -= means[0]
    img[:, :, 1] -= means[1]
    img[:, :, 2] -= means[2]

    x = torch.from_numpy(img).float().permute(2, 0, 1)
    x = x.unsqueeze(0)
    if(torch.cuda.is_available()):
        x = x.cuda()
    output = net(x)
    output = output.data.cpu().numpy()
    _, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(1, h, w)
    pred = pred[0]
    pred = np.int8(pred)

    mask = np.zeros((720, 1280))
    predict = cv2.resize(pred, (960, 720), interpolation=cv2.INTER_NEAREST)
    mask[:, 160:1120] = predict
    mask[mask != 0] = 255
    mask = mask.astype(np.uint8)

    return mask


