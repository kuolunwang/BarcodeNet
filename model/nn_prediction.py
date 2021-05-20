#!/usr/bin/env python

import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from fcn_resnet import build_fcn_resnet

class NNPrediction:
    def __init__(self):

        self.network = build_fcn_resnet()

        self.labels = ['background', 'barcode']
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.network = self.network.cuda()

    def load_weight(self, weight):
        return self.network.load_state_dict(weight)

    def predict(self, img):
        means = np.array([103.939, 116.779, 123.68]) / 255.
        img = img[:, 160:1120]
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
        img = img / 255.
        img[:, :, 0] -= means[0]
        img[:, :, 1] -= means[1]
        img[:, :, 2] -= means[2]

        x = torch.from_numpy(img).float().permute(2, 0, 1)
        x = x.unsqueeze(0)
        if self.use_gpu:
            x = x.cuda()
        output = self.network(x)
        output = output.data.cpu().numpy()
        _, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, len(self.labels)).argmax(axis=1).reshape(1, h, w)
        pred = pred[0]
        pred = np.int8(pred)

        mask = np.zeros((720, 1280))
        predict = cv2.resize(pred, (960, 720), interpolation=cv2.INTER_NEAREST)
        mask[:, 160:1120] = predict
        mask[mask != 0] = 255
        mask = mask.astype(np.uint8)
        return mask


