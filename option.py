#!/usr/bin/env python3

import argparse
import torch
import os 

class Option():
    def __init__(self):
        parser = argparse.ArgumentParser(prog="DLP homework 4", description='This lab implement ResNet-18 and ResNet-50 to analysis diabetic retinopathy')

        # training hyper parameters
        parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the trainer, default is 1e-3")
        parser.add_argument("--epochs", type=int, default=10, help="The number of the training, default is 10")
        parser.add_argument("--batch_size", type=int, default=8, help="Input batch size for training, default is 8")

        # save name and load model path
        parser.add_argument("--save_folder", type=str, default=os.getcwd(), help="save model in save folder, default is current path")
        parser.add_argument("--load_model",  type=str, default=None, help="load model from wandb, ex. 'kuolunwang/BarcodeNet/model:v0', default is None")
        parser.add_argument("--test", action="store_true", default=False, help="True is test model, False is keep train, default is False")
        # cuda
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training, default is False')

        # network
        parser.add_argument("--pretrained", action='store_false', default=True, help="load pretrained weight to train, default is True")

        self.parser = parser

    def create(self):

        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args 