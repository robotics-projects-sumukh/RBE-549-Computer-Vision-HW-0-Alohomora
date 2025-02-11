#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import Adam
from torchvision.datasets import CIFAR10
# import cv2
import sys
# import os
import numpy as np
import random
# import skimage
# import PIL
# import glob
import random
# from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
# import time
from torchvision.transforms import ToTensor
import argparse
# import shutil
# import string
# from termcolor import colored, cprint
# import math as m
from tqdm.notebook import tqdm
# import Misc.ImageUtils as iu
from Network.Network import CIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *
import torchvision.transforms as transforms
from Network.Network import *

# Don't generate pyc codes
sys.dont_write_bytecode = True

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

transform_train_custom = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),  # Flip images horizontally
    torchvision.transforms.RandomCrop(32, padding=4),  # Crop with padding
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

transform_test_custom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

transform_train_densenet = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
            ])

transform_test_densenet = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])

transform_train_resnet = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_resnet = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_resnext = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_resnext = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet)-1)
        
        ImageNum += 1

        I1, Label = TrainSet[RandIdx]
    	
        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################

        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))
        
    return torch.stack(I1Batch), torch.stack(LabelBatch)

def GenerateTestBatch(ValidationSet, TestLabels, ImageSize, BatchNumber):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 100*(BatchNumber-1)
    ImagesInBatch = 100

    ImageNumBeforeLoop = ImageNum
        
    while ImageNum < ImagesInBatch+ImageNumBeforeLoop:
        I1, Label = ValidationSet[ImageNum]
        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))
        ImageNum += 1
        
    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def TrainOperation(TrainLabels, TestLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, ValidationSet, LogsPath, NumClasses, NumTestRunsPerEpoch, ModelType):
    """
    Inputs: 
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Initialize the model
    # model = CIFAR10Model(BasicBlock, [2, 2, 2, 2]).to(device)
    # ###############################################
    # # Fill your optimizer of choice here!
    # ###############################################
    # Optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=200)   

    if ModelType == 'Custom':
        model = CIFAR10Model(InputSize=3*32*32,OutputSize=10).to(device)
        Optimizer = Adam(model.parameters(), lr=0.001)
        Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=10, gamma=0.1)
    elif ModelType == 'DenseNet':
        model = DenseNet_CIFAR10Model(100, 10, 12, 0.5, True, 0.0).to(device)
        Optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)
    elif ModelType == 'ResNet':
        model = ResNet_CIFAR10Model(ResNet_BasicBlock, [2, 2, 2, 2]).to(device)
        Optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=200)
    elif ModelType == 'ResNext':
        model = ResNext_CIFAR10Model(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64).to(device)
        Optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=200)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
        
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        if ModelType == 'DenseNet':
            Optimizer.param_groups[0]['lr']  = (0.1 ** (Epochs // 150)) * (0.1 ** (Epochs // 225))

        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)

            I1Batch, LabelBatch = Batch
            I1Batch, LabelBatch = I1Batch.to(device), LabelBatch.to(device)
            # Predict output with forward pass
            LossThisBatch = model.training_step((I1Batch, LabelBatch))

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            
            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                
                torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
                print('\n' + SaveName + ' Model Saved...')

            result = model.validation_step((I1Batch, LabelBatch))
            model.epoch_end(Epochs*NumIterationsPerEpoch + PerEpochCounter, result)
            # Tensorboard
            Writer.add_scalar('LossEveryIter', result["loss"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            Writer.add_scalar('Accuracy', result["acc"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        if ModelType == 'Custom' or ModelType == 'ResNet' or ModelType == 'ResNext':
            Scheduler.step()

        val_acc = 0
        val_loss = 0

        for i in range(NumTestRunsPerEpoch):
            I1Batch, LabelBatch = GenerateTestBatch(ValidationSet, TestLabels, ImageSize, i+1)
            I1Batch, LabelBatch = I1Batch.to(device), LabelBatch.to(device)
            result = model.validation_step((I1Batch, LabelBatch))
            val_acc += result["acc"]
            val_loss += result["loss"]
        val_acc /= NumTestRunsPerEpoch
        val_loss /= NumTestRunsPerEpoch

        Writer.add_scalar('ValidationLossEveryIter', val_loss, Epochs*NumIterationsPerEpoch + PerEpochCounter)
        Writer.add_scalar('ValidationAccuracy', val_acc, Epochs*NumIterationsPerEpoch + PerEpochCounter)
        Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
        print('\n' + SaveName + ' Model Saved...')
        

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=512, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    Parser.add_argument('--ModelType', default='Custom', help='Type of model to train, Default:Custom')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    if ModelType == 'Custom':
        transform_train = transform_train_custom
        transform_test = transform_test_custom
    elif ModelType == 'DenseNet':   
        transform_train = transform_train_densenet
        transform_test = transform_test_densenet
    elif ModelType == 'ResNet':
        transform_train = transform_train_resnet
        transform_test = transform_test_resnet
    elif ModelType == 'ResNext':
        transform_train = transform_train_resnext
        transform_test = transform_test_resnext
    else:
        print("Invalid Model Type")
        return
    
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
    ValidationSet = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)

    BasePath = './data'
    
    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses, TestLabels, NumTestRunsPerEpoch = SetupAll(BasePath, CheckPointPath)


    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(TrainLabels, TestLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, ValidationSet, LogsPath , NumClasses, NumTestRunsPerEpoch, ModelType)

    
if __name__ == '__main__':
    main()
 
