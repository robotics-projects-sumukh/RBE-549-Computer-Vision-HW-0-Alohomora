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

# import cv2
import os
import sys
# import glob
# import random
# from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
# from torchvision.transforms import ToTensor
import argparse
# import shutil
# import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch
from Network.Network import *
from Misc.MiscUtils import *
from Misc.DataUtils import *
import torchvision
import random
import seaborn as sns


# Don't generate pyc codes
sys.dont_write_bytecode = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing on: {device}")

transform_test_custom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

transform_test_densenet = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])

transform_test_resnet = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_resnext = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize

def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    return Img
    
def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """    
    I1 = Img
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred, mode='test'):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    if mode == 'train':
        print('Train Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')
    else:   
        print('Test Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')


def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, ModelType):
    """
    Inputs: 
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    if ModelType == 'Custom':
        model = CIFAR10Model(InputSize=3*32*32,OutputSize=10).to(device)
    elif ModelType == 'DenseNet':
        model = DenseNet_CIFAR10Model(100, 10, 12, 0.5, True, 0.0).to(device)
    elif ModelType == 'ResNet':
        model = ResNet_CIFAR10Model(ResNet_BasicBlock, [2, 2, 2, 2]).to(device)
    elif ModelType == 'ResNext':
        model = ResNext_CIFAR10Model(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64).to(device)
    
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.eval()
    print('Number of parameters in this model are %d ' % sum(p.numel() for p in model.parameters()))
    
    label_list = []
    pred_list = []
    OutSaveT = open(LabelsPathPred, 'w')

    for count in tqdm(range(len(TestSet))): 
        Img, Label = TestSet[count]
        label_list.append(Label)
        PredT = model(Img.to(device).unsqueeze(0))
        PredT = torch.argmax(PredT, dim =1).item()
        pred_list.append(PredT)
        OutSaveT.write(str(PredT)+'\n')
    OutSaveT.close()
    return label_list, pred_list

       
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='./Checkpoints/Custom.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--ModelType', default='Custom', help='Type of model to train, Default:Custom')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath
    ModelType = Args.ModelType

    if ModelType == 'Custom':
        transform_test = transform_test_custom
        ModelPath = './Checkpoints/Custom.ckpt'
    elif ModelType == 'DenseNet':   
        transform_test = transform_test_densenet
        ModelPath = './Checkpoints/DenseNet.ckpt'
    elif ModelType == 'ResNet':
        transform_test = transform_test_resnet
        ModelPath = './Checkpoints/ResNet.ckpt'
    elif ModelType == 'ResNext':
        transform_test = transform_test_resnext
        ModelPath = './Checkpoints/ResNext.ckpt'
    else:
        print("Invalid Model Type")
        return
    
    TestSet = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_test)

    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    test_label_list, test_pred_list = TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, ModelType)
    ConfusionMatrix(test_label_list, test_pred_list, mode='test')

    train_label_list, train_pred_list = TestOperation(ImageSize, ModelPath, TrainSet, LabelsPathPred, ModelType)
    ConfusionMatrix(train_label_list, train_pred_list, mode='train')
     
if __name__ == '__main__':
    main()
 
 
