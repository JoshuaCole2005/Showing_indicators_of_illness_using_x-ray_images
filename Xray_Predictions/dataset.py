import torch
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy
from PIL import Image
from matplotlib import pyplot as plt
import os

transformers = {
    "train": transforms.Compose([transforms.Resize((224,224)), transforms.RandomRotation(20), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize((224,224)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ]),
    "validation": transforms.Compose([transforms.Resize((224,224)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
}

PHASES = ["train", "test", "validation"]
IMG_PATH = "C:/Users/joshu/Python Projects/Machine Learning Projects/HackAppletonProject/Xray_Predictions/data/"
PATH_TYPE = ["train", "test", "validation"]


dataset = {x : torchvision.datasets.ImageFolder(IMG_PATH+x, transform=transformers[y]) for x,y in zip(PATH_TYPE, PHASES)}
dataset_sizes = {x : len(dataset[x]) for x in ["train","test"]}
loaders =  {x : torch.utils.data.DataLoader(dataset[x], batch_size=256, shuffle=True, num_workers=4)for x in PATH_TYPE}
