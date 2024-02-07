import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,ConcatDataset
from torchvision.datasets import ImageFolder
from PIL import Image


stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = tt.Compose([
    #tt.RandomHorizontalFlip(),
    #tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor(),
    #tt.Normalize(*stats)
])

test_transform = tt.Compose([
    tt.ToTensor(),
    #tt.Normalize(*stats)
])


# PyTorch datasets
data_dir = r"/home/fberanek/Desktop/datasets/classification/cifar100"
train_data = ImageFolder(data_dir+'/train', train_transform)
test_data = ImageFolder(data_dir+'/test', test_transform)

#train_data = CIFAR100(download=True,root="./data",transform=train_transform)
#test_data = CIFAR100(root="./data",train=False,transform=test_transform)

for image,label in train_data:
    print("Image shape: ",image.shape)
    print("Image tensor: ", image)
    print("Label: ", label)
    break

print(f"labels {train_data.classes}")

train_classes_items = dict()

for train_item in train_data:
    label = train_data.classes[train_item[1]]
    if label not in train_classes_items:
        train_classes_items[label] = 1
    else:
        train_classes_items[label] += 1

print(f"train_classes_items {train_classes_items}")


test_classes_items = dict()
for test_item in test_data:
    label = test_data.classes[test_item[1]]
    if label not in test_classes_items:
        test_classes_items[label] = 1
    else:
        test_classes_items[label] += 1

print(f"test_classes_items {test_classes_items}")

"""
Prepare network
"""
def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
    
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    
    return patches.view(b,-1,patches.shape[-2], patches.shape[-1])

def patchify(batch, patch_size):
    """
    Patchify the batch of images
        
    Shape:
        batch: (b, h, w, c)
        output: (b, nh, nw, ph, pw, c)
    """
    b, c, h, w = batch.shape
    ph, pw = patch_size
    nh, nw = h // ph, w // pw

    batch_patches = torch.reshape(batch, (b, c, nh, ph, nw, pw))
    batch_patches = torch.permute(batch_patches, (0, 1, 2, 4, 3, 5))

    return batch_patches

def display_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def display_patched_img(img, batch_patches):
    patches = batch_patches[0]
    c, nh, nw, ph, pw = patches.shape

    plt.figure(figsize=(5, 5))
    plt.imshow(img.movedim(1,-1)[0])
    plt.axis("off")

    plt.figure(figsize=(5, 5))
    for i in range(nh):
        for j in range(nw):
            plt.subplot(nh, nw, i * nw + j + 1)
            plt.imshow(patches[:, i, j].movedim(0,-1))
            plt.axis("off")

class Net(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        # size is size of kernel and stride/step is step on vertical as well as horizontal
        self.patch_size = patch_size
        # get number of patches
        self.num_of_patches = (image_size[0]/self.patch_size)*(image_size[1]/self.patch_size)
        self.fc_pre_attention = nn.Linear(192, 512)



        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Move dimension
        x_img = ((x.clone().detach()*255).to(torch.uint8))
        # Create patches
        x_patchified = patchify(x_img, (8,8))
        # Display patch images
        display_patched_img(x_img, x_patchified)
        # Switch to (batch, width, height, dim)
        x = x.movedim(1,-1)
        # Just to see image 
        x_img = Image.fromarray(x_img.numpy())
        display_img(x_img)
        # unfold(dimension, size, step)

        # Create patches accross width and height
        x_hat = x.unfold(1, 
                         self.patch_size, 
                         self.patch_size).unfold(2, 
                                                   self.patch_size, 
                                                   self.patch_size)    
        # Flat created patches so 4,4 patches become 16 flat dimension
        x_hat = x_hat.flatten(1,2)
        # Move RGB channel back at the end
        x_hat = x_hat.movedim(2,-1)
        # Flat patches
        x_hat = x_hat.flatten(2,-1)
        for patch_id in range(x_hat.shape[1]):
            x_hat[0][patch_id] = self.fc_pre_attention(x_hat[0][patch_id])
        print("")



        for img_id in range(x_hat.shape[1]):
            x_one_image = ((x_hat[0][img_id].clone().detach()*255).to(torch.uint8)).squeeze()
            x_one_image = Image.fromarray(x_one_image.numpy())
            display_img(x_one_image)



        x_img_1 = x_hat[0][0]

        y = x.unfold(1, size, stride).unfold(2, size, stride)
        y = y.contiguous().view(y.size(0), -1, 100, 100)
        y = y.permute(1, 0, 2, 3).contiguous()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net((32,32),8)

"""
Get loss function and optimizer
"""
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

"""
Get dataloader
"""
BATCH_SIZE=1
train_dl = DataLoader(train_data,BATCH_SIZE,num_workers=4,pin_memory=True,shuffle=True)
test_dl = DataLoader(test_data,BATCH_SIZE,num_workers=4,pin_memory=True)

"""
Train network
"""
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dl, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
print("")