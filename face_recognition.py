
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset,DataLoader,random_split
import torch.nn as nn


transform = transforms.Compose([transforms.ToTensor()])

def read_img(path):
    read_path = os.listdir(path)
    img_path = path+'/'
    n = 0
    face_label = [1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,5,6,7,8,9]
    face_list = []
    labels_list = []
    for i in read_path:
        if i == 'show.m':
            continue
        sx = os.listdir(img_path+i)
        for j in sx:
            face_path = img_path + i + '/' + j
            face = Image.open(face_path)
            face = face.resize((80,96),Image.NEAREST)
            face_tensor = transform(face)
            face_list.append(face_tensor)
            labels_list.append(torch.tensor(face_label[n]-1))
        n += 1
    return face_list,labels_list




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=0)
        self.pool3 = nn.MaxPool2d(2)
        self.act3 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=80*32,out_features=80)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=80,out_features=40)


    def forward(self,x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        out = x.view(-1,80*32)
        out = self.act4(self.fc1(out))
        out = self.fc2(out)
        return out



image_tensors,label_tensors = read_img('./ORL_Faces')

image_tensors = torch.stack(image_tensors)
label_tensors = torch.stack(label_tensors)

dataset = TensorDataset(image_tensors, label_tensors)
dataload = DataLoader(dataset,batch_size=16,shuffle=True)

train_size = 320
test_size = 80

train_set,test_set = random_split(dataset,[train_size,test_size])
train_load = DataLoader(train_set,batch_size=16,shuffle=True)
test_load = DataLoader(test_set,batch_size=16,shuffle=True)

model = Net()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 100

for epoch in range(num_epochs):
    loss_train = 0.0
    for X,y in train_load:
        
        y_hat = model(X)
        loss = loss_fn(y_hat,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    print('epochs: ',epoch,'   ',loss_train/len(train_load))

def accu(dataset,model):
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs,labels in dataset:
            outputs = model(imgs)
            _,labels_pre = torch.max(outputs,dim=1)
            total += labels.shape[0]
            correct += int((labels_pre == labels).sum())

    return correct/total

print('Train_Accuracy: ',accu(train_load,model))
print('Test_Accuracy: ',accu(test_load,model))

torch.save(model.state_dict(),'./face.pth')



