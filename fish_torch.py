from pandas.core.dtypes.missing import notnull
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

print(os.getcwd())
class Net(nn.Module):
    def __init__(self, featureLength):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(featureLength, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 7)
        # self.fc4 = nn.Linear(32, 7)
        #self.batch_norm1 = nn.BatchNorm1d(64)
        #self.batch_norm2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.7)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.fc1(x))
        #x = self.batch_norm1(x)
        x = self.relu(self.fc2(x))
        #x = self.batch_norm2(x)
        # x = self.relu(self.fc3(x))
        x = self.relu(self.fc3(x))
        # x = self.fc3(x)
        x = self.fc4(x)
        # x = self.dropout(x)
        
        # x = self.fc2(x)
        # x = self.sig(x)

        return x

class CustomDataset(Dataset):
    def __init__(self, data, label, transforms=None):
        self.x = [i for i in data]
        self.y = [i for i in label]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = np.array(x)

        return x, y

dataFrame = pd.read_csv("Fish.csv", delimiter=",");
# print(dataFrame.groupby("Species").std())

featureNames = dataFrame.columns # feature 이름
print(featureNames)

# 특정 feature 선택, featureNames[0]는 label, 필수
dataFrame = dataFrame[[featureNames[0], featureNames[2], featureNames[5], featureNames[6]]]
# feature 개수, linear input
featureLength = len(dataFrame.columns) - 1

x_data = []
y_data = []

name_list = list(set(dataFrame["Species"]))
name_list = np.sort(name_list)

for data in dataFrame.values:
    x_data.append(data[1:])

    for i, name in enumerate(name_list):
        if notnull(data[0]) and name == data[0]:
            y_data.append(i)

x_data = np.array(x_data, dtype="float64")
y_data = np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, shuffle=True, stratify=y_data, random_state=0) # stratify 비율을 맞춤

train_dataset = CustomDataset(x_train, y_train, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=7, shuffle=True, drop_last=True)
device = torch.device('cpu') # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net(featureLength).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr = 0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)

# lr = 0.01 gamma 0.01 batch 2 step 100 60%
epoch = 500
total_loss = 0

min_loss = 100.0
min_loss_epoch = 0
calc_loss = 0.0

enable_train = 1    # train enable

if enable_train == 1:
    model.train()
    for i in range(epoch):
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.long().to(device)

            outputs = model(x) # forward

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()  # 가중치와 편향에 대해 기울기 계산
            optimizer.step()

            total_loss += loss.item() # 텐서 중 값만 받음

            # outputs = outputs.detach().numpy()
            y = y.numpy()

        calc_loss = total_loss / len(x_train)

        if calc_loss < min_loss:
            min_loss = calc_loss
            min_loss_epoch = i

            # torch.save(model.state_dict(), f"weight/model_fish_min_loss{i}.pth")
            # model.eval()
            # model.load_state_dict(torch.load(f"weight/model_fish_min_loss{i}.pth"))
            # test_dataset = CustomDataset(x_test, y_test, transforms=None)
            # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            # total_right_cnt = 0
            # for x, y in test_loader:
            #     x = x.float().to(device)
            #     y = y.long().to(device)
            #     # print(x, y)
            #     outputs = model(x)

            #     # print(x, y, outputs)
            #     top = torch.topk(outputs, 1)
            #     # print(outputs, y)
            #     top_index = top.indices.numpy()

            #     for y, t in zip(y, top_index):
            #         # print(y, t)
            #         if y == t[0]:
            #             total_right_cnt += 1

            # print(f"score: ({total_right_cnt}/{len(x_test)}) | {total_right_cnt/len(x_test)}")
            torch.save(model.state_dict(), f"weight/model_fish_min_loss.pth")

        print(f"epoch -> {i}      loss -- > ", calc_loss)

        # if i != 0 and (i % 100 == 0 or i == epoch - 1):
        #     torch.save(model.state_dict(), f"weight/model_fish_min_loss{i}.pth")

        # if i == epoch - 1:
        #     torch.save(model.state_dict(), f"weight/model_fish_min_loss.pth")

        # if i != 0 and i % 100 == 0 or i == epoch - 1:
        #     print(f"epoch -> {i}      loss -- > ", calc_loss)
        # optimizer.step()
        total_loss = 0

    print("min_loss:", min_loss, " min_loss_epoch:", min_loss_epoch)

model.eval()
model.load_state_dict(torch.load('weight/model_fish_min_loss.pth'))
test_dataset = CustomDataset(x_test, y_test, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

total_right_cnt = 0
for x, y in test_loader:
    x = x.float().to(device)
    y = y.long().to(device)
    # print(x, y)
    outputs = model(x)

    # print(x, y, outputs)
    top = torch.topk(outputs, 1)
    # print(outputs, y)
    top_index = top.indices.numpy()

    for y, t in zip(y, top_index):
        # print(y, t)
        if y == t[0]:
            total_right_cnt += 1

print(f"score: ({total_right_cnt}/{len(x_test)}) | {total_right_cnt/len(x_test)}")
    
torch.save(model.state_dict(), f'weight/model_fish_min_loss{round(total_right_cnt/len(x_test), 3)}.pth')