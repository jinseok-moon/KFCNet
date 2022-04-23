from data_loader import Dataset
import network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

dataset = Dataset('./dataset/')
model = network.KFCNet(dataset.num_classes).to(device)

# 에포크, 배치 크기 지정
epochs = 5
batch_size = 64
learning_rate = 0.01

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

network.train_model(model, dataset, criterion, optimizer, exp_lr_scheduler, device, num_epochs=epochs)
