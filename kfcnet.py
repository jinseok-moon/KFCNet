from data_loader import Dataset
import network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    dataset = Dataset('./dataset/')
    print("Dataset Loaded")

    model = network.KFCNet(dataset.num_classes).to(device)

    # print(model)

    epochs = 50
    learning_rate = 0.0005

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # network.train_model(model, dataset, criterion, optimizer,device, False, num_epochs=epochs)
    model = network.load_model(model, "model")
    # network.test_model(model, dataset, device)
    network.test_one_case(model, "IMG_9319.JPG", dataset, device)
    network.test_one_case(model, "20210820_123511.JPG", dataset, device)
    network.test_one_case(model, "20210817_114609.JPG", dataset, device)



if __name__ == '__main__':
    main()
