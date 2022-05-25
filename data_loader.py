import torch
from torchvision import datasets, transforms
import numpy as np

class Dataset(object):
    def __init__(self, directory, batch_size):
        """Load and Set Datasets

        Args:
            directory (str) : directory of dataset

        """

        self.transform = transforms.Compose([transforms.Resize((255,255)),
                                     transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

        self.data_set = {}
        self.data_loader = {}
        if directory[-1] != "/":
            directory += "/"

        for phase in ['train', 'test', 'val']:
            self.data_set[phase] = datasets.ImageFolder(directory+phase, transform=self.transform)  # Dataset
            self.data_loader[phase] = torch.utils.data.DataLoader(self.data_set[phase], batch_size=batch_size, shuffle=True)

        self.num_classes = len(self.data_set["train"].classes)

    def check_label(self, preds):
        return np.array(self.data_set["train"].classes)[preds]
