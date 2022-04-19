import torch
from torchvision import datasets, transforms
from torchvision.transforms import Compose


class Dataset(object):
    def __init__(self, directory):
        """Load and Set Datasets

        Args:
            directory (str) : directory of dataset

        """

        self.transform = transforms.Compose([transforms.Resize((255,255)),
                                     transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

        self.dataset = datasets.ImageFolder(directory, transform=self.transform)  # Dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)  # Data Loader
