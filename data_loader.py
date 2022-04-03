import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

import unicodedata

plt.rc('font', family='AppleGothic') # For MacOS Korean font
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (6.4, 6.4)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.grid'] = False

transform = transforms.Compose([transforms.Resize((255,255)),
                                 transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

train_dataset = datasets.ImageFolder('./dataset/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

dataiter = iter(train_loader)
fig, axes = plt.subplots(3,3)
images, labels = dataiter.next()

for n in range(9):
    img = images[n]
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    axes[n//3, n%3].imshow(np.transpose(npimg, (1, 2, 0)))
    axes[n//3, n%3].set_title(unicodedata.normalize('NFC', train_dataset.classes[labels[n]]))  # for macOS korean fonts
    # axes[n // 3, n % 3].set_title(train_dataset.classes[labels[n]])
    axes[n//3, n%3].axis('off')
plt.show()