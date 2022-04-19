from torch import nn
import torchvision.models as models


class KFCNet(nn.Module):
    def __init__(self, num_classes):
        super(KFCNet, self).__init__()

        self.model = models.resnet50(pretrained=True)  # For transfer learning

        in_feat = self.model.fc.in_features
        self.fc = nn.Linear(in_feat, num_classes)  # Change output classes of fcn

    def forward(self, x):
        return self.model(x)

