from data_loader import Dataset
from network import KFCNet

trainSet = Dataset('./dataset/train')
testSet = Dataset('./dataset/test')
valSet = Dataset('./dataset/val')
num_class = trainSet.dataset.classes
kfc_net = KFCNet(num_class)


