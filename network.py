import torch
from torch import nn
import torchvision.models as models
import time
import copy


class KFCNet(nn.Module):
    def __init__(self, num_classes):
        super(KFCNet, self).__init__()

        self.model = models.resnet50(pretrained=True)  # For transfer learning

        in_feat = self.model.fc.in_features
        self.fc = nn.Linear(in_feat, num_classes)  # Change output classes of fcn

    def forward(self, x):
        return self.model(x)


def train_model(model, dataset, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataset.data_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataset.data_set[phase])
            epoch_acc = running_corrects.double() / len(dataset.data_set[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), './model/models.pt')
    return model


# todo: Need to check model path is valid
def load_model(model, path):
    model.load_state_dict(torch.load(path))  # load 함수 내에 저장 디렉토리 작성


def test_model(model, dataset, device):
    since = time.time()
    accuracy = 0.0
    total = 0.0
    model.eval()   # 모델을 평가 모드로 설정
    with torch.no_grad():

        # 데이터 반복
        for inputs, labels in dataset.data_loader['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            accuracy += (preds == labels).sum().item()

    time_elapsed = time.time() - since
    print(f'Test complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    accuracy = (100 * accuracy / total)
    print(f'Accuracy: {accuracy:4f}')