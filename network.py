import PIL.Image
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
import time
import glob
import matplotlib.pyplot as plt

plt.rc('font', family='NanumGothic')


class KFCNet(nn.Module):
    def __init__(self, num_classes):
        super(KFCNet, self).__init__()

        self.model = models.resnet50(pretrained=True)  # For transfer learning

        in_feat = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feat, num_classes)  # Change output classes of fcn

    def forward(self, x):
        return self.model(x)


def train_model(model, dataset, criterion, optimizer, device, loadmd=True, num_epochs=25, model_save_step=5):
    if loadmd:
        start, model = load_model(model, "model")
    else:
        start = 1

    num_epochs += start
    for epoch in range(start, num_epochs):
        since = time.time()
        print(f'Epoch {epoch:03d}/{num_epochs:03d}')
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
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / len(dataset.data_set[phase])
            epoch_acc = running_corrects.double() / len(dataset.data_set[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            with open("model_loss.txt", "a") as f:
                f.write(f'Epoch {epoch:03d}, {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

            if epoch % model_save_step == 0:
                # 모델 객체의 state_dict 저장
                torch.save(model.state_dict(), './model/model_state_dict_{0:03d}.pt'.format(epoch))

        time_elapsed = time.time() - since
        print(f'Epoch {epoch:03d} Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model


def load_model(model, path):
    file_list = sorted(glob.glob(path +"/*"))
    if file_list:
        file_list_pt = [file for file in file_list if file.endswith(".pt")]
        model.load_state_dict(torch.load(file_list_pt[-1]))  # load 함수 내에 저장 디렉토리 작성
        start = int(file_list_pt[-1][file_list_pt[-1].index("dict_")+5:file_list_pt[-1].index(".pt")])+1  # start epoch
    else:
        start = 1
    return start, model


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


def test_case(model, img_path, dataset, device):
    model.eval()
    file_list = sorted(glob.glob(img_path + "/*"))
    tf = transforms.Compose([transforms.Resize((255, 255)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    with torch.no_grad():
        for n, img in enumerate(file_list):
            imgpil = PIL.Image.open(img)
            img_t = tf(imgpil)
            img_t = img_t.view([-1,3,255,255])
            img_t = img_t.to(device)
            outputs = model(img_t)
            _, preds = torch.max(outputs, 1)
            prob = torch.softmax(outputs, 1)
            probs_top5 = prob.data[0]
            probs_top5, indices = probs_top5.sort(descending=True)
            probs_np = probs_top5.cpu().numpy()[:5]*100
            indices_np = indices.cpu().numpy()[:5]
            labels = dataset.check_label(indices_np)
            fig, ax = plt.subplots(1)
            ax.axis('off')
            plt.imshow(imgpil)
            ax.set_title(f"이 음식은 {probs_np[0]:.2f}%의 확률로 " + labels[0] + "입니다.")
            plt.show()
            print(f"---{n}번째 사진 {img}---")
            for num in range(5):
                print(f"top {num+1}: {labels[num]} {probs_np[num]:.2f}%")