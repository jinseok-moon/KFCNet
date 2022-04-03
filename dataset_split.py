import os
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

dirlst = []
rootdir = './basedata'
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        for temp in os.listdir(d):
            e = os.path.join(d, temp)
            if os.path.isdir(e):
                ss = e.replace(rootdir,"")
                dirlst.append(ss)


def batch_move_files(file_list, source_path, destination_path):
    for file in file_list:
        image = file.split('/')[-1]  # 파일 이름(+확장자)을 가져옴
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        shutil.copy(os.path.join(source_path, image), destination_path)  # 파일 복사
    return


for directory in dirlst:
    source_dir = "./basedata" + directory
    image_files = glob(source_dir+"/*")

    images = [name for name in image_files]


    # splitting the dataset
    # train:val:test = 8:1:1
    train_names, test_names = train_test_split(images, test_size=0.2, random_state=42, shuffle=True)
    val_names, test_names = train_test_split(test_names, test_size=0.5, random_state=42, shuffle=True)

    # new data path
    test_dir = "./dataset/test" + directory
    train_dir = "./dataset/train" + directory
    val_dir = "./dataset/val" + directory
    batch_move_files(train_names, source_dir, train_dir)
    batch_move_files(test_names, source_dir, test_dir)
    batch_move_files(val_names, source_dir, val_dir)