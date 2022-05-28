import matplotlib.pyplot as plt
import numpy as np
train_acc = []
train_loss = []
val_acc = []
val_loss = []
with open("model_loss.txt", "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        ls, ac = line.strip().split()[4], line.strip().split()[6]
        train_acc.append(float(ac))
        train_loss.append(float(ls))

        line = f.readline()
        if not line:
            break
        ls, ac = line.strip().split()[4], line.strip().split()[6]
        val_acc.append(float(ac))
        val_loss.append(float(ls))
        print(line.strip().split())

fig, loss_ax = plt.subplots(1, figsize=(6.4, 6.4), dpi=150)
acc_ax = loss_ax.twinx()

loss_ax.plot(train_loss, 'y', label='train loss')
loss_ax.plot(val_loss, 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(train_acc, 'b', label='train acc')
acc_ax.plot(val_acc, 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper right')
fig.savefig("graph.png")
