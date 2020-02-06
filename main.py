import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as tud
import time

start = time.perf_counter()

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCH = 2
BATCH_SIZE = 32
LR = 0.01

train_set = torchvision.datasets.MNIST(root="./data", transform=torchvision.transforms.ToTensor(), train=True,
                                       download=True)
train_loader = tud.DataLoader(train_set, BATCH_SIZE, True, num_workers=4)
test_set = torchvision.datasets.MNIST(root="./data", transform=torchvision.transforms.ToTensor(), train=False,
                                      download=True)
test_loader = tud.DataLoader(test_set, 10, True, num_workers=4)

plt.figure()
plt.imshow(train_set.data[0].numpy(), cmap="gray")
print(train_set.data[0].numpy().shape)
plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv2d(1, 20, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = self.pool(x)
        x = F.relu(self.Conv2(x))
        x = self.pool(x)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = CNN().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCH):

    for i, data in enumerate(train_loader):
        inputs, label = data
        inputs = inputs.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("epoch:", epoch, "iter", i+1, "loss", torch.sum(loss).cpu().item())

end = time.perf_counter()
print('Finished Training')
print("time cost:", end-start)

with torch.no_grad():
    for i, data in enumerate(test_loader):
        test, label = data
        outputs = net(test)
        _, predicted = torch.max(outputs.data, 1)
        print("true:", label.numpy())
        print("exact:", predicted.numpy())
        for j in test:
            plt.imshow(j[0].numpy(), cmap="gray")
            plt.show()
        break

