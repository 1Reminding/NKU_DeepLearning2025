import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# -------- DenseNet 组件 --------
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        return self.pool(x)

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()
        for i in range(len(nblocks)):
            self.features.add_module(f'denseblock{i+1}', self._make_dense_layers(block, num_channels, nblocks[i]))
            num_channels += nblocks[i] * growth_rate
            if i != len(nblocks) - 1:
                out_channels = int(num_channels * reduction)
                self.features.add_module(f'transition{i+1}', Transition(num_channels, out_channels))
                num_channels = out_channels

        self.bn = nn.BatchNorm2d(num_channels)
        self.linear = nn.Linear(num_channels, num_classes)

    def _make_dense_layers(self, block, in_channels, nblock):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.linear(x)

# -------- 验证函数 --------
def evaluate_model(net, dataloader, criterion, device):
    correct, total, running_loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), 100 * correct / total

# -------- 主流程 --------
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DenseNet(Bottleneck, [6, 12, 24, 16]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    training_loss, training_acc, val_loss, val_acc = [], [], [], []
    num_epochs = 20

    for epoch in range(num_epochs):
        net.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        training_loss.append(running_loss / len(trainloader))
        training_acc.append(100 * correct / total)
        val_l, val_a = evaluate_model(net, testloader, criterion, device)
        val_loss.append(val_l)
        val_acc.append(val_a)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {training_acc[-1]:.2f}% | Val Acc: {val_a:.2f}%")

    torch.save(net.state_dict(), "densenet_cifar10.pth")
    os.makedirs("plots", exist_ok=True)

    # -------- 曲线图 --------
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, num_epochs+1), training_loss, 'b-o', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_loss, 'r-s', label='Validation Loss')
    plt.title("Loss Curve (DenseNet)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/densenet_loss_curve.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, num_epochs+1), training_acc, 'b-o', label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_acc, 'r-s', label='Validation Accuracy')
    plt.title("Accuracy Curve (DenseNet)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/densenet_accuracy_curve.png")
    plt.show()

    # -------- 每类准确率柱状图 --------
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    class_acc = {cls: 100 * correct_pred[cls] / total_pred[cls] for cls in classes}
    plt.figure(figsize=(12, 4))
    plt.bar(class_acc.keys(), class_acc.values(), color='skyblue')
    plt.title("Accuracy per Class (DenseNet)")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("plots/densenet_class_accuracy_bar.png")
    plt.show()
