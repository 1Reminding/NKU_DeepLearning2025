import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# -------------------------
# ResNet BasicBlock 定义
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# -------------------------
# ResNet 主结构
# -------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 适配 CIFAR10
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------------------------
# 验证函数
# -------------------------
def evaluate_model(net, dataloader, criterion, device):
    correct, total, running_loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), 100 * correct / total

# -------------------------
# 主流程
# -------------------------
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
    net = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)  # ResNet-18
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    training_loss, training_acc = [], []
    val_loss, val_acc = [], []
    num_epochs = 20

    for epoch in range(num_epochs):
        net.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        training_loss.append(running_loss / len(trainloader))
        training_acc.append(100 * correct / total)
        val_l, val_a = evaluate_model(net, testloader, criterion, device)
        val_loss.append(val_l)
        val_acc.append(val_a)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {training_acc[-1]:.2f}% | Val Acc: {val_a:.2f}%")

    # 保存模型
    torch.save(net.state_dict(), "resnet18_cifar10.pth")
    print("✅ 模型已保存为 resnet18_cifar10.pth")

    os.makedirs("plots", exist_ok=True)

    # 绘图 - Loss
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, num_epochs+1), training_loss, 'b-o', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_loss, 'r-s', label='Validation Loss')
    plt.title("Loss Curve (ResNet-18)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, num_epochs+1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/resnet_loss_curve.png")
    plt.show()

    # 绘图 - Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, num_epochs+1), training_acc, 'b-o', label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_acc, 'r-s', label='Validation Accuracy')
    plt.title("Accuracy Curve (ResNet-18)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(1, num_epochs+1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/resnet_accuracy_curve.png")
    plt.show()
    # ---------- 每类预测准确率柱状图 ----------
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                label_str = classes[label]
                if prediction == label:
                    correct_pred[label_str] += 1
                total_pred[label_str] += 1

    class_acc = {cls: 100 * correct_pred[cls] / total_pred[cls] for cls in classes}

    # 绘制柱状图
    plt.figure(figsize=(12, 4))
    plt.bar(class_acc.keys(), class_acc.values(), color='skyblue')
    plt.title("Accuracy per Class (ResNet-18 on CIFAR-10)")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("plots/resnet_class_accuracy_bar.png")
    plt.show()
