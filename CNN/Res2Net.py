import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import json

# ========== Res2Block ==========
class Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=4):
        super(Res2Block, self).__init__()
        assert out_channels % scale == 0
        self.scale = scale
        self.width = out_channels // scale
        self.stride = stride

        # 首先进行1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 如果stride>1，先进行下采样，确保所有分支的空间维度一致
        self.downsample_input = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride > 1 else nn.Identity()

        # 为每个分支创建3x3卷积
        self.convs = nn.ModuleList()
        for s in range(scale):
            if s == 0:
                self.convs.append(nn.Identity())
            else:
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, bias=False))

        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(scale)])
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        identity = x
        
        # 1x1卷积降维
        out = self.relu(self.bn1(self.conv1(x)))
        
        # 如果stride>1，先进行下采样
        out = self.downsample_input(out)
        
        # 将特征图分成scale份
        spx = torch.chunk(out, self.scale, 1)
        out_chunks = []
        
        for s in range(self.scale):
            if s == 0:
                out_chunks.append(spx[s])
            else:
                # 直接相加，因为空间维度已经一致
                if s == 1:
                    y = spx[s]
                else:
                    y = spx[s] + out_chunks[-1]
                y = self.relu(self.bns[s](self.convs[s](y)))
                out_chunks.append(y)
                
        # 拼接所有分支的输出
        out = torch.cat(out_chunks, dim=1)
        out = self.bn3(self.conv3(out))
        
        # 残差连接
        out += self.downsample(identity)
        return self.relu(out)

# ========== Res2Net ==========
class Res2Net(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Res2Net, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
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
        return self.fc(x)

# ========== Train & Eval ==========
def evaluate_model(net, dataloader, criterion, device):
    net.eval()
    correct, total, running_loss = 0, 0, 0.0
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

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Res2Net(Res2Block, [2,2,2,2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    best_val_acc = 0.0

    for epoch in range(20):
        net.train()
        total, correct, running_loss = 0, 0, 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        train_loss.append(running_loss / len(trainloader))
        train_acc.append(100 * correct / total)

        v_loss, v_acc = evaluate_model(net, testloader, criterion, device)
        val_loss.append(v_loss)
        val_acc.append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(net.state_dict(), "res2net_best_model.pth")
        print(f"Epoch {epoch+1} | Train Acc: {train_acc[-1]:.2f}% | Val Acc: {v_acc:.2f}%")

    # ========== 每类准确率 ==========
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct_pred = {cls: 0 for cls in classes}
    total_pred = {cls: 0 for cls in classes}
    net.eval()
    with torch.no_grad():
        for imgs, lbls in testloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = net(imgs)
            _, preds = torch.max(outs, 1)
            for label, pred in zip(lbls, preds):
                if label == pred:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    class_acc = {cls: 100 * correct_pred[cls] / total_pred[cls] for cls in classes}

    # ========== 保存日志 ==========
    os.makedirs("logs", exist_ok=True)
    with open("logs/res2net_log.json", "w") as f:
        json.dump({
            "training_loss": train_loss,
            "training_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "per_class_acc": class_acc
        }, f)

    # ========== 绘图 ==========
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Res2Net Loss")
    plt.legend()
    plt.savefig("plots/res2net_loss_curve.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Res2Net Accuracy")
    plt.legend()
    plt.savefig("plots/res2net_accuracy_curve.png")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.bar(class_acc.keys(), class_acc.values())
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Class - Res2Net")
    plt.savefig("plots/res2net_class_accuracy_bar.png")
    plt.show()
