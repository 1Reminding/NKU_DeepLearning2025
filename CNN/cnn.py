import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import json

# 原始网络结构保持不变
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

if __name__ == '__main__':
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 训练
    training_loss, training_acc = [], []
    val_loss, val_acc = [], []
    num_epochs = 20
    best_val_acc = 0.0

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

        train_l = running_loss / len(trainloader)
        train_a = 100 * correct / total
        training_loss.append(train_l)
        training_acc.append(train_a)

        val_l, val_a = evaluate_model(net, testloader, criterion, device)
        val_loss.append(val_l)
        val_acc.append(val_a)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_a:.2f}% | Val Acc: {val_a:.2f}%")

        if val_a > best_val_acc:
            best_val_acc = val_a
            torch.save(net.state_dict(), "cnn_best_model.pth")
            print(f"✅ Best model updated at epoch {epoch+1}, Val Acc = {val_a:.2f}%")

    print("🎉 最优模型已保存为 cnn_best_model.pth")

    # 保存日志
    os.makedirs("logs", exist_ok=True)
    log_dict = {
        "training_loss": training_loss,
        "training_acc": training_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    }
    with open("logs/cnn_log.json", "w") as f:
        json.dump(log_dict, f)
    print("📊 日志已保存为 logs/cnn_log.json")

    # 绘图保存
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, num_epochs + 1), training_loss, 'b-o', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, 'r-s', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, num_epochs + 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/loss_curve.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, num_epochs + 1), training_acc, 'b-o', label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_acc, 'r-s', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(1, num_epochs + 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/accuracy_curve.png")
    plt.show()
    # 计算每类准确率
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    per_class_acc = [100 * c / t for c, t in zip(class_correct, class_total)]

    # 保存每类准确率数据（也可用于未来对比）
    log_dict["per_class_acc"] = {cls: round(acc, 2) for cls, acc in zip(classes, per_class_acc)}
    with open("logs/cnn_log.json", "w") as f:
        json.dump(log_dict, f)

    # 绘制每类准确率柱状图
    plt.figure(figsize=(10, 5))
    plt.bar(classes, per_class_acc, color='skyblue')
    plt.title('Per-Class Accuracy on CIFAR-10')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("plots/per_class_accuracy.png")
    plt.show()
