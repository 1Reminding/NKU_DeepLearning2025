# # # ----------- 每类准确率柱状图 -----------
# # import torch
# # import torchvision
# # import torchvision.transforms as transforms
# # import matplotlib.pyplot as plt
# # from cnn import Net

# # # 数据
# # transform = transforms.Compose([
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# # ])
# # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# # testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# # # 模型
# # net = Net()
# # net.load_state_dict(torch.load("./cifar_net.pth"))
# # net.eval()

# # # 每类准确率
# # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# # correct_pred = {classname: 0 for classname in classes}
# # total_pred = {classname: 0 for classname in classes}

# # with torch.no_grad():
# #     for data in testloader:
# #         images, labels = data
# #         outputs = net(images)
# #         _, predictions = torch.max(outputs, 1)
# #         for label, prediction in zip(labels, predictions):
# #             if label == prediction:
# #                 correct_pred[classes[label]] += 1
# #             total_pred[classes[label]] += 1

# # class_acc = {cls: 100 * correct_pred[cls] / total_pred[cls] for cls in classes}

# # # 绘图
# # plt.figure(figsize=(12, 4))
# # plt.bar(class_acc.keys(), class_acc.values(), color='skyblue')
# # plt.title("Accuracy per Class after Training")
# # plt.xlabel("Class")
# # plt.ylabel("Accuracy (%)")
# # plt.ylim(0, 100)
# # plt.grid(axis='y')
# # plt.tight_layout()
# # plt.savefig("plot_class_accuracy_bar.png")
# # plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# import json

# # 加载Res2Net的日志数据
# with open(r'C:\Xing\大三下\深度学习\CNN\logs\res2net_log.json', 'r') as f:
#     res2net_data = json.load(f)
# # 绘制Loss曲线
# plt.figure(figsize=(10, 4))
# plt.plot(range(1, len(res2net_data['training_loss'])+1), res2net_data['training_loss'], 'b-o', label='Training Loss')
# plt.plot(range(1, len(res2net_data['val_loss'])+1), res2net_data['val_loss'], 'r-s', label='Validation Loss')
# plt.title("Loss Curve (Res2Net)", fontsize=14)
# plt.xlabel("Epoch", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.grid(True)
# plt.legend(fontsize=10)
# plt.tight_layout()
# plt.savefig(r"C:\Xing\大三下\深度学习\CNN\plots\res2net_loss_curve.png", dpi=300)
# plt.show()

# # 绘制Accuracy曲线
# plt.figure(figsize=(10, 4))
# plt.plot(range(1, len(res2net_data['training_acc'])+1), res2net_data['training_acc'], 'b-o', label='Training Accuracy')
# plt.plot(range(1, len(res2net_data['val_acc'])+1), res2net_data['val_acc'], 'r-s', label='Validation Accuracy')
# plt.title("Accuracy Curve (Res2Net)", fontsize=14)
# plt.xlabel("Epoch", fontsize=12)
# plt.ylabel("Accuracy (%)", fontsize=12)
# plt.grid(True)
# plt.legend(fontsize=10)
# plt.tight_layout()
# plt.savefig(r"C:\Xing\大三下\深度学习\CNN\plots\res2net_accuracy_curve.png", dpi=300)
# plt.show()

# # 绘制每类准确率柱状图
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# class_acc = res2net_data['per_class_acc']

# plt.figure(figsize=(12, 4))
# plt.bar(classes, list(class_acc.values()), color='skyblue')
# plt.title("Accuracy per Class (Res2Net)", fontsize=14)
# plt.xlabel("Class", fontsize=12)
# plt.ylabel("Accuracy (%)", fontsize=12)
# plt.ylim(0, 100)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.savefig(r"C:\Xing\大三下\深度学习\CNN\plots\res2net_class_accuracy_bar.png", dpi=300)
# plt.show()
