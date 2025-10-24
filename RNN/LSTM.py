import os
import glob
import string
import unicodedata
import random
import time
import math
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ----- 设备设置 -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- 数据处理 -----
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

def readLines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicodeToAscii(line.strip()) for line in f]

def loadData(path):
    category_lines = {}
    all_categories = []
    for filename in findFiles(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        category_lines[category] = readLines(filename)
    return category_lines, all_categories

def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def randomTrainingExample(category_lines, all_categories):
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# ----- LSTM 模型定义 -----
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.i2f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2i = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2c = nn.Linear(input_size + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        forget_gate = torch.sigmoid(self.i2f(combined))
        input_gate = torch.sigmoid(self.i2i(combined))
        output_gate = torch.sigmoid(self.i2o(combined))
        cell_candidate = torch.tanh(self.i2c(combined))

        cell = forget_gate * cell + input_gate * cell_candidate
        hidden = output_gate * torch.tanh(cell)

        output = self.out(hidden)
        output = self.softmax(output)
        return output, hidden, cell

    def initHidden(self): return torch.zeros(1, self.hidden_size).to(device)
    def initCell(self): return torch.zeros(1, self.hidden_size).to(device)

# ----- 训练函数 -----
def train_step_lstm(category_tensor, line_tensor, lstm, criterion, optimizer):
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    hidden = lstm.initHidden()
    cell = lstm.initCell()

    lstm.train()
    optimizer.zero_grad()

    for i in range(line_tensor.size(0)):
        output, hidden, cell = lstm(line_tensor[i], hidden, cell)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

# ----- 推理与评估 -----
def evaluate_lstm(line_tensor, lstm):
    line_tensor = line_tensor.to(device)
    hidden = lstm.initHidden()
    cell = lstm.initCell()
    with torch.no_grad():
        for i in range(line_tensor.size(0)):
            output, hidden, cell = lstm(line_tensor[i], hidden, cell)
    return output

def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def evaluate_accuracy(lstm, category_lines, all_categories, n_samples=1000):
    correct = 0
    with torch.no_grad():
        for _ in range(n_samples):
            category, line, _, line_tensor = randomTrainingExample(category_lines, all_categories)
            output = evaluate_lstm(line_tensor, lstm)
            guess, _ = categoryFromOutput(output, all_categories)
            if guess == category:
                correct += 1
    return correct / n_samples

# ----- 绘图 -----
def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations (x1000)')
    plt.ylabel('Loss')
    plt.savefig("loss_curve_new.png")

def plot_accuracy(accs, interval):
    plt.figure()
    x = [i * interval for i in range(1, len(accs)+1)]
    plt.plot(x, accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig("accuracy_curve_new.png")

def plot_confusion(confusion, all_categories):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix_new.png")

# ----- 主流程 -----
if __name__ == '__main__':
    category_lines, all_categories = loadData('./data/names/*.txt')
    n_categories = len(all_categories)

    lstm = LSTM(n_letters, 128, n_categories).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

    n_iters = 300000
    print_every = 5000
    plot_every = 1000
    accuracy_every = 1000

    current_loss = 0
    all_losses = []
    accuracies = []
    best_acc = 0.0

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        output, loss = train_step_lstm(category_tensor, line_tensor, lstm, criterion, optimizer)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            correct = '✓' if guess == category else f'✗ ({category})'
            elapsed = time.time() - start
            print(f'{iter} {iter / n_iters * 100:.2f}% ({elapsed // 60:.0f}m) {loss:.4f} {line} → {guess} {correct}')

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

        if iter % accuracy_every == 0:
            acc = evaluate_accuracy(lstm, category_lines, all_categories)
            accuracies.append(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save(lstm.state_dict(), "best_lstm_model_new.pth")

    # 保存日志
    with open("lstm_training_log_new.json", "w") as f:
        json.dump({"loss": all_losses, "accuracy": accuracies}, f)

    plot_loss(all_losses)
    plot_accuracy(accuracies, accuracy_every)

    # 混淆矩阵
    confusion = torch.zeros(n_categories, n_categories)
    for _ in range(10000):
        category, line, _, line_tensor = randomTrainingExample(category_lines, all_categories)
        output = evaluate_lstm(line_tensor, lstm)
        guess, guess_i = categoryFromOutput(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    plot_confusion(confusion, all_categories)

    print("\n✅ 训练完成，模型保存在 best_lstm_model.pth，日志记录为 lstm_training_log.json")