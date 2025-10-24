import time
import math
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datetime import datetime

from model import EncoderRNN, DecoderRNN
from dataset import get_dataloader, prepareData

EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建图片保存目录
def create_save_dirs():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"rnn_results_{timestamp}"
    dirs = {
        "base": base_dir,
        "plots": os.path.join(base_dir, "plots"),
        "heatmaps": os.path.join(base_dir, "heatmaps"),
        "models": os.path.join(base_dir, "models"),
        "data": os.path.join(base_dir, "data")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    return dirs

# 时间工具函数
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# 损失绘图函数
def showPlot(points, title="Training Loss", ylabel="Loss", save_path="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"图表已保存到 {save_path}")
    
    # 保存原始数据
    data_path = save_path.replace('.png', '.npy')
    np.save(data_path, np.array(points))
    print(f"图表数据已保存到 {data_path}")

# 单个 epoch 的训练逻辑
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion, epoch=None, total_epochs=None):
    encoder.train()
    decoder.train()

    total_loss = 0
    progress = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100,
                    desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for step, (input_tensor, target_tensor) in progress:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)),
                         target_tensor.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"\u2713 Epoch {epoch} 完成 | 平均损失: {avg_loss:.4f}")
    return avg_loss

# 计算准确率
def calculate_accuracy(decoder_outputs, target_tensor):
    # 获取预测结果
    _, topi = decoder_outputs.topk(1)
    predicted = topi.squeeze().detach()
    
    # 计算非填充位置的准确率
    correct = (predicted == target_tensor).float()
    # 忽略 EOS_token 之后的部分
    mask = (target_tensor != 0).float()  # 0 是 PAD_token
    correct = (correct * mask).sum().item()
    total = mask.sum().item()
    
    return correct / total if total > 0 else 0

# 完整训练过程
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
          print_every=1, plot_every=1, save_every=10, save_dirs=None):
    if save_dirs is None:
        save_dirs = create_save_dirs()
    
    start = time.time()
    plot_losses = []
    plot_accuracies = []
    print_loss_total = 0
    plot_loss_total = 0
    print_acc_total = 0
    plot_acc_total = 0
    best_loss = float('inf')
    
    # 保存训练配置
    config = {
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "device": str(device),
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(os.path.join(save_dirs["base"], "training_config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=5)
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=5)

    print(f"训练 {n_epochs} 个 epochs | 学习率={learning_rate} | 设备={device}")
    
    # 创建训练日志文件
    log_file = os.path.join(save_dirs["data"], "training_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,loss,accuracy,learning_rate\n")

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder,
                           encoder_optimizer, decoder_optimizer, criterion,
                           epoch=epoch, total_epochs=n_epochs)

        # 计算准确率
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            total_acc = 0
            total_samples = 0
            for input_tensor, target_tensor in train_dataloader:
                encoder_outputs, encoder_hidden = encoder(input_tensor)
                decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
                
                accuracy = calculate_accuracy(decoder_outputs, target_tensor)
                batch_size = input_tensor.size(0)
                total_acc += accuracy * batch_size
                total_samples += batch_size
            
            epoch_accuracy = total_acc / total_samples if total_samples > 0 else 0

        print_loss_total += loss
        plot_loss_total += loss
        print_acc_total += epoch_accuracy
        plot_acc_total += epoch_accuracy

        encoder_scheduler.step(loss)
        decoder_scheduler.step(loss)
        
        # 记录训练日志
        with open(log_file, "a") as f:
            current_lr = encoder_optimizer.param_groups[0]['lr']
            f.write(f"{epoch},{loss:.6f},{epoch_accuracy:.6f},{current_lr}\n")

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_acc_avg = plot_acc_total / plot_every
            plot_accuracies.append(plot_acc_avg)
            plot_loss_total = 0
            plot_acc_total = 0
            
            # 保存中间结果图
            if epoch % (plot_every * 5) == 0:
                loss_plot_path = os.path.join(save_dirs["plots"], f"loss_plot_epoch_{epoch}.png")
                showPlot(plot_losses, title="训练损失", ylabel="损失", save_path=loss_plot_path)
                
                acc_plot_path = os.path.join(save_dirs["plots"], f"accuracy_plot_epoch_{epoch}.png")
                showPlot(plot_accuracies, title="训练准确率", ylabel="准确率", save_path=acc_plot_path)

        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': loss,
                'accuracy': epoch_accuracy,
            }, os.path.join(save_dirs["models"], 'rnn_best_model.pth'))
            print(f"✓ 保存新的最佳模型 (损失: {best_loss:.4f}, 准确率: {epoch_accuracy:.4f})")

        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': loss,
                'accuracy': epoch_accuracy,
            }, os.path.join(save_dirs["models"], f'checkpoint_epoch_{epoch}.pth'))

    print(f"训练完成 | 最佳损失: {best_loss:.4f}")
    
    # 保存最终的损失和准确率图
    final_loss_path = os.path.join(save_dirs["plots"], "final_loss_plot.png")
    showPlot(plot_losses, title="训练损失", ylabel="损失", save_path=final_loss_path)
    
    final_acc_path = os.path.join(save_dirs["plots"], "final_accuracy_plot.png")
    showPlot(plot_accuracies, title="训练准确率", ylabel="准确率", save_path=final_acc_path)
    
    return plot_losses, plot_accuracies, save_dirs

# 模型加载
def load_model(encoder, decoder, model_path):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"模型已加载: {model_path} | Epoch {checkpoint['epoch']} | 损失: {checkpoint['loss']:.4f}")
        return checkpoint['epoch'], checkpoint['loss']
    else:
        print(f"未找到模型: {model_path}")
        return 0, float('inf')

# 评估相关
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# 修改 evaluate 函数以返回隐藏状态
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        hidden_states = []
        
        # 收集解码器隐藏状态
        for i, idx in enumerate(decoded_ids):
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
            # 保存隐藏状态
            if i < len(decoder_hidden):
                hidden_states.append(decoder_hidden[i].cpu().numpy())
    
    return decoded_words, hidden_states

# 添加隐藏状态热力图可视化函数
def showHiddenStateHeatmap(input_sentence, output_words, hidden_states, idx=0, save_dir="heatmaps"):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 将隐藏状态转换为可视化格式
    hidden_array = np.array(hidden_states)
    if len(hidden_array.shape) > 2:
        # 如果隐藏状态是多维的，取第一个维度
        hidden_array = hidden_array.reshape(hidden_array.shape[0], -1)
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    plt.imshow(hidden_array, aspect='auto', cmap='viridis')
    plt.colorbar()
    
    # 设置标签
    plt.xlabel('Hidden State Dimension')
    plt.ylabel('Decoding Step')
    plt.title(f"RNN Hidden States: '{input_sentence}' -> '{' '.join(output_words[:-1] if output_words[-1]=='<EOS>' else output_words)}'")
    
    # 保存图片
    timestamp = datetime.now().strftime("%H%M%S")
    filename = os.path.join(save_dir, f'hidden_state_heatmap_{idx}_{timestamp}.png')
    plt.savefig(filename)
    plt.close()
    # print(f"隐藏状态热力图已保存到 {filename}")
    
    # 保存隐藏状态数据
    data_filename = os.path.join(save_dir, f'hidden_state_data_{idx}_{timestamp}.npy')
    np.save(data_filename, hidden_array)
    # print(f"隐藏状态数据已保存到 {data_filename}")

def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n=5, save_dir=None):
    encoder.eval()
    decoder.eval()
    
    if save_dir is None:
        save_dir = "heatmaps"
        os.makedirs(save_dir, exist_ok=True)

    print("\n评估样本:")
    for i in range(n):
        pair = random.choice(pairs)
        print(f"> {pair[0]}\n= {pair[1]}")
        output_words, hidden_states = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        showHiddenStateHeatmap(pair[0], output_words, hidden_states, idx=i, save_dir=save_dir)
        print(f"< {' '.join(output_words)}")
        print("-" * 30)

# 从保存的数据重新绘制图表
def replot_from_saved_data(data_path, output_path=None, title="从保存数据重新绘制", ylabel="值"):
    if not os.path.exists(data_path):
        print(f"数据文件未找到: {data_path}")
        return
    
    # 加载数据
    data = np.load(data_path)
    
    # 设置输出路径
    if output_path is None:
        output_path = data_path.replace('.npy', '_replot.png')
    
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"重新绘制的图表已保存到 {output_path}")

# 主函数
def main():
    hidden_size = 64
    batch_size = 32
    n_epochs = 100
    learning_rate = 0.001

    print("加载数据...")
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
    _, _, pairs = prepareData('eng', 'fra', True)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    print(f"输入语言: {input_lang.name} | 输出语言: {output_lang.name}")
    print(f"训练批次数: {len(train_dataloader)}")
    
    # 创建保存目录
    save_dirs = create_save_dirs()

    # 尝试加载已有模型
    model_path = "/defaultShare/archive/zhaopenghai/seq2seq/rnn_results_20250624_145701/models/rnn_best_model.pth"
    if os.path.exists(model_path):
        load_model(encoder, decoder, model_path)
    
    # # 取消注释以开始训练
    # print("开始训练...")
    # _, _, save_dirs = train(train_dataloader, encoder, decoder, n_epochs, learning_rate, save_dirs=save_dirs)

    print("评估...")
    evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n=5, save_dir=save_dirs["heatmaps"])
    
    # 示例：从保存的数据重新绘制图表
    # replot_from_saved_data("rnn_results_20230101_120000/data/loss_plot.npy", 
    #                       "rnn_results_20230101_120000/plots/loss_replot.png", 
    #                       "重新绘制的损失曲线", "损失")

if __name__ == "__main__":
    import random
    main()
