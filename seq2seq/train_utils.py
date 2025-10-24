import os
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def asMinutes(s):
    m = int(s // 60)
    s -= m * 60
    return f"{m}m {int(s)}s"

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return f"{asMinutes(s)} (- {asMinutes(rs)})"

def log_models(encoder, decoder, prefix="rnn", log_path=None):
    if log_path is None:
        log_path = f"{prefix}_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Encoder structure:\n")
        f.write(str(encoder) + "\n\n")
        f.write("Decoder structure:\n")
        f.write(str(decoder) + "\n")

def save_best_model(encoder, decoder, loss, best_loss, prefix="rnn"):
    if loss < best_loss:
        torch.save(encoder.state_dict(), f"{prefix}_best_encoder.pt")
        torch.save(decoder.state_dict(), f"{prefix}_best_decoder.pt")
        print(f"\n✅ New best model saved with loss {loss:.4f}")
        return loss
    return best_loss

def plot_losses(losses, title="Loss Curve", prefix="rnn", path=None):
    if path is None:
        path = f"{prefix}_loss_curve.png"
    plt.figure()
    plt.plot(losses)
    plt.title(f"{prefix.upper()} {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(path)
    plt.close()

def plot_accuracies(accs, title="Accuracy Curve", prefix="rnn", path=None):
    if path is None:
        path = f"{prefix}_acc_curve.png"
    plt.figure()
    plt.plot(accs)
    plt.title(f"{prefix.upper()} {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(path)
    plt.close()

def compute_accuracy_and_matrix(decoder_outputs, target_tensor):
    with torch.no_grad():
        pred_ids = decoder_outputs.argmax(dim=-1).view(-1).cpu().numpy()
        true_ids = target_tensor.view(-1).cpu().numpy()

        # 过滤掉填充值（通常是0）
        mask = true_ids != 0  # 假设0是填充值
        filtered_true_ids = true_ids[mask]
        filtered_pred_ids = pred_ids[mask]

        acc = accuracy_score(filtered_true_ids, filtered_pred_ids)
        
        # 使用固定的标签范围，确保所有批次生成相同大小的混淆矩阵
        # 获取输出词汇表大小（假设是decoder_outputs的最后一个维度）
        vocab_size = decoder_outputs.size(-1)
        all_labels = np.arange(vocab_size)
        cm = confusion_matrix(filtered_true_ids, filtered_pred_ids, labels=all_labels)

        return acc, cm

def plot_confusion_matrix(cm, title="Confusion Matrix", prefix="rnn", path=None):
    if path is None:
        path = f"{prefix}_conf_matrix.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap="GnBu")
    plt.title(f"{prefix.upper()} {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(path)
    plt.close()

def show_attention(input_sentence, output_words, attentions, prefix="attn", save_path=None):
    if save_path is None:
        save_path = f"{prefix}_attention_matrix.png"
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title(f"{prefix.upper()} Attention Weights")
    plt.savefig(save_path)
    plt.close()

def log_epoch_metrics(log_path, epoch, loss, acc):
    with open(log_path, "a") as f:
        f.write(f"{epoch},{loss:.4f},{acc:.4f}\n")