import os
import random
import torch
import numpy as np
from datetime import datetime

# 导入模型和数据处理相关模块
from model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from dataset import prepareData, tensorFromSentence, EOS_token, get_dataloader

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 评估函数 - RNN模型
def evaluate_rnn(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words

# 评估函数 - 带注意力的RNN模型
def evaluate_attn(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words

# 加载模型
def load_model(model_class, model_path, input_size, hidden_size, output_size):
    model = model_class(input_size, hidden_size).to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['encoder_state_dict'] if 'encoder_state_dict' in checkpoint else checkpoint['state_dict'])
        print(f"模型已加载: {model_path}")
        return model
    else:
        print(f"未找到模型: {model_path}")
        return None

# 主函数
def main():
    # 模型参数
    hidden_size = 64
    
    # 加载数据
    print("加载数据...")
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    
    # 模型路径
    rnn_model_path = "/defaultShare/archive/zhaopenghai/seq2seq/rnn_results_20250624_145701/models/rnn_best_model.pth"
    attn_model_path = "/defaultShare/archive/zhaopenghai/seq2seq/results_20250624_132820/models/best_model.pth"
    
    # 加载RNN模型
    rnn_encoder = load_model(EncoderRNN, rnn_model_path, input_lang.n_words, hidden_size, output_lang.n_words)
    rnn_decoder = load_model(DecoderRNN, rnn_model_path, input_lang.n_words, hidden_size, output_lang.n_words)
    
    # 加载带注意力的RNN模型
    attn_encoder = load_model(EncoderRNN, attn_model_path, input_lang.n_words, hidden_size, output_lang.n_words)
    attn_decoder = load_model(AttnDecoderRNN, attn_model_path, input_lang.n_words, hidden_size, output_lang.n_words)
    
    # 设置为评估模式
    rnn_encoder.eval()
    rnn_decoder.eval()
    attn_encoder.eval()
    attn_decoder.eval()
    
    # 随机选择5个样本进行翻译对比
    print("\n" + "=" * 50)
    print("模型翻译对比 (随机5个样本)")
    print("=" * 50)
    
    for i in range(5):
        pair = random.choice(pairs)
        print(f"\n样本 {i+1}:")
        print(f"原文: {pair[0]}")
        print(f"答案: {pair[1]}")
        
        # RNN模型翻译
        rnn_output = evaluate_rnn(rnn_encoder, rnn_decoder, pair[0], input_lang, output_lang)
        print(f"RNN翻译: {' '.join(rnn_output)}")
        
        # 带注意力的RNN模型翻译
        attn_output = evaluate_attn(attn_encoder, attn_decoder, pair[0], input_lang, output_lang)
        print(f"注意力RNN翻译: {' '.join(attn_output)}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()