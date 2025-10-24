import torch
import random
import os
from model import EncoderRNN, AttnDecoderRNN
from data import prepareData, tensorFromSentence, SOS_token, EOS_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence).to(device)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, attentions = decoder(encoder_outputs, encoder_hidden)

        # 获取每个时间步的最可能单词索引
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze(-1)  # 移除最后一个维度，保留 batch 和 time 维度

        decoded_words = []
        # 对于每个时间步（句子中的每个位置）
        for i in range(decoded_ids.size(1)):
            idx = decoded_ids[0, i].item()  # 获取第一个批次，第i个时间步的索引
            if idx == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word.get(idx, '?'))
    return decoded_words, attentions

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=5):
    encoder.eval()
    decoder.eval()
    print("\n===== 随机翻译示例 =====\n")
    for i in range(n):
        pair = random.choice(pairs)
        print(f'示例 {i+1}:')
        print('> 输入:', pair[0])
        print('= 正确翻译:', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        print('< 模型翻译:', ' '.join(output_words))
        print()

def load_models(model_path, input_size, hidden_size, output_size):
    # 创建模型实例
    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_size).to(device)
    
    # 加载预训练权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"模型已从 {model_path} 加载 | Epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.4f}")
    else:
        print(f"未找到模型文件: {model_path}")
    
    # 设置为评估模式
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder

def main():
    # 数据路径
    data_path = "data/eng-fra.txt"
    model_path = "models/best_model.pth"  # 使用合并的模型文件
    
    # 加载数据
    print("正在加载数据...")
    input_lang, output_lang, pairs = prepareData(data_path, reverse=True)
    
    # 模型参数
    hidden_size = 64
    
    # 加载模型
    print("正在加载模型...")
    encoder, decoder = load_models(
        model_path, 
        input_lang.n_words, 
        hidden_size, 
        output_lang.n_words
    )
    
    # 随机评估几个例子
    evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=5)
    
    # 交互式翻译
    print("\n===== 交互式翻译 =====\n")
    print("输入法语句子进行翻译 (输入'q'退出)：")
    
    while True:
        input_sentence = input("> ")
        if input_sentence.lower() == 'q':
            break
        
        output_words, _ = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
        translated_sentence = ' '.join(output_words)
        print("< ", translated_sentence)
        print()

if __name__ == "__main__":
    main()