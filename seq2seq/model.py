import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_token = 0
MAX_LENGTH = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))  # [B, T] -> [B, T, H]
        output, hidden = self.gru(embedded)             # output: [B, T, H], hidden: [1, B, H]
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward_step(self, input, hidden):
        embedded = self.embedding(input)  # [B, 1] -> [B, 1, H]
        output, hidden = self.gru(embedded, hidden)  # output: [B, 1, H]
        output = self.out(output)  # [B, 1, output_size]
        return output, hidden

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for t in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, t].unsqueeze(1)  # Teacher Forcing
            else:
                top1 = decoder_output.argmax(-1)
                decoder_input = top1.detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)  # [B, T, output_size]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: [B, 1, H], keys: [B, T, H]
        score = self.Va(torch.tanh(self.Wa(keys) + self.Ua(query)))  # [B, T, 1]
        attn_weights = F.softmax(score, dim=1)  # [B, T, 1]
        context = torch.sum(attn_weights * keys, dim=1, keepdim=True)  # [B, 1, H]
        return context, attn_weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))  # [B, 1, H]
        query = hidden.permute(1, 0, 2)  # [1, B, H] -> [B, 1, H]
        context, attn_weights = self.attention(query, encoder_outputs)  # [B, 1, H], [B, T, 1]
        rnn_input = torch.cat((embedded, context), dim=2)  # [B, 1, 2H]
        output, hidden = self.gru(rnn_input, hidden)  # [B, 1, H]
        output = self.out(output)  # [B, 1, output_size]
        return output, hidden, attn_weights

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for t in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, t].unsqueeze(1)  # Teacher Forcing
            else:
                top1 = decoder_output.argmax(-1)
                decoder_input = top1.detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions
