import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import EncoderRNN, AttnDecoderRNN
from data import get_dataloader, tensorFromSentence, SOS_token, EOS_token
from train_utils import (
    asMinutes, timeSince, log_models, save_best_model,
    plot_losses, plot_accuracies,
    compute_accuracy_and_matrix, plot_confusion_matrix
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder.train()
    decoder.train()

    total_loss = 0
    total_acc = 0
    all_cm = None

    progress = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100, desc="Training", leave=False)

    for _, (input_tensor, target_tensor) in progress:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        encoder_optimizer.step()
        decoder_optimizer.step()

        acc, cm = compute_accuracy_and_matrix(decoder_outputs, target_tensor)
        total_loss += loss.item()
        total_acc += acc
        all_cm = cm if all_cm is None else all_cm + cm

        progress.set_postfix(loss=loss.item(), acc=acc)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc, all_cm


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=1, plot_every=1):
    start = time.time()
    loss_curve = []
    acc_curve = []
    print_loss_total = 0
    print_acc_total = 0
    best_loss = float("inf")

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=5)
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=5)

    log_models(encoder, decoder, prefix="attn")

    for epoch in range(1, n_epochs + 1):
        loss, acc, cm = train_epoch(train_dataloader, encoder, decoder,
                                    encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        print_acc_total += acc

        encoder_scheduler.step(loss)
        decoder_scheduler.step(loss)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_acc_avg = print_acc_total / print_every
            print_loss_total = 0
            print_acc_total = 0
            print(f"{timeSince(start, epoch / n_epochs)} (Epoch {epoch}) Loss: {print_loss_avg:.4f} | Acc: {print_acc_avg:.4f}")

            best_loss = save_best_model(encoder, decoder, print_loss_avg, best_loss, prefix="attn")

        if epoch % plot_every == 0:
            loss_curve.append(loss)
            acc_curve.append(acc)

    plot_losses(loss_curve, prefix="attn")
    plot_accuracies(acc_curve, prefix="attn")
    plot_confusion_matrix(cm, prefix="attn")


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence).to(device)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word.get(idx.item(), '?'))
    return decoded_words


def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=5):
    encoder.eval()
    decoder.eval()
    print("Encoder structure:\n", encoder)
    print("\nDecoder structure:\n", decoder)
    for _ in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        print('<', ' '.join(output_words))
        print()


if __name__ == '__main__':
    hidden_size = 64
    batch_size = 32
    num_epochs = 100

    input_lang, output_lang, train_dataloader = get_dataloader("data/eng-fra.txt", batch_size=batch_size, reverse=True)
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(train_dataloader, encoder, decoder, n_epochs=num_epochs, print_every=1, plot_every=1)

    _, _, pairs = prepareData("data/eng-fra.txt", reverse=True)
    evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang)
