import math

import torch
from torch import nn, optim
from torch.optim import Adam

from tqdm import tqdm
import time

from data import *
from param import *
from models.model.transformer import Transformer

def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weight(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

model = Transformer(PAD_ID, PAD_ID, th_vocab_size, zh_vocab_size, MAX_LENGTH, D_MODEL, N_HEADS, D_FF,
                    N_LAYERS, DROP_PROB, device).to(device)

print(f"the model has {param_count(model):,} trainable parameters")

initialize_weight(model)
optimizer = Adam(params=model.parameters(),
                 lr=INIT_LR,
                 weight_decay=WEIGHT_DECAY,
                 eps=ADAM_EPS)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=FACTOR,
                                                 patience=PATIENCE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def train(model, iterator, optimizer, criterion, clip, epoch):
    model.train()
    epoch_loss = 0
    i = 0
    pbar = tqdm(total=len(iterator), desc=f"Epoch {epoch+1}")
    for batch in iterator:
        src = batch['src']
        trg = batch['trg']

        optimizer.zero_grad()

        '''
        During training, the last token of the target sequence is removed before
        inputting it to the decoder. The model should generate an output sequence
        identical to the target sequence with the first token removed.
        '''
        output = model(src,trg[:,:-1]) 
        predicted = output.contiguous().view(-1, output.shape[-1])

        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(predicted, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        pbar.set_postfix(loss=loss.item())
        pbar.update()
        if (i % 6000 == 0):
            example("ฉัน อารมณ์ ดี เพราะ วันนี้ อากาศ ดี กว่า วัน อื่น และ มี เมฆ มาก")
        i+=1
    pbar.close()
    return epoch_loss / len(iterator)

'''
validation not finish

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_lost = 0
    batch_bleu = []

    with torch.no_grad():
        for batch in tqdm(iterator):
            src = batch['src']
            trg = batch['trg']

            output = model(src,trg[:,:-1])
            predicted = output.contiguous().view(-1, output.shape[-1])

            trg = trg[:, 1:].contiguous().view(-1, output.shape[-1])

            loss = criterion(predicted, trg)
            epoch_loss += loss.item()
'''

# still have scheduler, valid loss, evaluation, BLEU, test_loss
def run():
    best_lost = INF
    train_losses = []
    for step in range(EPOCH):
        print(f"Training epoch {step+1}")

        train_losses = train(model, dataloader, optimizer, criterion, CLIP, step)
        
    torch.save(model.state_dict(), 'saved/model-16epochs.pt')

def example(th_input):
    cur = [1]
    for token in th_tokenizer(th_input):
        cur.append(th_vocab[token])
    cur.append(2)

    cur2 = [1]
    enc_sample = torch.tensor([cur], dtype=torch.long).to(device)

    i = 0
    while True and i < 255:
        dec_sample = torch.tensor([cur2], dtype=torch.long).to(device)
        out_sample = model.forward(enc_sample, dec_sample, True)

        next = out_sample[0].argmax(-1)[-1]
        cur2 += [int(next)]
        if(next == 2):
            break
        i+=1
        print(i)

    print(f"{th_input} {tensor_to_string(cur2, zh_vocab)}")


if __name__ == '__main__':
    print('Script Running')
    try:
        run()
    except KeyboardInterrupt:
        example("ฉัน อารมณ์ ดี เพราะ วันนี้ อากาศ ดี กว่า วัน อื่น และ มี เมฆ มาก")
        print('saving model')
        torch.save(model.state_dict(), 'saved/model-interupted.pt')
        print('finished')


    