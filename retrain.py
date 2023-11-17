import math

import torch
from torch import nn, optim
from torch.optim import Adam

from tqdm import tqdm
import time

from nltk.translate.bleu_score import corpus_bleu


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

saved_path = "saved/model-14epochs.pt"
print(f"loading parameters from {saved_path}")
model.load_state_dict(torch.load(saved_path))

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
    pbar = tqdm(total=len(iterator), desc=f"Training Epoch {epoch+1}")
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
        if (i % 10000 == 0):
            exp = "<start> ฉัน อยาก ทาน อาหาร ที่ อร่อย <end>"
            exp_out = example(exp)
            print(f"{exp} {exp_out}")
            exp = "<start> สวัสดี , ฉัน รัก เธอ มาก <end>"
            exp_out = example(exp)
            print(f"{exp} {exp_out}")
        i+=1
    pbar.close()
    return epoch_loss / len(iterator)


def evaluate(model, iterator):
    model.eval()
    batch_bleu = []

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            src = batch['src']
            trg = batch['trg']

            trg_list, pred_list = batch_example(src,trg)
            batch_bleu.append(corpus_bleu(trg_list, pred_list))
    
    return sum(batch_bleu) / len(batch_bleu)


            
# still have scheduler, valid loss, evaluation, BLEU, test_loss
def run():
    for step in range(EPOCH):
        print(f"Training epoch {step+1}")

        train_losses = train(model, dataloader, optimizer, criterion, CLIP, step)
        #bleu_score = evaluate(model, valid_dataloader)

        scheduler.step(train_losses)

        print(f'Epoch {step+1} loss: {train_losses}')
    
def example(src_input):
    with torch.no_grad():
        src_lst = []
        for token in src_input.split():
            src_lst.append(th_vocab[token])
        enc_sample = torch.tensor([src_lst], dtype=torch.long).to(device)

        trg_lst = [START_ID]
        len = 0
        while True and len < 255:
            dec_sample = torch.tensor([trg_lst], dtype=torch.long).to(device)
            out_sample = model.forward(enc_sample, dec_sample, True)

            next = out_sample[0].argmax(-1)[-1]
            trg_lst.append(int(next))
            if(next == END_ID):
                break
            len += 1

        return tensor_to_string(trg_lst, zh_vocab)

def batch_example(batch_src_input,batch_trg_input):
    with torch.no_grad():
        trg_list = []
        pred_list = []
        for src_sentence, trg_sentence in zip(batch_src_input,batch_trg_input):
            try:
                str_trg = tensor_to_string(trg_sentence, zh_vocab)
                output = model(src_sentence.unsqueeze(0), trg_sentence.unsqueeze(0)[:,:-1], True)
                str_pred = "<start> "+tensor_to_string(output[0].argmax(-1),zh_vocab)
                trg_list.append([str_trg.split()])
                pred_list.append(str_pred.split())
            except:
                pass
        
    return trg_list, pred_list


if __name__ == '__main__':
    print('Script Running')
    try:
        #evaluate(model, valid_dataloader)
        exp = "<start> ฉัน อารมณ์ ดี เพราะ วันนี้ อากาศ ดี กว่า วัน อื่น และ มี เมฆ มาก <end>"
        exp_out = example(exp)
        print(f"{exp} {exp_out}")
        run()
        torch.save(model.state_dict(), 'saved/model-27epochs.pt')
    except KeyboardInterrupt:
        exp = "<start> ฉัน อารมณ์ ดี เพราะ วันนี้ อากาศ ดี กว่า วัน อื่น และ มี เมฆ มาก <end>"
        exp_out = example(exp)
        print(f"{exp} {exp_out}")
        print('saving model')
        torch.save(model.state_dict(), 'saved/model-interupted.pt')
        print('finished')


    