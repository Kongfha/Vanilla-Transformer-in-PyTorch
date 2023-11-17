from param import *
from models.model.transformer import Transformer
import torch
import argparse

parser = argparse.ArgumentParser(description='use the model')

parser.add_argument('--input', metavar='TEXT', type=str, required=True, help='starter text')

path_to_th_vocab = "th_vocab.pth"
path_to_zh_vocab = "zh_vocab.pth"
print(f"Importing vocabulary set")
print(f"source vocabulary path: {path_to_th_vocab}")
    # Importing Thai Vocab
th_vocab = torch.load(path_to_th_vocab)
th_vocab_size = len(th_vocab.vocab.get_itos())
th_vocab.set_default_index(th_vocab_size)
th_vocab.append_token("<unk>")

print(f"target vocabulary path: {path_to_zh_vocab}")
# Importing Chinese Vocab
zh_vocab = torch.load(path_to_zh_vocab)
zh_vocab_size = len(zh_vocab.get_itos())
zh_vocab.set_default_index(zh_vocab_size)
zh_vocab.append_token("<unk>")

load_path = "saved/model-interupted.pt"

model = Transformer(PAD_ID, PAD_ID, th_vocab_size, zh_vocab_size, MAX_LENGTH, D_MODEL, N_HEADS, D_FF,
                    N_LAYERS, DROP_PROB, device).to(device)
print(f"loading parameters from {load_path}")
model.load_state_dict(torch.load(load_path))

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

def tensor_to_string(tensor,vocab,padding_id=0, end_id = 2):
    str = ''
    prior_id = -1
    for id in list(tensor):
        if(id == padding_id or prior_id == end_id):
            break
        str += vocab.get_itos()[id] + ' '
        prior_id = id

    return str

if __name__ == "__main__":
    args = parser.parse_args()
    exp = args.input
    exp_out = example(exp)
    print(f"th: \'{exp}\' zh: \'{exp_out}\'")

