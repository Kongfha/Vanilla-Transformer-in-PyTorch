import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from os.path import exists
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Importing Training Data')

path_src = "dataset/train.seg.th"
print(f'Path to train source: {path_src}')
th_src = open(path_src).read().splitlines()

path_trg = "dataset/train.seg.zh"
print(f'Path to train target: {path_trg}')
zh_target = open(path_trg).read().splitlines()

print(f'Importing Validation Data')

valid_path_src = "dataset/valid.seg.th"
print(f"Path to validation source: {valid_path_src}")
valid_th_src = open(valid_path_src).read().splitlines()

valid_path_trg = "dataset/valid.seg.zh"
print(f"Path to validation target: {valid_path_trg}")
valid_zh_trg = open(valid_path_trg).read().splitlines()

BATCH_SIZE = 24

path_to_th_vocab = "th_vocab.pth"
path_to_zh_vocab = "zh_vocab.pth"

def th_tokenizer(text):
    return text.split(' ')

def zh_tokenizer(text):
    return text.split(' ')

def add_tokens(vocab, padding_id=0, start_id=1, end_id=2):
    padding_token = '<pad>'
    start_token = '<start>'
    end_token = '<end>'
    vocab.insert_token(padding_token,padding_id)
    vocab.insert_token(start_token,start_id)
    vocab.insert_token(end_token,end_id)
    return vocab

def tensor_to_string(tensor,vocab,padding_id=0, end_id = 2):
    str = ''
    prior_id = -1
    for id in list(tensor):
        if(id == padding_id or prior_id == end_id):
            break
        str += vocab.get_itos()[id] + ' '
        prior_id = id

    return str
    
def get_padded_sequences(text_list, tokenizer, vocab, padding_id=0, start_id=1, end_id=2, max_length=256):
    seq = []
    for sentence in tqdm(text_list):
        cur = [start_id]
        for token in tokenizer(sentence):
            cur.append(vocab[token])
        cur.append(end_id)
        for _ in range(max_length - len(cur)):
            cur.append(padding_id)
        
        seq.append(torch.tensor(cur, dtype=torch.long).to(device))
        
    return seq

class ThaiChineseDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences
    
    def __len__(self):
        return len(self.src_sequences)
    
    def __getitem__(self, idx):
        return {'src':self.src_sequences[idx], 'trg':self.trg_sequences[idx]}



if exists(path_to_th_vocab) and exists(path_to_zh_vocab):
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

else:
    print(f"Creating vocabulary set")
    # Create a vocabulary for Thai input text
    th_vocab = add_tokens(build_vocab_from_iterator(map(th_tokenizer, th_src)))
    th_vocab_size = len(th_vocab.vocab.get_itos())
    th_vocab.set_default_index(th_vocab_size)

    # Create a vocabulary for Chinese target text
    zh_vocab = add_tokens(build_vocab_from_iterator(map(zh_tokenizer, zh_target)))
    zh_vocab_size = len(zh_vocab.get_itos())
    zh_vocab.set_default_index(zh_vocab_size)

print("Padding and Tokenizing Data")
# Convert each word in the Thai input text to its corresponding ID
th_sequences = get_padded_sequences(th_src, th_tokenizer, th_vocab)
zh_sequences = get_padded_sequences(zh_target, zh_tokenizer, zh_vocab)

valid_th_sequences = get_padded_sequences(valid_th_src, th_tokenizer, th_vocab)
valid_zh_sequences = get_padded_sequences(valid_zh_trg, zh_tokenizer, zh_vocab)

print("Making bathces")
print(f"BATCH_SIZE: {BATCH_SIZE}")
# Create a dataset from the Thai and Chinese sequences
dataset = ThaiChineseDataset(th_sequences, zh_sequences)
# Create a data loader with batch size 32 and shuffling
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create validation dataset
valid_dataset = ThaiChineseDataset(valid_th_sequences, valid_zh_sequences)
# Create a data loader for validation dataset
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

print("------------------------------")
