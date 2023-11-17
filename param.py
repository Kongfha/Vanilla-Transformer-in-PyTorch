import torch

PAD_ID = 0
START_ID = 1
END_ID  =2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 10

# Transformer Parameter
MAX_LENGTH = 256
D_MODEL = 512
N_HEADS = 8
D_FF = 2048
N_LAYERS = 6
DROP_PROB = 0.25
print(f"Dropout prob: {DROP_PROB}")

# Optimizer Parameter
INIT_LR = 1e-5
FACTOR = 0.9
ADAM_EPS = 5e-9
PATIENCE = 10
WARMUP = 100
CLIP = 1.0
WEIGHT_DECAY = 5e-4
INF = float('inf')

print(f"Epoch = {EPOCH}")