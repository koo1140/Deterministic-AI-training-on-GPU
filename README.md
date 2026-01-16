# Deterministic-AI-training-on-GPU
Deterministic AI training on GPU

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=koo1140.Deterministic-AI-training-on-GPU)

# Output example
```
🚀 Training on: cuda
Step 1: Loading and Tokenizing...
Scanning Vocab:   0%|          | 0/20000 [00:00<?, ?it/s]'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 8262726a-0f13-46ec-a435-c18ce7384d9c)')' thrown while requesting GET https://huggingface.co/datasets/pansophic/newsophy-v0.1/resolve/4574544c2c3dabc1a5d2ca1143d9fb86ff908b12/newsophy-v0.1.json
Retrying in 1s [Retry 1/5].
Scanning Vocab: 100%|██████████| 20000/20000 [00:45<00:00, 437.18it/s] 
Step 2: Initializing GPU Matrices...
Step 3: Solving (Saturating P100)...
GPU Processing: 100%|██████████| 425/425 [02:26<00:00,  2.91it/s] 
Step 4: Finalizing Weights (Linear Solve)...

✅ Done! Solved 6956087 patterns in 192.33s
```

# Script
```py
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import time

# ---------------------------
# 1. Configuration & Device
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "pansophic/newsophy-v0.1"
TRAIN_SUBSET = 20000
CONTEXT_SIZE = 24
EMBED_DIM = 64
K_NEURONS = 10000   # Hidden neurons (compression bottleneck)
LAMBDA_REG = 0.5   # Regularization
BATCH_SIZE = 16384 # Large batch to saturate GPU VRAM
SAVE_PATH = "vordex_p100_model.pth"

print(f"🚀 Training on: {DEVICE}")

# ---------------------------
# 2. Data Loading & Tokenization (CPU)
# ---------------------------
print("Step 1: Loading and Tokenizing...")
dataset = load_dataset(DATASET_NAME, split='train', streaming=True).take(TRAIN_SUBSET)

all_ids = []
all_tokens = set()
processed_lines = []

# First pass: build vocab
for item in tqdm(dataset, total=TRAIN_SUBSET, desc="Scanning Vocab"):
    text = f"<s> <user> {item['user']} <bot> {item['assistant']} </bot> <eos>"
    processed_lines.append(text.split())
    all_tokens.update(text.split())

tokens = sorted(list(all_tokens))
token2id = {tok: i for i, tok in enumerate(tokens)}
id2token = {i: tok for tok, i in token2id.items()}
vocab_size = len(tokens)

# Second pass: Flatten everything into one massive GPU tensor
# We pad each sentence with CONTEXT_SIZE zeros to handle starts
for tokens_list in processed_lines:
    ids = [token2id[t] for t in tokens_list]
    all_ids.extend([0]*CONTEXT_SIZE + ids)

all_ids_gpu = torch.tensor(all_ids, dtype=torch.long, device=DEVICE)

# ---------------------------
# 3. GPU Matrix Initialization
# ---------------------------
print("Step 2: Initializing GPU Matrices...")
embeddings = torch.randn(vocab_size, EMBED_DIM, device=DEVICE) * 0.1

input_dim = (CONTEXT_SIZE * EMBED_DIM) + 1
# Xavier/Glorot Initialization
limit = np.sqrt(6 / (input_dim + K_NEURONS))
W1 = (torch.rand(input_dim, K_NEURONS, device=DEVICE) * 2 * limit - limit)

# Knowledge Accumulators
A = torch.zeros((K_NEURONS, K_NEURONS), device=DEVICE)
B = torch.zeros((K_NEURONS, EMBED_DIM), device=DEVICE)

# ---------------------------
# 4. Vectorized Deterministic Solve
# ---------------------------
print("Step 3: Solving (Saturating P100)...")
start_time = time.time()

# Create sliding window views: [Total_Tokens, Context_Size + 1]
# This is a 'view', it doesn't duplicate memory!
windows = all_ids_gpu.unfold(0, CONTEXT_SIZE + 1, 1)

for i in tqdm(range(0, windows.size(0), BATCH_SIZE), desc="GPU Processing"):
    batch = windows[i : i + BATCH_SIZE]
    
    # Check if we hit the end
    if batch.size(0) < 1: continue
    
    ctx_ids = batch[:, :-1]  # The context window
    tgt_ids = batch[:, -1]   # The word to predict
    
    # GATHER: GPU pulls all embeddings in parallel
    # Shape: [Batch, Context, Embed] -> flatten to [Batch, Context*Embed]
    X_batch = embeddings[ctx_ids].view(ctx_ids.size(0), -1)
    
    # Add Bias term
    bias = torch.ones((X_batch.size(0), 1), device=DEVICE)
    X_batch = torch.cat([X_batch, bias], dim=1)
    
    Y_batch = embeddings[tgt_ids]
    
    # Deterministic Logic (The Math)
    H = torch.tanh(X_batch @ W1) # Batch Projection
    A += H.t() @ H               # Covariance update
    B += H.t() @ Y_batch         # Target update

# The Final Global Solve
print("Step 4: Finalizing Weights (Linear Solve)...")
A += LAMBDA_REG * torch.eye(K_NEURONS, device=DEVICE)
W2 = torch.linalg.solve(A, B)

end_time = time.time()
print(f"\n✅ Done! Solved {windows.size(0)} patterns in {end_time - start_time:.2f}s")

# ---------------------------
# 5. Save Model
# ---------------------------
torch.save({
    'W1': W1.cpu(),
    'W2': W2.cpu(),
    'embeddings': embeddings.cpu(),
    'token2id': token2id,
    'id2token': id2token,
    'config': {'context_size': CONTEXT_SIZE, 'k': K_NEURONS, 'embed_dim': EMBED_DIM}
}, SAVE_PATH)
```
