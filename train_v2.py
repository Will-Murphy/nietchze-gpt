import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
block_size = 8 # length of the context
batch_size = 32 # length of the context
learning_rate = 1e-3 # how fast to update the model
max_iters = 5000 
eval_interval = 500 
eval_iters = 200
n_embed = 32
n_heads = 4

torch.manual_seed(1337)

# import text to train on
with open('data/nietzsche.txt', encoding='utf-8') as f:
    text = f.read()
    
print('text length:', len(text))
print('text[:1000]:', text[:1000])
    
# create a set of all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('vocab size:', vocab_size)
print(''.join(chars))

# tokenize the text
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[c] for c in x])

print(encode('This is a test'))
print(decode(encode('This is a test')))

# encode dataset as a torch tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

# do train/val split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# random sampling of context and target with real data
print("\ntesting context and target with real data sample & batching")

def get_batch(split):
    # generate a batch of data on inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
        
# estimate loss during training
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

class Head(nn.Module):
    """ One head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
   
        k = self.key(x) # (B, T, head_size)  
        q = self.query(x) # (B, T, head_size)
        
        # compute attention weights
        wei = q @ k.transpose(-2, -1) * C**-0.5 # scale by 1/sqrt(head_size) to preserve variance of softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # ensures future tokens are always zeroed out so we only look at previous tokens
        wei = F.softmax(wei, dim=-1) # softmax over rows to ensure each row sums to one

        # weighted aggregation of values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        
        return out
    
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel"""
    
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embed, n_embed)

        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return out
      
class FeedFoward(nn.Module):
    """ Simple Linear Feed forward layer followed by non-linearity """
    
    def __init__(self, n_embed):
        super().__init__()
        self.non_linear =nn.Linear(n_embed, n_embed), nn.ReLU(), nn.Linear(n_embed, n_embed)
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4), 
            nn.ReLU(), 
            nn.Linear(n_embed* 4, n_embed)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: combines self-attention with feed-forward """
    def __init__(self, n_embded, n_head):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedFoward(n_embed)
        
    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
            

    
        
        
# implement langauge model to train
class BigramLangaugeModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        n_transf_blocks = 3
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads) for _ in range(n_transf_blocks)]
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        
        # idx and targets are both (batch_size, block_size) or (B, T) tensors
        # This arranges them in a (B, T, C); C = n_embed and will be used to
        # hold the "scores" for the next character in the emebdding sequence
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = token_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # reshape so it fits cross entropy loss function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # compute loss, or quality of prediction between logits and targets
        
        return logits, loss
    
    def generate(self, idx, max_len_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_len_tokens):
            # crop idx to last blockzie tokens 
            idx_cond = idx[:, -block_size:] # (B, T)
            # get the logits (predictions)
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # get probability distribution over the vocab
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the new index to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
           
    
model = BigramLangaugeModel()

# test generate function to get some predictions (text)!
idx = torch.zeros((1,1), dtype=torch.long)
print("Untrained we generate")
print(decode(model.generate(idx, max_len_tokens=100).tolist()[0])) # will be random

# Get pytorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        
for i in range(max_iters):
    
    if i % eval_iters == 0:
        losses = estimate_loss()
        print(f"iteration {i}, train loss is {losses['train']}, val loss is {losses['val']}")
        
        
    # get a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print("And we generate")
print(decode(model.generate(idx, max_len_tokens=1000).tolist()[0])) # should be better