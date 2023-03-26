import torch
import torch.nn as nn
from torch.nn import functional as F


# model hyperparameters
block_size = 8  # length of the context
batch_size = 32  # length of the context
learning_rate = 1e-3  # how fast to update the model
n_embed = 32  # embedding dimension
n_heads = 4  # number of heads in multi-head attention
n_layer = 4  # layers of heads in multi-head attention
dropout = 0.02  # used to prevent overfitting during training, shuts off some nodes randoly each pass
test_train_ratio = 0.9


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # compute attention weights
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # scale by 1/sqrt(head_size) to preserve variance of softmax
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # ensures future tokens are always zeroed out so we only look at previous tokens
        wei = F.softmax(wei, dim=-1)  # softmax over rows to ensure each row sums to one

        wei = self.dropout(wei)

        # weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embed, n_embed)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """Simple Linear Feed forward layers with reinforcement in between"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4), 
            nn.ReLU(), 
            nn.Linear(n_embed * 4, n_embed)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: combines self-attention with feed-forward"""

    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLangaugeModel(nn.Module):
    """Simple statistical Language Model that predicts pairings of words/characters"""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads) for _ in range(n_layer)],
            nn.LayerNorm(n_embed),  # normalize the output of the last block
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (batch_size, block_size) or (B, T) tensors
        # This arranges them in a (B, T, C); C = n_embed and will be used to
        # hold the "scores" for the next character in the emebdding sequence
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape so it fits cross entropy loss function
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(
                logits, targets
            )  # compute loss, or quality of prediction between logits and targets

        return logits, loss

    def generate_all(self, idx, max_len_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_len_tokens):
            idx = self.generate_next(idx)

        return idx
    
    def generate_next(self, idx):
        # crop idx to last blockzie tokens
        idx_cond = idx[:, -block_size:]  # (B, T)
        # get the logits (predictions)
        logits, loss = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # (B, C)
        # get probability distribution over the vocab
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # append the new index to the sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx


class GPT:
    """Decoding only generative pre-trained transformer to generate text from a given input"""

    def __init__(
        self,
        max_len,
        max_iters,
        eval_iters,
        eval_interval,
        manual_seed=42,
        stream=True,
        input_file="data/nietzsche_aphorisms.txt",
    ):
        torch.manual_seed(manual_seed)

        self.max_len = max_len
        self.input_file = input_file
        self.stream = stream

        self.max_iters = max_iters
        self.eval_iters = eval_interval
        self.eval_iters = eval_iters

        text, chars, vocab_size = self.load_data()

        # tokenize the text
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}
        self.encode = lambda x: [stoi[c] for c in x]
        self.decode = lambda x: "".join([itos[c] for c in x])

        # encode dataset as a torch tensor
        data = torch.tensor(self.encode(text), dtype=torch.long)

        # do train/val split
        n = int(test_train_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        # initialize language model & optimizer
        self.model = BigramLangaugeModel(vocab_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_data(self):
        # import text to train on
        with open(self.input_file, encoding="utf-8") as f:
            text = f.read()

        print("\nInput text name:", str(self.input_file), "\n")
        print("Input text length:", len(text), "\n")
        print("Input text first chars:\n ", text[:1000])

        # create a set of all the unique characters in the text
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        return text, chars, vocab_size

    def train(self):
        print("\n Training...")
        for i in range(self.max_iters):
            if i % self.eval_iters == 0:
                losses = self.estimate_loss()
                print(
                    f"iteration {i}, train loss is {losses['train']}, val loss is {losses['val']}"
                )

            # get a batch of data
            xb, yb = self.get_batch("train")

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def generate(self):
        print("\n Generating...")
        idx = torch.zeros((1, 1), dtype=torch.long)
        if self.stream:
            for i in range(self.max_len):
                idx = self.model.generate_next(idx)
                print(f"\n Generating token {i + 1} ... \n")
                print(self.decode(idx.tolist()[0]))
        else:
            print(
                self.decode(
                    self.model.generate_all(idx, max_len_tokens=self.max_len).tolist()[0]
                )
            ) 

    def get_batch(self, split):
        # generate a batch of data on inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size - 1, size=(batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                xb, yb = self.get_batch(split)
                _, loss = self.model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
