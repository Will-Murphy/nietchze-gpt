
import torch 
import torch.nn as nn
from torch.nn import functional as F


"""Demonstration of the mathematical "trick" used in transformer attention"""

def get_weights():
    torch.manual_seed(1337)

    B, T, C = 4, 8, 2 # batch, time, channels
    x = torch.randn(B, T, C)

    print(x)
    print(x.shape)

    # We want the tokens the interact with each previous token, so we taken the average
    # of all previous t in a the given channel

    # xbow is x "bag of words" or average of previous context at a given t
    xbow = torch.zeros((B, T, C)) # x[b,t] = mean_{i<t} x[b,i]
    for b in range(B):
        for t in range(T):
            xprev = x[b,:t+1] # (t, C)
            xbow[b,t] = torch.mean(xprev, dim=0) # (t)
    
    print("Compare raw to x input to xbow (which is the backward looking moving average of x)")        
    print({x[0]})
    print({xbow[0]})
    
    return xbow

# however this ^ is not efficient the "trick" is to do this with matrix multiplication

xbow = get_weights()

# toy matrix multiplication example to start

def get_weights_v2(xbow):
    B, T, C = 4, 8, 2 # batch, time, channels
    x = torch.randn(B, T, C)

    print(x)
    print(x.shape)
    
    torch.manual_seed(42)
    a = torch.ones(3,3)
    b = torch.randint(0, 10, (3,2)).float()
    c = a @ b # matrix multiplication

    print(f"a = \n{a}")
    print(f"b = \n{b}")
    print(f"c = a @b = \n{c}")

    # now we do the same but but use lower triangle matrix to get a moving sum
    a = torch.tril(torch.ones(3,3)) # lower triangle matrix
    c = a @ b # matrix multiplication, yields moving sum of a and b

    print(f"a = \n{a}")
    print(f"b = \n{b}")
    print(f"c = a @b = \n{c}")

    # now we normalize the lower trianglar so each row sums to one, and we the moving average!
    a = torch.tril(torch.ones(3,3)) # lower trianglur ones
    a = a / torch.sum(a, dim=1, keepdim=True) # normalize each row to sum to one
    c = a @ b # matrix multiplication, yields moving average of a and b

    print(f"a = \n{a}")
    print(f"b = \n{b}")
    print(f"c = a @b = \n{c}")


def get_weights_v3(xbow):
    B, T, C = 4, 8, 2 # batch, time, channels
    x = torch.randn(B, T, C)

    print(x)
    print(x.shape)
    
    # now, use this trick to create our "bag of words" 
    wei = torch.tril(torch.ones(T,T)) # lower trianglur ones (T, T) 
    # we normalize by 1/sqrt(head_size) to ensure the variance of the output is the same as the input
    wei = wei / torch.sum(wei, dim=1, keepdim=True)   #weighted for producing moving average
    xbow2 = wei @ x # (B, T, T) @ (B, T, C) --> (B, T, C)

    print("Xbow and xbow2 should be equivalent")
    print(f"xbow2[0] = \n{xbow2[0]}")
    print(f"xbow[0] = \n{xbow[0]}")

    equal = torch.allclose(xbow, xbow2)
    print(f"Are xbow and xbow2 equal? {equal}")
    
    return xbow2


def get_weights_v4(xbow):
    B, T, C = 4, 8, 2 # batch, time, channels
    x = torch.randn(B, T, C)

    print(x)
    print(x.shape)

    # version 4: open the door beyond just moving average to arbitrary function of previous tokens
    tril = torch.tril(torch.ones(T,T)) # lower trianglur ones (T, T)
    wei = torch.zeros((T,T)) # (T, T)
    wei = wei.masked_fill(tril==0, float("-inf")) # ensures future tokens are always zeroed out so we only look at previous tokens
    wei = F.softmax(wei, dim=-1) # softmax over rows to ensure each row sums to one

    xbow3 = wei @ x # (B, T, T) @ (B, T, C) --> (B, T, C)
    equal = torch.allclose(xbow, xbow3)
    print(f"Xbow and xbow3 should be equivalent? {equal}")
    

# version 5: add a data depenedent bias to the attention weights instead of just moving average
def get_weights_v5():
    torch.manual_seed(1337)

    B, T, C = 4, 8, 32 
    x = torch.randn(B, T, C)
    print(x)
    print(x.shape)

    # add a single attention head
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)

    # forard these on 
    k = key(x) # (B, T, head_size)  
    q = query(x) # (B, T, head_size)
    # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
    wei = q @ k.transpose(-2, -1) * head_size**-0.5 # scale by 1/sqrt(head_size) to ensure the variance of the output is the same as the input

    tril = torch.tril(torch.ones(T,T)) # lower trianglur ones (T, T) 

    # If this line is removed, all tokens will be able to attend to all other tokens and it will be an "encoder" block 
    # (for something like sentiment analysis) instead of "decoder" block, which is what we want for the "bag of words"
    # representation, or text generation moving forward in time
    wei = wei.masked_fill(tril==0, float("-inf")) # ensures future tokens are always zeroed out so we only look at previous tokens
    wei = F.softmax(wei, dim=-1) # softmax over rows to ensure each row sums to one


    # this output is called "self attention" because the query, key, and value are all from the same source "x"
    v = value(x) # (B, T, head_size)
    out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)

    print("out.shape", out.shape, "\n")
    print("out", out[0], "\n")
    print("wei.shape", wei.shape, "\n")
    print("wei", wei[0], "\n")


    # we finally normalize by 1/sqrt(head_size) to ensure the variance of the output is the same as the input
    print("\n After normalizing by 1/sqrt(head_size), variance is retained \n")
    print("key variance k.var()", k.var(), "\n")
    print("query variance q.var()", q.var(), "\n")
    print("value variance v.var()", wei.var(), "\n")
    
get_weights_v5()



