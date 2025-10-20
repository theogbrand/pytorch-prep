import torch
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F

d = [1, 2]
t = torch.tensor(d)
print(t.device)
print(t.shape)
print(t.dtype)

d = [[1.1, 2.1], [3.1, 4.1]]
t = torch.tensor(d)
print(t.device)
print(t.shape)
print(t.dtype)

shape = (2, 3)
rand_tens = torch.rand(shape)
ones = torch.ones(shape)
zeroes = torch.zeros(shape)

print(rand_tens)
print(ones)
print(zeroes)

d = [[1.1, 2.1, 9.9], [3.1, 4.1, 9.9], [5.1, 5.6, 9.9]]
t = torch.tensor(d)
print(f"original tensor:", t)
print(f"first row: {t[0]}")
print(f"first col: {t[:, 0]}")
print(f"last col: {t[..., -1]}")

t[:,-1] = 0 
print(f"original tensor after edit:", t)

d = [[0,1,2,3]]
t = torch.tensor(d)
t1 = torch.cat([t,t],dim=1) # CONCAT along existing dim
print(f"torch.cat t1 dim=1: {t1}")
t1 = torch.cat([t,t],dim=0)
print(f"torch.cat t1 dim=0: {t1}")

s1 = torch.stack([t,t],dim=1) # creates new dim at specified position (adds one dimension)
print(f"torch.stack t1 dim=1: {s1}")
s1 = torch.stack([t,t],dim=0)
print(f"torch.stack t1 dim=0: {s1}")


# Convert to one-hot encoding
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

def flatten(x: torch.Tensor) -> torch.Tensor:
    y = x.view(-1)
    return y

def transpose_and_reverse(x: torch.Tensor) -> torch.Tensor:
    # reverse from 1D to 2D with 2* rows
    result = x.view(2, -1)
    # reverse rows to columns
    result = torch.transpose(input=x, dim0=0, dim1=1) # dim0 and dim1 are swapped during transpose
    return result

def slice_and_sum(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x[::2], dim=1)

x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 5.0, 3.0]])
print(F.softmax(x, dim=1))
# tensor([[0.0900, 0.2447, 0.6652],
#         [0.0159, 0.8668, 0.1173]])
print(F.softmax(x,dim=0))
# tensor([[0.5000, 0.0474, 0.5000],
#         [0.5000, 0.9526, 0.5000]])

x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 5.0, 3.0]])
x # flattens and averages across if no dim specified
# tensor([[1., 2., 3.],
#         [1., 5., 3.]])
torch.mean(x) 
# tensor(2.5000)
torch.mean(x, 1)
# tensor([2., 3.])

# "Pure" tensor-based variance calculation instead of using .item() to perform elem-wise ops
t = torch.tensor([1,2,3])
t_mean = torch.mean(x)
sq_diff = (t - t_mean)**2
var = torch.mean(sq_diff)
print(var)

# the point of keepdim=True
def softmax(logits: torch.Tensor) -> torch.Tensor:
    # keepdim so shape is consistent with existingly operated tensors, easier for follow up ops
    t_norm = logits - torch.max(logits, dim=1, keepdim=True).values # columns represent class proba, hence dim=1
    t_exp = torch.exp(t_norm)
    t_sum = torch.sum(t_exp, dim=1, keepdim=True) # sum across dim=1 reduces the dim, without keepdim, manual .view(-1,1) required for broadcasting later
    z = t_exp / t_sum
    return z

logits = torch.tensor([[1.,2.,3.], [2.,4.,6.]])
assert torch.allclose(softmax(logits), F.softmax(logits, dim=1))

torch.arrange()

# vertically stack tensors
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y