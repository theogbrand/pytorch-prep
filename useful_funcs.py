import torch

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

