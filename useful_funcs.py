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

d = [[1.1, 2.1, 9.9], [3.1, 4.1, 9.9], [5.1, 5.6, 9.9]]
t = torch.tensor(d)
print(f"original tensor:", t)
print(f"first row: {t[0]}")
print(f"first col: {t[:, 0]}")
print(f"last col: {t[..., -1]}")

t[:,-1] = 0 
print(f"original tensor after edit:", t)