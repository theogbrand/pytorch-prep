import torch
from torchvision.transforms import ToTensor, Lambda

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