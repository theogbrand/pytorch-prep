import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

for X, y in train_dataloader: 
    print(X.shape) # B, C, H, W
    print(y.shape, y.dtype) 
    print(X.dim(), y.dim())
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_network = nn.Sequential(
            nn.Linear(28*28, 512), # y = mx + c
            nn.ReLU(), # Activation Function f(x) = max(0, x)
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x) # 
        logits = self.linear_relu_network(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # prepare loss computation
        pred = model(X) # computes activations for batch
        loss = loss_fn(pred, y)

        # backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # clear for next step

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X) # total optimizer steps needed
            print(f"loss: {loss:>7f}, current: [{current:>5d}/{size:>5d}]")
        

# def test(dataloader, model, loss_fn):

epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train(train_dataloader, model, loss_fn, optim)

print("DONE!")