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
        pred = model(X) # computes activations for batch, this is the most mem intensive, in Grad Accum, we refer to accum grads across batches to prevent loading single large batch of activations at once
        loss = loss_fn(pred, y)

        # backprop
        loss.backward() # computes gradient for step, accumulates across batches (if any)
        optimizer.step() # parameters are stored here
        optimizer.zero_grad() # clear for next step

        if batch % 100 == 0: # every 100 optimizer steps
            loss, current = loss.item(), (batch+1) * len(X) # extract current loss, current batch size
            print(f"loss: {loss:>7f}, current: [{current:>5d}/{size:>5d}]")
        

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X) # shape (batch_size, 10)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    ave_loss = test_loss / num_batches
    acc = correct / size

    print(f"Test Accuracy: {(100*acc):>0.1f}%, Avg Loss: {ave_loss:>8f}\n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train(train_dataloader, model, loss_fn, optim)
    test(test_dataloader, model, loss_fn)
print("DONE!")


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[99][0], test_data[99][1] # x is a 1D tensor (no batch dim), not X
with torch.no_grad():
    x = x.to(device) # shape (10,)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')