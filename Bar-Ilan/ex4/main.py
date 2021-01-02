import sys
import torch.nn.functional as F
import ModelA
import ModelB
import ModelC
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import torch


def main():
    train_x, train_y, test_x = loadData()
    train_loader, test_loader = toTens(64, train_x, train_y, test_x)
    model = ModelC.ModelC(image_size=28 * 28)
    optimizer = model.getOpt(model.parameters(), 0.001)
    # for every epoch send to test(take epoch out of function train here)
    train_model(model, train_loader, optimizer, 10)
    test_model(model, test_loader)
    z = 2


def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n Test set: Avarage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    z = 2


def train_model(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        print(epoch)
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, labels, reduction='sum')
            loss.backward()
            optimizer.step()



def toTens(batch_size, train_x, train_y, test_x):
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)
    # train_set = TensorDataset(train_x, train_y)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # for 1-2
    # train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    # test_x = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # for 3
    train_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', train=True, download=True, transform=transform)
    test_x = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # train_loader_y = DataLoader(train_y, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_x, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def loadData():
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2]).astype(int)
    test_x = np.loadtxt(sys.argv[3])

    # train_x = torch.from_numpy(train_x_np)
    # train_y = torch.from_numpy(train_y_np)
    # test_x = torch.from_numpy(test_x_np)

    return train_x, train_y, test_x


if __name__ == "__main__":
    main()
