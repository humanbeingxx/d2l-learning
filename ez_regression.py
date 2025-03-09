import manual_regression
import torch
from torch import nn
from torch.utils import data


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 1.5
    features, labels = manual_regression.generate_data(true_w, true_b, 10000)

    batch_size = 100
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3

    for epoch in range(num_epochs):
        for X, y in data_iter:
            ls = loss(net(X), y)
            trainer.zero_grad()
            ls.backward()
            trainer.step()
        ls = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {ls:f}")
