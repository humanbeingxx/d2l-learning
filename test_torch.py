import matplotlib.pyplot as plt
import torch


def generate_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return y, y.reshape((-1, 1))


y, ry = generate_data(torch.tensor([2, -3.4]), 4, 1000)

# plt.scatter([1] * len(y), y, label="y", alpha=0.5)
# plt.legend()
# plt.show()

k = torch.ones(3)
# k = k.reshape((-1, 1))
k = k.reshape(-1, 1)
print(k.shape)
print(k)

a = [[1, 2], [3, 4]]

print(len(a))

a = [1, 2]
a = torch.tensor(a)
print(a * 2)
print(a / 2)

print(a[0:1])
