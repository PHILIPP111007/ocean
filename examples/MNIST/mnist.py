import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# ----------------------------
# 1. Определение модели
# ----------------------------
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)  # для NLLLoss

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)            # без активации, потом LogSoftmax
        return self.log_softmax(x)

# ----------------------------
# 2. Загрузка данных (MNIST)
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# ----------------------------
# 3. Инициализация модели, оптимизатора, функции потерь
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)
criterion = nn.NLLLoss()   # Negative Log Likelihood (подходит с LogSoftmax)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 4. Обучение
# ----------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Валидация после эпохи
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')

print("Обучение завершено.")

# ----------------------------
# 5. Сохранение весов в .npy
# ----------------------------
# Создадим папку для весов
os.makedirs("./examples/MNIST/weights", exist_ok=True)

# Сохраняем веса и смещения для каждого слоя
# В PyTorch weight имеет форму (out_features, in_features)
# В Ocean ожидается (in_features, out_features) — транспонируем сразу.
for name, param in model.named_parameters():
    # name будет например "fc1.weight", "fc1.bias", "fc2.weight", ...
    # Преобразуем в numpy, переводим на CPU
    np_data = param.detach().cpu().numpy()
    if 'weight' in name:
        # Транспонируем для совместимости с Ocean
        np_data = np_data.T   # теперь (in_features, out_features)
    elif 'bias' in name:
        np_data = np_data.reshape(1, -1)
    # Сохраняем с понятным именем
    filename = f"./examples/MNIST/weights/{name.replace('.', '_')}.npy"
    np_data = np.ascontiguousarray(np_data)
    np.save(filename, np_data)
    print(f"Saved {filename} with shape {np_data.shape}")

print("Все веса сохранены в папке ocean_weights/")

image, label = test_set[0]
# Преобразуем в плоский вектор (1, 784)
flat_image = image.view(1, -1).numpy()
np.save("./examples/MNIST/data/one_sample_mnist.npy", flat_image)
print(f"Label: {label}")
