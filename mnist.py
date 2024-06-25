import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)


class MNISTPointCloudDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = sample[0]
        points = sample[1:].reshape(-1, 3)  # Reshape into (x, y, v) format
        # Filter out points where x, y, v are all -1
        valid_points = points[~(points == -1).all(axis=1)]
        valid_points = torch.tensor(valid_points, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return valid_points, label


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ClassifierNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassifierNet, self).__init__()
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        x_sum = torch.sum(x, dim=1)
        x = self.fc(x_sum)
        x = F.softmax(x, dim=1)
        return x


def gaussian_kernel(x, y, sigma):
    beta = 1. / (2. * sigma ** 2)
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-beta * dist)


def compute_mmd(x, y, sigma=1.0):
    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    zz = gaussian_kernel(x, y, sigma)

    return xx.mean() + yy.mean() - 2 * zz.mean()


# Hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 64
num_iterations = 10  # Number of iterations for MMD flow
sigma = 1.0  # Kernel bandwidth parameter
beta = 0.01  # Regularization hyperparameter

# Load the datasets
train_dataset = MNISTPointCloudDataset('MNISTPointCloud/train.csv')
test_dataset = MNISTPointCloudDataset('MNISTPointCloud/test.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
point_net = PointNet().to(device)
classifier_net = ClassifierNet(num_classes=10).to(device)


# Optimizers
optimizer = torch.optim.Adam(list(point_net.parameters()) + list(classifier_net.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1} of {num_epochs}')
    for batch in train_loader:
        points_batch, labels_batch = zip(*batch)

        # Stack points and labels, handling variable lengths
        points_batch = [p.to(device) for p in points_batch]
        labels_batch = torch.tensor(labels_batch, dtype=torch.long).to(device)

        all_points = torch.cat(points_batch)

        # Run MMD flow until convergence without gradient tracking
        X = all_points.clone().detach().requires_grad_(True)
        for i in range(num_iterations):
            U = torch.randn_like(X)  # Gaussian noise with the same shape as X, as in MMD Flow paper
            X_regularized = X + beta * U
            F_X = point_net(X_regularized)
            loss = compute_mmd(X_regularized, F_X, sigma)
            grad = torch.autograd.grad(loss, X_regularized, retain_graph=True)[0]
            X = X - learning_rate * grad
            print(f'Iteration {i + 1} of MMD, loss {loss.item():.4f}')

        # Disable gradient tracking for the MMD flow part
        X = X.detach().requires_grad_(True)
        F_X = point_net(X)

        # Pass the updated points through the classifier
        optimizer.zero_grad()
        outputs = []
        start_idx = 0
        for points in points_batch:
            end_idx = start_idx + points.size(0)
            output = classifier_net(F_X[start_idx:end_idx].unsqueeze(0))
            outputs.append(output)
            start_idx = end_idx

        outputs = torch.cat(outputs)
        classification_loss = F.cross_entropy(outputs, labels_batch)
        classification_loss.backward()


        # Backpropagate through F(X*) using implicit differentiation
        def backward_hook(grad):
            implicit_grad = torch.autograd.grad(F_X, X, grad_outputs=grad)[0]
            return implicit_grad


        # Register the hook on F(X*) to use implicit differentiation
        F_X.register_hook(backward_hook)

        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {classification_loss.item():.4f}')