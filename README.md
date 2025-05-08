import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
pressure_data = pd.read_csv("/content/drive/MyDrive/Copy of artificial_pressure_data_500.csv")
tectonics_data = pd.read_csv("/content/drive/MyDrive/Tectonic.csv")
gwl_data = pd.read_csv("/content/drive/MyDrive/gwl-monthly.csv")

# Define features
pressure_features = ['Pressure (Pa)', 'Signal']
tectonics_features = ['lat', 'lon']
gwl_features = ['WSE']

# Select relevant columns
pressure_data = pressure_data[pressure_features]
tectonics_data = tectonics_data[tectonics_features]
gwl_data = gwl_data[gwl_features]

# Feature Engineering: Add past WSE values
gwl_data["WSE_lag1"] = gwl_data["WSE"].shift(1)
gwl_data["WSE_lag2"] = gwl_data["WSE"].shift(2)
gwl_data["WSE_lag3"] = gwl_data["WSE"].shift(3)
gwl_data.dropna(inplace=True)

# Merge datasets
data = pd.concat([pressure_data, tectonics_data, gwl_data], axis=1)
data.dropna(inplace=True)

# Convert target to binary
threshold = 1.0
Y_binary = (data['WSE'] > threshold).astype(int)

# Normalize and apply PCA
scaler = StandardScaler()
X = scaler.fit_transform(data.iloc[:, :-1])
Y = Y_binary

pca = PCA(n_components=4)
X_fused = pca.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_fused, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y.values, dtype=torch.float32).unsqueeze(1).to(device)

# Train-test split
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
train_dataset, test_dataset = random_split(TensorDataset(X_tensor, Y_tensor), [train_size, test_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define Model
class EarthquakeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(EarthquakeModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize Model
input_dim = X_tensor.shape[1]
model = EarthquakeModel(input_dim).to(device)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training
num_epochs = 200
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
latencies = []

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    epoch_loss = 0
    correct, total = 0, 0

    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = (Y_pred > 0.5).float()
        correct += (preds == Y_batch).sum().item()
        total += Y_batch.size(0)

    train_losses.append(epoch_loss / len(train_loader))
    train_accuracies.append(correct / total)

    # Validation
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_pred = model(X_batch)
            val_loss += criterion(Y_pred, Y_batch).item()
            preds = (Y_pred > 0.5).float()
            correct += (preds == Y_batch).sum().item()
            total += Y_batch.size(0)

    test_losses.append(val_loss / len(test_loader))
    test_accuracies.append(correct / total)

    # Record latency
    latencies.append(time.time() - epoch_start)

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

print("Training complete.")

# Final Predictions & Metrics
model.eval()
Y_true, Y_preds = [], []

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        Y_pred = model(X_batch)
        Y_preds.extend((Y_pred > 0.5).cpu().numpy())
        Y_true.extend(Y_batch.cpu().numpy())

# Convert to numpy
Y_true = np.array(Y_true).flatten()
Y_preds = np.array(Y_preds).flatten()

# Compute Metrics
accuracy = accuracy_score(Y_true, Y_preds)
precision = precision_score(Y_true, Y_preds)
recall = recall_score(Y_true, Y_preds)
f1 = f1_score(Y_true, Y_preds)

print(f"\nFinal Model Metrics:")
print(f" Accuracy: {accuracy * 100:.2f}%")
print(f" Precision: {precision * 100:.2f}%")
print(f" Recall: {recall * 100:.2f}%")
print(f" F1-Score: {f1 * 100:.2f}%")

# Plot 1: Training vs Testing Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_accuracies, label="Train Accuracy", color="blue")
plt.plot(range(num_epochs), test_accuracies, label="Test Accuracy", color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy Over Epochs")
plt.legend()
plt.show()

# Plot 2: Accuracy Over Time
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), test_accuracies, marker="o", linestyle="-", color="green")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.title("Accuracy Over Time")
plt.show()

# Plot 3: Latency Over Time
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), latencies, marker="o", linestyle="-", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Training Time (seconds)")
plt.title("Latency Over Time")
plt.show()
