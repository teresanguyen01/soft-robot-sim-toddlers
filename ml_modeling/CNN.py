import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_input_path = 'output_outliers/sensor_train.csv'
train_output_path = 'output_outliers/mocap_train.csv'
test_input_path = 'output_outliers/sensor_test.csv'
test_output_path = 'output_outliers/mocap_test.csv'

X_train = pd.read_csv(train_input_path).values
y_train_full = pd.read_csv(train_output_path).values
X_test = pd.read_csv(test_input_path).values
y_test_full = pd.read_csv(test_output_path).values

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_full.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test_full.shape}")

# Standardize input and output
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train_full)
y_test_scaled = scaler_y.transform(y_test_full)


# Function to create sequences for time-series modeling
def create_sequences(data, targets, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = targets[i + seq_length]  # Predict the next step's output
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 50  # Optimal window from IMU studies; tune based on your data (e.g., 20-100)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)

print(f"Sequenced X_train shape: {X_train_seq.shape}, y_train shape: {y_train_seq.shape}")
print(f"Sequenced X_test shape: {X_test_seq.shape}, y_test shape: {y_test_seq.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
y_test_tensor = torch.FloatTensor(y_test_seq).to(device)

# Create DataLoaders
batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define CNN-LSTM model for advanced sequence modeling
class CNNLSTM(nn.Module):
    def __init__(self, input_size, output_size, seq_length):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256 * 2, output_size)  # Bidirectional doubles hidden size

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq, features) -> (batch, features, seq) for Conv1D
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Back to (batch, seq, channels)
        x = self.dropout(x)
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)  # Concat bidirectional hidden states
        out = self.fc(hn)
        return out


# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]  # Features per timestep
output_size = y_train_full.shape[1]
model = CNNLSTM(input_size, output_size, seq_length).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-5)  # Switch to Adam for better convergence
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

# Training parameters
max_epochs = 5000
patience = 50
best_test_r2 = -float('inf')
no_improvement_count = 0

for epoch in tqdm(range(max_epochs), desc="Training Progress"):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)

    train_loss /= len(train_loader.dataset)

    # Evaluate on train and test
    model.eval()
    with torch.no_grad():
        y_train_pred_scaled = model(X_train_tensor)
        train_r2 = r2_score(y_train_tensor.cpu().numpy(), y_train_pred_scaled.cpu().numpy())

        y_test_pred_scaled = model(X_test_tensor)
        test_loss = criterion(y_test_pred_scaled, y_test_tensor).item()
        test_r2 = r2_score(y_test_tensor.cpu().numpy(), y_test_pred_scaled.cpu().numpy())

    print(
        f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train R2: {train_r2:.4f}, Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}")

    # Scheduler step on test R2 (maximize)
    scheduler.step(test_r2)

    # Early stopping
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        no_improvement_count = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        no_improvement_count += 1

    if no_improvement_count >= patience:
        print("Early stopping triggered.")
        break

# Load best model
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Final evaluation
with torch.no_grad():
    y_train_pred_scaled = model(X_train_tensor)
    y_test_pred_scaled = model(X_test_tensor)

# Convert predictions back to original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.cpu().numpy())
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.cpu().numpy())

# Calculate metrics (using sequenced y for consistency)
mse = mean_squared_error(y_test_seq, y_test_pred_scaled.cpu().numpy())
train_r2 = r2_score(y_train_seq, y_train_pred_scaled.cpu().numpy())

print("Mean Squared Error (MSE):", mse)
print("Training R2 Score:", train_r2)
print("Best Testing R2 Score:", best_test_r2)