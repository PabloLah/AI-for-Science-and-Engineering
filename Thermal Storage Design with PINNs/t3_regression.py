import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Load CSV
data = pd.read_csv('housing.csv')
data['ocean_proximity'] = pd.factorize(data['ocean_proximity'])[0].astype(float)
data.dropna(inplace=True)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Step 4: Hyperparameters and Configurations
input_size = data.shape[1] - 1
hidden_size = 64 
output_size = 1 
learning_rate = 0.001
batch_size = 1000
num_epochs = 300
max_iter = 1000

model = NeuralNetwork(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer_ADAM = optim.Adam(model.parameters(), lr=learning_rate)
'''optimizer_LBFGS = optim.LBFGS(model.parameters(),
                              lr=float(0.5),
                              max_iter=max_iter,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
'''
optimizer = optimizer_ADAM

# make train test split
X_data = data.drop(columns=['median_house_value'])
y_data = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=42)

X_train_tensor = torch.tensor(X_train.values).float()
X_test_tensor = torch.tensor(X_test.values).float()
y_train_tensor = torch.tensor(y_train.values).float()
y_test_tensor = torch.tensor(y_test.values).float()

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()
        optimizer.step() 
        running_loss += loss.item() * inputs.size(0)
    # Calculate average training loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {np.sqrt(epoch_loss):.4f}")

# Evaluation loop
model.eval()  
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

# Calculate average test loss
avg_test_loss = test_loss / len(test_loader.dataset)
print(f"Average Test Loss RMSE: {np.sqrt(avg_test_loss) :.4f}")

