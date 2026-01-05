# training/run_baseline.py
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from models.hybrid_baseline import HybridClassifier
from training.train_eval import train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV features
X = np.loadtxt("data/train.csv", delimiter=",", skiprows=1)[:, :-1]
Y = np.loadtxt("data/train.csv", delimiter=",", skiprows=1)[:, -1].astype(int)

dataset = TUDataset(root="./data", name="ENZYMES")
for i, g in enumerate(dataset):
    g.graph_id = i

moment_tensor = torch.tensor(X, dtype=torch.float32).to(device)

idx_train, idx_test = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=Y, random_state=42)
train_loader = DataLoader([dataset[i] for i in idx_train], batch_size=32, shuffle=True)
test_loader = DataLoader([dataset[i] for i in idx_test], batch_size=32)

model = HybridClassifier(gcn_out_dim=64, moment_dim=X.shape[1], num_classes=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    train(model, train_loader, optimizer, criterion, moment_tensor, device)

acc, report = evaluate(model, test_loader, moment_tensor, device)
print(f"Baseline Accuracy: {acc:.4f}")

