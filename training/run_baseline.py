# training/run_baseline.py
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader

from models.hybrid_baseline import HybridClassifier
from training.train_eval import train, evaluate

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Helper: load CSV into PyG Data objects
# ---------------------------
def load_graph_dataset(csv_file="data/train.csv"):
    """
    Load graphs from CSV. Assumes:
      - last column is label
      - other columns are moment-based features
    Each graph is represented as a single-node Data object (simplest placeholder).
    """
    X = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    features = X[:, :-1]
    labels = X[:, -1].astype(int)

    graphs = []
    for i in range(len(labels)):
        x = torch.tensor(features[i:i+1], dtype=torch.float32)  # node feature
        y = torch.tensor([labels[i]], dtype=torch.long)
        graph = Data(x=x, y=y)
        graph.graph_id = i  # optional, links to moment vector row
        graphs.append(graph)
    return graphs, features, labels

# ---------------------------
# Load TRAIN data
# ---------------------------
train_dataset, moment_features, labels = load_graph_dataset("data/train.csv")

# Convert moment features to tensor
moment_tensor = torch.tensor(moment_features, dtype=torch.float32).to(device)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# ---------------------------
# Model
# ---------------------------
num_classes = len(np.unique(labels))
moment_dim = moment_features.shape[1]

model = HybridClassifier(
    gcn_out_dim=64,
    moment_dim=moment_dim,
    num_classes=num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# ---------------------------
# Training
# ---------------------------
epochs = 20
for epoch in range(epochs):
    loss = train(model, train_loader, optimizer, criterion, moment_tensor, device)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss:.4f}")

# ---------------------------
# Train accuracy sanity check
# ---------------------------
acc, report = evaluate(model, train_loader, moment_tensor, device)
print(f"Train Accuracy (sanity check): {acc:.4f}")

