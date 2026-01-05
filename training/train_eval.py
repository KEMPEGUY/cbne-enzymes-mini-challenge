# training/train_eval.py
import torch
from sklearn.metrics import accuracy_score, classification_report

def train(model, loader, optimizer, criterion, moment_tensor, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data, moment_tensor)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, moment_tensor, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data, moment_tensor)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return accuracy_score(y_true, y_pred), classification_report(y_true, y_pred, output_dict=False)

