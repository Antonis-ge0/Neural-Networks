import os
import time
import pickle
import argparse
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in data.items()}

def load_cifar10(data_dir):
    """Φορτώνει όλα τα data_batch_* και test_batch, επιστρέφει X_train, y_train, X_test, y_test, label_names"""
    X_train = []
    y_train = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        X_train.append(batch['data'])
        y_train += batch['labels']
    X_train = np.vstack(X_train).astype(np.uint8)
    y_train = np.array(y_train, dtype=np.int64)
    test = unpickle(os.path.join(data_dir, 'test_batch'))
    X_test = test['data'].astype(np.uint8)
    y_test = np.array(test['labels'], dtype=np.int64)
    meta = unpickle(os.path.join(data_dir, 'batches.meta'))
    label_names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in meta['label_names']]
    # reshape to (N,3,32,32)
    def reshape_raw(X):
        N = X.shape[0]
        Xr = X.reshape(N, 3, 32, 32).astype(np.float32) / 255.0
        return Xr
    return reshape_raw(X_train), y_train, reshape_raw(X_test), y_test, label_names

class CIFARDataset(Dataset):
    """PyTorch Dataset wrapper για CIFAR-10 δεδομένα σε μορφή (N,3,32,32)."""
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        img = self.X[idx]  # float 0..1
        # μετατρέπουμε σε PIL για πιθανές μετασχηματίσεις
        img_pil = Image.fromarray((img.transpose(1,2,0)*255).astype(np.uint8))
        if self.transform:
            img_pil = self.transform(img_pil)
        # επιστρέφουμε tensor HWC -> CHW
        img_arr = np.array(img_pil).astype(np.float32) / 255.0
        img_chw = img_arr.transpose(2,0,1)
        return torch.tensor(img_chw, dtype=torch.float32), int(self.y[idx])

class SimpleCNN_MLP(nn.Module):
    """
    Απλό CNN και στο τέλος ένα MLP. Το μεγέθος του MLP (hidden neurons) είναι παραμετρικό.
    - conv_layers: προεπιλογή 3 στρώματα με pool
    - mlp_hidden: αριθμός νευρώνων στο κρυφό πλήρως συνδεδεμένο επίπεδο
    """
    def __init__(self, mlp_hidden=256, num_classes=10, dropout=0.5):
        super().__init__()
        # Συνελικτικό μέρος
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),  # 16x16

            ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2)),  # 8x8

            ('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool2d(2)),  # 4x4
        ]))
        # Τέλος: flatten και MLP
        flat_dim = 128 * 4 * 4
        self.fc1 = nn.Linear(flat_dim, mlp_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_hidden, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

def eval_model(model, dataloader, criterion, device, save_examples=False, out_dir='results'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    misclassified = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            if save_examples:
                # αποθήκευση μερικών λανθασμένων / σωστών
                for i in range(inputs.size(0)):
                    if preds[i] != targets[i] and len(misclassified) < 50:
                        img = (inputs[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
                        misclassified.append( (img, int(preds[i].cpu().item()), int(targets[i].cpu().item())) )
    avg_loss = running_loss / total
    acc = correct / total
    if save_examples and len(misclassified) > 0:
        ex_dir = os.path.join(out_dir, 'misclassified')
        os.makedirs(ex_dir, exist_ok=True)
        for idx, (img, pred, targ) in enumerate(misclassified):
            im = Image.fromarray(img)
            im.save(os.path.join(ex_dir, f'mis_{idx}_pred{pred}_true{targ}.png'))
    return avg_loss, acc, np.concatenate(all_preds), np.concatenate(all_targets)

def flatten_dataset(X):
    """Μετατρέπει (N,3,32,32) -> (N,3072) float."""
    N = X.shape[0]
    return X.transpose(0,2,3,1).reshape(N, -1).astype(np.float32)

def run_classical(X_train, y_train, X_test, y_test, method='knn', pca_dim=None, k=3):
    """
    Εκτελεί kNN ή Nearest Centroid.
    Αν δοθεί pca_dim, εφαρμόζει PCA στα flattened pixels και τρέχει εκεί.
    Επιστρέφει accuracy.
    """
    Xtr = flatten_dataset(X_train)
    Xte = flatten_dataset(X_test)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    if pca_dim is not None:
        pca = PCA(n_components=pca_dim, svd_solver='randomized', random_state=0)
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)
    else:
        Xtr_p = Xtr_s
        Xte_p = Xte_s

    if method == 'knn':
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    else:
        clf = NearestCentroid()
    t0 = time.time()
    clf.fit(Xtr_p, y_train)
    preds = clf.predict(Xte_p)
    t1 = time.time()
    acc = accuracy_score(y_test, preds)
    return acc, t1 - t0

def run_experiment(data_dir, results_dir, device, epochs=20, batch_size=128,
                   lr=1e-3, mlp_hidden_list=[128,256], dropout=0.5, pclog=False):
    os.makedirs(results_dir, exist_ok=True)
    # Φόρτωση δεδομένων
    X_train, y_train, X_test, y_test, label_names = load_cifar10(data_dir)
    print("Φορτώθηκαν δεδομένα:", X_train.shape, X_test.shape)
    train_ds = CIFARDataset(X_train, y_train)
    test_ds = CIFARDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    results = []

    for mlp_hidden in mlp_hidden_list:
        print(f"\n=== Πείραμα: mlp_hidden = {mlp_hidden}, lr={lr}, batch={batch_size}, epochs={epochs} ===")
        model = SimpleCNN_MLP(mlp_hidden=mlp_hidden, dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        train_acc_history = []
        test_acc_history = []
        train_loss_history = []
        test_loss_history = []

        t_start = time.time()
        for epoch in range(1, epochs+1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc, _, _ = eval_model(model, test_loader, criterion, device, save_examples=False)
            scheduler.step()
            t1 = time.time()
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  epoch_time={t1-t0:.1f}s")
        t_end = time.time()
        total_time = t_end - t_start

        # Τελική αξιολόγηση και αποθήκευση μερικών παραδειγμάτων
        test_loss, test_acc, preds, trues = eval_model(model, test_loader, criterion, device, save_examples=True, out_dir=results_dir)
        print(f"Τελικό test accuracy: {test_acc:.4f}, συνολικό training time: {total_time:.1f}s")

        # Αποθήκευση ιστορικού και μετρικών
        exp_res = {
            'mlp_hidden': mlp_hidden,
            'lr': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'train_acc_history': train_acc_history,
            'test_acc_history': test_acc_history,
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history,
            'test_acc_final': test_acc,
            'train_time_s': total_time
        }
        results.append(exp_res)

        # Σχεδίαση accuracy over epochs
        plt.figure(figsize=(6,4))
        plt.plot(train_acc_history, label='train_acc')
        plt.plot(test_acc_history, label='test_acc')
        plt.title(f'Accuracy (mlp_hidden={mlp_hidden})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'accuracy_hidden{mlp_hidden}.png'))
        plt.close()

    # Κλασσικοί ταξινομητές: kNN και Nearest Centroid (PCA 100)
    print("\n=== Κλασσικοί ταξινομητές (kNN, Nearest Centroid) ===")
    knn_acc, knn_time = run_classical(X_train, y_train, X_test, y_test, method='knn', pca_dim=100, k=3)
    nc_acc, nc_time = run_classical(X_train, y_train, X_test, y_test, method='centroid', pca_dim=100)
    print(f"kNN (pca100) acc={knn_acc:.4f} time={knn_time:.2f}s")
    print(f"Nearest Centroid (pca100) acc={nc_acc:.4f} time={nc_time:.2f}s")

    # Αποθήκευση συνοψίσεων
    import csv
    csv_path = os.path.join(results_dir, 'summary_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mlp_hidden','lr','batch_size','epochs','train_time_s','test_acc_final'])
        for r in results:
            writer.writerow([r['mlp_hidden'], r['lr'], r['batch_size'], r['epochs'], r['train_time_s'], r['test_acc_final']])
        writer.writerow(['kNN_pca100', '', '', '', knn_time, knn_acc])
        writer.writerow(['NearestCentroid_pca100', '', '', '', nc_time, nc_acc])
    print("Αποθηκεύτηκε το summary σε", csv_path)
    return results, (knn_acc, nc_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Φάκελος με τα CIFAR pickles')
    parser.add_argument('--results_dir', type=str, default='results', help='Φάκελος αποθήκευσης αποτελεσμάτων')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mlp_hidden', type=int, nargs='+', default=[128, 256], help='λίστα τιμών mlp hidden για sweep')
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    # Επιλογή συσκευής
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    run_experiment(args.data_dir, args.results_dir, device,
                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,

                   mlp_hidden_list=args.mlp_hidden, dropout=args.dropout)
