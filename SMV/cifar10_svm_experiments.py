"""
Πρόγραμμα (Python 3) για εκπαίδευση και αξιολόγηση SVM σε CIFAR-10 (pickled Python version)
Υποστηρίζει:
 - φόρτωση των data_batch_x και test_batch (pickled CIFAR-10 από: https://www.cs.toronto.edu/~kriz/cifar.html )
 - προεπεξεργασία (scaling, προαιρετικό PCA με διατήρηση >= 90% της διόρθωσης)
 - SVM (γραμμικό και RBF) με GridSearchCV
 - σύγκριση με k-NN (k=1,3), Nearest Class Centroid
 - MLP με 1 κρυφό επίπεδο, βελτιστοποίηση με multiclass hinge loss (PyTorch)
 - αναφορά χρόνου, accuracy train/test, confusion matrix, δείγματα σωστής/λανθασμένης ταξινόμησης

Προαπαιτούμενα:
  - python3
  - numpy, scipy, scikit-learn, matplotlib, Pillow
  - (προαιρετικά) torch για το MLP με hinge loss
"""

import os
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

# Προσπάθεια εισαγωγής PyTorch για MLP με hinge loss
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def unpickle(file):
    """Ανοίγει pickled CIFAR-10 αρχείο και επιστρέφει το dictionary με δεδομένα και labels."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(data_dir):
    """Φορτώνει CIFAR-10 από τα data_batch_1..5 και test_batch.
    Επιστρέφει X_train, y_train, X_test, y_test, label_names."""
    data_list = []
    labels_list = []
    # φόρτωση training batches
    for i in range(1,6):
        path = os.path.join(data_dir, f'data_batch_{i}')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Δεν βρέθηκε {path}. Βεβαιωθείτε ότι έχετε αποσυμπιέσει τα CIFAR-10 αρχεία.")
        batch = unpickle(path)
        data = batch[b'data'] if b'data' in batch else batch['data']
        labels = batch[b'labels'] if b'labels' in batch else batch['labels']
        data_list.append(data)
        labels_list.extend(labels)
    X_train = np.vstack(data_list).astype(np.float32)
    y_train = np.array(labels_list, dtype=np.int64)

    # φόρτωση test set
    test_path = os.path.join(data_dir, 'test_batch')
    test_batch = unpickle(test_path)
    X_test = test_batch[b'data'] if b'data' in test_batch else test_batch['data']
    y_test = np.array(test_batch[b'labels'] if b'labels' in test_batch else test_batch['labels'], dtype=np.int64)

    # φόρτωση label names
    meta = unpickle(os.path.join(data_dir, 'batches.meta'))
    label_names = meta[b'label_names'] if b'label_names' in meta else meta['label_names']
    label_names = [ln.decode('utf-8') if isinstance(ln, bytes) else ln for ln in label_names]

    return X_train, y_train, X_test.astype(np.float32), y_test, label_names


def reshape_images(X):
    """Μετατρέπει τα δεδομένα 3072-dim σε εικόνες 32x32x3 (Numpy array)."""
    N = X.shape[0]
    imgs = X.reshape(N,3,32,32).transpose(0,2,3,1).astype(np.uint8)
    return imgs


def save_sample_images(imgs, y_true, y_pred, label_names, out_dir, prefix, max_examples=8):
    """Αποθηκεύει παραδείγματα σωστής και λανθασμένης ταξινόμησης σε εικόνες PNG."""
    os.makedirs(out_dir, exist_ok=True)
    correct_idx = np.where(y_true==y_pred)[0]
    wrong_idx = np.where(y_true!=y_pred)[0]

    def save_some(idx_list, kind):
        sel = idx_list[:max_examples]
        for i, idx in enumerate(sel):
            im = Image.fromarray(imgs[idx])
            gt = label_names[y_true[idx]]
            pr = label_names[y_pred[idx]]
            fname = os.path.join(out_dir, f"{prefix}_{kind}_{i}_gt_{gt}_pred_{pr}.png")
            im.save(fname)

    save_some(correct_idx, 'correct')
    save_some(wrong_idx, 'wrong')


class SimpleMLP(nn.Module):
    """Απλό MLP με 1 κρυφό επίπεδο και ReLU ενεργοποίηση."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_mlp_hinge(X_train, y_train, X_val, y_val, hidden=256, epochs=20, batch_size=256, lr=1e-3, device='cpu'):
    """Εκπαίδευση MLP με multiclass hinge loss (MultiMarginLoss).
    Επιστρέφει το εκπαιδευμένο μοντέλο και ιστορικό εκπαίδευσης."""
    if not TORCH_AVAILABLE:
        print('\nPyTorch δεν είναι διαθέσιμο — παραλείπεται MLP με hinge loss.\n')
        return None

    device = torch.device(device)
    # Μετατροπή σε torch tensors
    Xtr = torch.from_numpy(X_train).float().to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val).long().to(device)

    model = SimpleMLP(X_train.shape[1], hidden, int(y_train.max()+1)).to(device)
    criterion = nn.MultiMarginLoss()  # hinge-like loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = 0.0
    best_state = None
    history = {'train_loss':[], 'val_acc':[]}

    n_batches = int(np.ceil(len(X_train)/batch_size))
    for epoch in range(1, epochs+1):
        model.train()
        perm = torch.randperm(len(Xtr))
        running_loss = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / len(Xtr)

        # Αξιολόγηση στο validation set
        model.eval()
        with torch.no_grad():
            outv = model(Xv)
            preds = outv.argmax(dim=1).cpu().numpy()
            val_acc = (preds==y_val).mean()
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f"MLP Epoch {epoch}/{epochs}  train_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main(args):
    """Κύρια ρουτίνα: φόρτωση δεδομένων, προεπεξεργασία, εκπαίδευση και αξιολόγηση μοντέλων."""
    X_train, y_train, X_test, y_test, label_names = load_cifar10(args.data_dir)
    print('Loaded CIFAR-10: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # --- Επιλογή binary προβλήματος (προαιρετικό) ---
    if args.binary is not None:
        # Δύο περιπτώσεις: one-vs-rest (π.χ. "3") ή two-class ("2,5")
        if ',' in args.binary:
            a,b = [int(x) for x in args.binary.split(',')]
            sel_train = np.where((y_train==a)|(y_train==b))[0]
            sel_test = np.where((y_test==a)|(y_test==b))[0]
            y_train_sel = y_train[sel_train]
            y_test_sel = y_test[sel_test]
            y_train_sel = (y_train_sel==b).astype(int)
            y_test_sel = (y_test_sel==b).astype(int)
            X_train = X_train[sel_train]
            y_train = y_train_sel
            X_test = X_test[sel_test]
            y_test = y_test_sel
            print(f"Binary two-class problem: {a} vs {b}   #train={len(y_train)} #test={len(y_test)}")
        else:
            c = int(args.binary)
            y_train = (y_train==c).astype(int)
            y_test = (y_test==c).astype(int)
            print(f"Binary one-vs-rest problem: class {c} ('{label_names[c]}') vs rest")

    # --- Standardization ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- PCA (προαιρετικό) ---
    pca = None
    if args.pca is not None and args.pca>0 and args.pca<1.0:
        t0 = time.time()
        pca = PCA(n_components=args.pca, svd_solver='full')
        X_train_p = pca.fit_transform(X_train_scaled)
        X_test_p = pca.transform(X_test_scaled)
        print(f"PCA done. kept components={pca.n_components_}  explained_variance_ratio_.sum()={pca.explained_variance_ratio_.sum():.4f}  time={(time.time()-t0):.2f}s")
    else:
        X_train_p = X_train_scaled
        X_test_p = X_test_scaled

    # --- Υποδειγματοληψία για γρήγορο grid search ---
    if args.subsample is not None and args.subsample>0 and args.subsample < 1.0:
        rng = np.random.RandomState(args.random_seed)
        sel_tr = rng.choice(len(X_train_p), size=int(len(X_train_p)*args.subsample), replace=False)
        X_sub = X_train_p[sel_tr]
        y_sub = y_train[sel_tr]
        print(f"Subsampled training set to {X_sub.shape[0]} examples for fast grid search")
    else:
        X_sub = X_train_p
        y_sub = y_train

    results = {}

    # --- Linear SVM ---
    print('\n=== Linear SVM (LinearSVC) grid search ===')
    param_grid = {'C':[0.01, 0.1, 1.0, 10.0]}
    svc_lin = LinearSVC(max_iter=20000)
    g_lin = GridSearchCV(svc_lin, param_grid, cv=3, verbose=1, n_jobs=args.n_jobs)
    t0 = time.time()
    g_lin.fit(X_sub, y_sub)
    t_lin = time.time()-t0
    best_lin = g_lin.best_estimator_
    y_pred_train = best_lin.predict(X_train_p)
    y_pred_test = best_lin.predict(X_test_p)
    results['linear_svm'] = {
        'train_acc':accuracy_score(y_train, y_pred_train),
        'test_acc':accuracy_score(y_test, y_pred_test),
        'time':t_lin,
        'model':best_lin
    }

    # --- RBF SVM ---
    print('\n=== RBF SVM grid search (may be slow) ===')
    param_grid_rbf = {'C':[0.1,1,10],'gamma':['scale','auto']}
    svc_rbf = SVC(kernel='rbf')
    g_rbf = GridSearchCV(svc_rbf, param_grid_rbf, cv=3, verbose=1, n_jobs=args.n_jobs)
    t0 = time.time()
    g_rbf.fit(X_sub, y_sub)
    t_rbf = time.time()-t0
    best_rbf = g_rbf.best_estimator_
    y_pred_train = best_rbf.predict(X_train_p)
    y_pred_test = best_rbf.predict(X_test_p)
    results['rbf_svm'] = {
        'train_acc':accuracy_score(y_train, y_pred_train),
        'test_acc':accuracy_score(y_test, y_pred_test),
        'time':t_rbf,
        'model':best_rbf
    }

    # --- k-NN ---
    print('\n=== k-NN (k=1 and k=3) ===')
    for k in [1,3]:
        t0 = time.time()
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=args.n_jobs)
        knn.fit(X_sub, y_sub)
        time_fit = time.time()-t0
        y_pred = knn.predict(X_test_p)
        acc = accuracy_score(y_test, y_pred)
        results[f'knn_{k}'] = {'test_acc':acc,'time':time_fit,'model':knn}

    # --- Nearest Centroid ---
    print('\n=== Nearest Class Centroid ===')
    nc = NearestCentroid()
    t0 = time.time()
    nc.fit(X_sub, y_sub)
    t_nc = time.time()-t0
    y_pred = nc.predict(X_test_p)
    acc_nc = accuracy_score(y_test, y_pred)
    results['nearest_centroid'] = {'test_acc':acc_nc,'time':t_nc,'model':nc}

    # --- MLP με hinge loss ---
    print('\n=== MLP (1 hidden layer) with multiclass hinge loss (if PyTorch available) ===')
    mlp_model = None
    if TORCH_AVAILABLE:
        X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
            X_sub, y_sub, test_size=0.1, random_state=args.random_seed,
            stratify=y_sub if len(np.unique(y_sub))>1 else None)
        t0 = time.time()
        mlp_model, mlp_history = train_mlp_hinge(
            X_tr_sub, y_tr_sub, X_val_sub, y_val_sub,
            hidden=args.mlp_hidden, epochs=args.mlp_epochs,
            batch_size=args.mlp_batch, lr=args.mlp_lr,
            device=args.mlp_device)
        t_mlp = time.time()-t0
        if mlp_model is not None:
            mlp_model.eval()
            with torch.no_grad():
                Xtest_t = torch.from_numpy(X_test_p).float().to(args.mlp_device)
                out = mlp_model(Xtest_t).cpu().numpy()
                y_pred = out.argmax(axis=1)
            acc_mlp = accuracy_score(y_test, y_pred)
            results['mlp_hinge'] = {'test_acc':acc_mlp,'time':t_mlp,'model':mlp_model,'history':mlp_history}

    # --- Επιλογή καλύτερου μοντέλου ---
    best_key = max([k for k in results.keys()], key=lambda k: results[k].get('test_acc',0.0))
    best_model = results[best_key]['model']
    print(f"\nBest model by test accuracy: {best_key}  test_acc={results[best_key].get('test_acc',0.0):.4f}")

    # προβλέψεις με το καλύτερο μοντέλο
    if best_key=='mlp_hinge' and TORCH_AVAILABLE:
        with torch.no_grad():
            Xtest_t = torch.from_numpy(X_test_p).float().to(args.mlp_device)
            out = best_model(Xtest_t).cpu().numpy()
            y_best = out.argmax(axis=1)
    else:
        y_best = best_model.predict(X_test_p)

    imgs = reshape_images((scaler.inverse_transform(X_test_p) if pca is None else (pca.inverse_transform(X_test_p) if pca is not None else X_test_p)).astype(np.uint8)) if False else reshape_images((X_test).astype(np.uint8))
    # Η παραπάνω γραμμή προσπαθεί να ανακατασκευάσει εικόνες: original X_test
    save_sample_images(reshape_images(X_test.astype(np.uint8)), y_test, y_best, label_names, out_dir=args.output_dir, prefix=best_key)

    # Εκτύπωση summary πίνακα
    print('\n=== Summary of results ===')
    for k,v in results.items():
        print(f"{k:20s}  test_acc={v.get('test_acc',v.get('test_acc',0.0)):.4f}  train_acc={v.get('train_acc',float('nan')) if 'train_acc' in v else float('nan'):.4f}  time={v.get('time',0.0):.2f}s")

    # Confusion matrix και αναφορά ταξινόμησης για το καλύτερο μοντέλο
    print('\nClassification report for best model:')
    print(classification_report(y_test, y_best, target_names=label_names if len(label_names)==len(np.unique(y_test)) else None))
    cm = confusion_matrix(y_test, y_best)
    print('Confusion matrix:\n', cm)

    # Αποθήκευση εικόνας confusion matrix
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion matrix ({best_key})')
    plt.colorbar()
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, f'confusion_{best_key}.png')
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(cm_path)
    print('Saved confusion matrix to', cm_path)

    # Αποθηκεύει τα αποτελέσματα στο αρχείο JSON
    import json
    serial = {k:{'test_acc':float(v.get('test_acc',0.0)),'train_acc':float(v.get('train_acc',0.0)),'time':float(v.get('time',0.0))} for k,v in results.items()}
    with open(os.path.join(args.output_dir,'results_summary.json'),'w') as fo:
        json.dump(serial, fo, indent=2)
    print('Saved summary JSON to', os.path.join(args.output_dir,'results_summary.json'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 SVM experiments')
    parser.add_argument('--data_dir', required=True, help='folder with CIFAR-10 pickled files')
    parser.add_argument('--pca', type=float, default=0.90, help='PCA variance to keep (0..1), set 0 or 1 to disable')
    parser.add_argument('--binary', type=str, default=None, help='set to class idx for one-vs-rest, or "a,b" for two-class between a and b; default None => multi-class')
    parser.add_argument('--subsample', type=float, default=0.2, help='subsample fraction of training data to use for grid-search to save time (0..1)')
    parser.add_argument('--n_jobs', type=int, default=4, help='n_jobs for sklearn parallelism')
    parser.add_argument('--output_dir', type=str, default='cifar_results', help='where to save outputs')
    parser.add_argument('--random_seed', type=int, default=123)
    # MLP hyperparams
    parser.add_argument('--mlp_hidden', type=int, default=512)
    parser.add_argument('--mlp_epochs', type=int, default=20)
    parser.add_argument('--mlp_batch', type=int, default=256)
    parser.add_argument('--mlp_lr', type=float, default=1e-3)
    parser.add_argument('--mlp_device', type=str, default='cpu')

    args = parser.parse_args()
    main(args)
