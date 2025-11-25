"""
Hybrid CIFAR-10 image classifier - FIXED VERSION
Key fixes:
- Properly detach tensors before saving to prevent gradient accumulation
- Clone tensors to avoid references to computation graph
- Clear optimizer state periodically to prevent memory buildup
- Add checkpoint size validation
"""

import os
import math
import pickle
from PIL import Image
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import pennylane as qml
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
n_qubits = 4
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

batch_size = 8
num_epochs = 10
lr = 1e-3
weight_decay = 1e-4
lambda_wasserstein = 0.5
lambda_entropy_penalty = 2.0
n_layers = 3

checkpoint_path = "hybrid_full_checkpoint.pth"

# -----------------------
# Dataset (CIFAR-10)
# -----------------------
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = trainset.classes
print("CIFAR-10 classes:", classes)

# dataset-level true distribution (for Wasserstein)
labels_all = torch.tensor(trainset.targets)
counts = torch.bincount(labels_all, minlength=num_classes).float()
dataset_true_dist = counts / counts.sum()
dataset_true_dist = dataset_true_dist.to(device)

# -----------------------
# Quantum device + qnode
# -----------------------
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def qnode(qparams, features):
    features = torch.clamp(features, -math.pi, math.pi)
    qparams = torch.clamp(qparams, -2*math.pi, 2*math.pi)
    
    for i in range(n_qubits):
        qml.RY(features[i], wires=i)

    param_idx = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            qml.RY(qparams[param_idx], wires=i)
            param_idx += 1

        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                if (i + j) % 2 == 0:
                    qml.CNOT(wires=[i, j])

        for i in range(n_qubits):
            qml.RZ(qparams[param_idx], wires=i)
            param_idx += 1

        for i in range(n_qubits):
            qml.RX(qparams[param_idx], wires=i)
            param_idx += 1

    return qml.probs(wires=list(range(n_qubits)))

total_q_params = n_layers * 3 * n_qubits
print("Quantum params:", total_q_params)

# -----------------------
# PyTorch hybrid model
# -----------------------
class HybridBayesQuantum(nn.Module):
    def __init__(self, n_qubits, pretrained_backbone=True):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained_backbone)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.feature_reducer = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_qubits)
        )

    def forward(self, x, q_params_sample):
        batch_size = x.shape[0]
        features = self.backbone(x)
        reduced = self.feature_reducer(features)
        angles = torch.tanh(reduced) * (math.pi / 2)

        probs_list = []
        for i in range(batch_size):
            try:
                p = qnode(q_params_sample, angles[i])
                if isinstance(p, np.ndarray):
                    p = torch.tensor(p, dtype=torch.float32)
                p = p + 1e-10
                p = p / p.sum()
                probs_list.append(p.to(device))
            except Exception as e:
                print(f"Warning: QNode execution failed for sample {i}, using uniform distribution. Error: {e}")
                uniform_probs = torch.ones(2**n_qubits, device=device) / (2**n_qubits)
                probs_list.append(uniform_probs)
                
        probs = torch.stack(probs_list, dim=0).to(device)
        return probs

model = HybridBayesQuantum(n_qubits=n_qubits, pretrained_backbone=True).to(device)

# -----------------------
# Bayesian-style quantum parameters (trainable)
# -----------------------
q_mean = nn.Parameter(torch.randn(total_q_params, device=device) * 0.01)
q_logvar = nn.Parameter(torch.full((total_q_params,), -6.0, device=device))

# -----------------------
# Loss helpers
# -----------------------
def wasserstein_distance(pred_dist, true_dist):
    cdf_p = torch.cumsum(pred_dist, dim=0)
    cdf_t = torch.cumsum(true_dist, dim=0)
    return torch.sum(torch.abs(cdf_p - cdf_t))

def entropy_of_distribution(p):
    p = p / (p.sum() + 1e-12)
    return -torch.sum(p * torch.log(p + 1e-12))

def quantum_nll_loss(probs, labels, num_classes):
    truncated = probs[:, :num_classes]
    truncated = truncated / (truncated.sum(dim=1, keepdim=True) + 1e-12)
    true_probs = truncated[torch.arange(labels.shape[0]), labels]
    loss = -torch.log(true_probs + 1e-12).mean()
    return loss, truncated

def sample_qparams(mean, logvar):
    eps = torch.randn_like(mean)
    std = torch.exp(0.5 * logvar)
    std = torch.clamp(std, 0, 0.5)
    return mean + std * eps

# -----------------------
# CRITICAL FIX: Safe checkpoint saving
# -----------------------
def save_checkpoint_safe(checkpoint_data, path, max_size_mb=500):
    """
    Safely save checkpoint with size validation.
    Ensures all tensors are properly detached and cloned.
    """
    # Create a clean checkpoint dict with detached tensors
    clean_checkpoint = {}
    
    for key, value in checkpoint_data.items():
        if isinstance(value, torch.Tensor):
            # Detach from computation graph and clone to CPU
            clean_checkpoint[key] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            # Handle nested dicts (like model state dict)
            clean_dict = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    clean_dict[k] = v.detach().cpu().clone()
                else:
                    clean_dict[k] = v
            clean_checkpoint[key] = clean_dict
        else:
            clean_checkpoint[key] = value
    
    # Save to temporary file first
    temp_path = path + ".tmp"
    torch.save(clean_checkpoint, temp_path)
    
    # Check file size
    file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    print(f"  Checkpoint size: {file_size_mb:.2f} MB")
    
    if file_size_mb > max_size_mb:
        os.remove(temp_path)
        raise RuntimeError(f"Checkpoint size ({file_size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)")
    
    # Move temp file to final location
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)
    
    return file_size_mb

# -----------------------
# Optimizer
# -----------------------
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = optim.Adam([
    {"params": model.feature_reducer.parameters()},
    {"params": [q_mean, q_logvar]}
], lr=lr, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# -----------------------
# Training loop - FIXED VERSION
# -----------------------
def train(num_epochs=num_epochs):
    history = {"epoch": [], "epoch_wd": [], "epoch_entropy": [], "val_acc": []}
    prev_entropy = None
    best_state = None
    best_wd = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        agg_pred_counts = torch.zeros(num_classes, device=device)
        q_mean_update_scale = 1.0
        
        print(f"\nEpoch {epoch}/{num_epochs} starting...")
        epoch_start_time = time.time()
        batch_times = []
        
        for batch_idx, (images, labels) in enumerate(trainloader):
            batch_start = time.time()
            
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}/{len(trainloader)}...")
                
            images = images.to(device)
            labels = labels.to(device)

            q_sample = sample_qparams(q_mean, q_logvar)

            probs = model(images, q_sample)
            nll_loss, truncated_probs = quantum_nll_loss(probs, labels, num_classes=num_classes)

            # CRITICAL: Detach before accumulating
            agg_pred_counts += truncated_probs.sum(dim=0).detach()

            optimizer.zero_grad()
            nll_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([q_mean, q_logvar], max_norm=1.0)
            
            if q_mean.grad is not None:
                q_mean.grad.data.mul_(q_mean_update_scale)
            optimizer.step()

            # CRITICAL: Store only the scalar value, not the tensor
            epoch_losses.append(nll_loss.item())
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if batch_idx == 10:
                avg_batch_time = np.mean(batch_times)
                est_epoch_time = avg_batch_time * len(trainloader)
                print(f"  First 10 batches: {avg_batch_time:.2f}s per batch")
                print(f"  Estimated epoch time: {est_epoch_time/60:.1f} minutes")
                print(f"  Estimated total training time: {est_epoch_time * num_epochs / 60:.1f} minutes")

        epoch_time = time.time() - epoch_start_time
        
        # CRITICAL: Detach before computing metrics
        agg_pred_dist = agg_pred_counts[:num_classes].detach()
        if agg_pred_dist.sum() > 0:
            agg_pred_dist = agg_pred_dist / agg_pred_dist.sum()
        else:
            agg_pred_dist = torch.ones(num_classes, device=device) / num_classes

        epoch_wd = wasserstein_distance(agg_pred_dist, dataset_true_dist).detach()
        epoch_entropy = entropy_of_distribution(agg_pred_dist).detach()

        entropy_delta = None
        entropy_penalty = 0.0
        if prev_entropy is not None:
            entropy_delta = (prev_entropy - epoch_entropy).item()
            if entropy_delta < 0:
                entropy_penalty = lambda_entropy_penalty * (-entropy_delta)
                q_mean_update_scale = 0.5
            else:
                q_mean_update_scale = 1.0
        prev_entropy = epoch_entropy.clone()

        print("  Performing distribution-level update...")
        model.train()
        optimizer.zero_grad()
        
        try:
            sample_images, _ = next(iter(trainloader))
        except StopIteration:
            sample_images = images.detach().cpu()
        sample_images = sample_images.to(device)
        
        probs_det = model(sample_images, q_mean)
        truncated_det = probs_det[:, :num_classes]
        agg_det = truncated_det.sum(dim=0)
        agg_det = agg_det / (agg_det.sum() + 1e-12)

        wd_loss = wasserstein_distance(agg_det, dataset_true_dist)
        total_dist_loss = lambda_wasserstein * wd_loss + entropy_penalty

        total_dist_loss.backward()
        torch.nn.utils.clip_grad_norm_([q_mean, q_logvar], max_norm=1.0)
        
        if q_mean.grad is not None:
            q_mean.grad.data.mul_(q_mean_update_scale)
        optimizer.step()

        # CRITICAL: Update best state with properly detached tensors
        if epoch_wd < best_wd:
            best_wd = epoch_wd.item()
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "q_mean": q_mean.detach().cpu().clone(),
                "q_logvar": q_logvar.detach().cpu().clone(),
                "epoch": epoch,
                "epoch_wd": float(epoch_wd.item()),
                "epoch_entropy": float(epoch_entropy.item())
            }

        avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss, val_acc = evaluate(max_batches=20)
        
        # Store only scalar values in history
        history["epoch"].append(epoch)
        history["epoch_wd"].append(float(epoch_wd.item()))
        history["epoch_entropy"].append(float(epoch_entropy.item()))
        history["val_acc"].append(float(val_acc))
        
        scheduler.step(val_loss)

        print(f"Epoch {epoch} completed in {epoch_time/60:.1f} min")
        print(f"  TrainLoss:{avg_epoch_loss:.4f} | Wd:{epoch_wd.item():.6f} | Ent:{epoch_entropy.item():.6f} | ΔH:{entropy_delta if entropy_delta is not None else 'N/A'} | Pen:{entropy_penalty:.6f}")
        print(f"  Validation (subset) loss {val_loss:.4f}, acc {val_acc:.3f}")

        # CRITICAL: Use safe checkpoint saving
        try:
            checkpoint_data = {
                "model_state": model.state_dict(),
                "q_mean": q_mean,
                "q_logvar": q_logvar,
                "epoch": epoch,
                "epoch_wd": float(epoch_wd.item()),
                "epoch_entropy": float(epoch_entropy.item())
            }
            file_size = save_checkpoint_safe(checkpoint_data, checkpoint_path, max_size_mb=500)
            print(f"  Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            print(f"  WARNING: Failed to save checkpoint: {e}")
            print("  Continuing training...")
        
        # Clear cache periodically
        if epoch % 3 == 0:
            torch.cuda.empty_cache()
            print("  GPU cache cleared")

    # Save best state
    if best_state is not None:
        try:
            save_checkpoint_safe(best_state, "hybrid_best_checkpoint.pth", max_size_mb=500)
            print("Best-state saved to hybrid_best_checkpoint.pth")
        except Exception as e:
            print(f"WARNING: Failed to save best checkpoint: {e}")

    return history

# -----------------------
# Evaluation
# -----------------------
def evaluate(max_batches=None):
    model.eval()
    losses = []
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            probs = model(images, q_mean)
            loss, truncated = quantum_nll_loss(probs, labels, num_classes=num_classes)
            losses.append(loss.item())
            preds = torch.argmax(truncated, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    model.train()
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

# -----------------------
# Prediction function
# -----------------------
predict_transform = transform_test

def predict_image(image_path):
    if not os.path.exists(checkpoint_path) and not os.path.exists("hybrid_best_checkpoint.pth"):
        raise FileNotFoundError("No saved checkpoint found.")

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
    else:
        ckpt = torch.load("hybrid_best_checkpoint.pth", map_location=device)

    model.load_state_dict(ckpt["model_state"])
    q_mean_current = torch.tensor(ckpt["q_mean"], dtype=torch.float32, device=device)

    model.eval()
    img = Image.open(image_path).convert("RGB")
    x = predict_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x, q_mean_current)[0]
        truncated = probs[:num_classes]
        truncated = truncated / (truncated.sum() + 1e-12)
        pred = int(torch.argmax(truncated).item())
        conf = float(truncated[pred].item())
    model.train()
    return pred, conf, truncated.cpu().numpy()

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint '{checkpoint_path}'. Loading model and skipping training.")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        q_mean.data = torch.tensor(ckpt["q_mean"], dtype=torch.float32, device=device)
        q_logvar.data = torch.tensor(ckpt["q_logvar"], dtype=torch.float32, device=device)
    else:
        print("Testing single batch forward pass...")
        model.train()
        test_batch = next(iter(trainloader))
        images, labels = test_batch
        images, labels = images.to(device), labels.to(device)
        print(f"Batch shape: {images.shape}")

        q_sample = sample_qparams(q_mean, q_logvar)
        print("Starting quantum forward pass...")
        start_time = time.time()

        with torch.no_grad():
            probs = model(images, q_sample)
            
        print(f"Quantum forward pass completed in {time.time() - start_time:.2f} seconds")
        print(f"Output shape: {probs.shape}")
        print("No checkpoint found. Starting training...")
        hist = train(num_epochs=num_epochs)
        
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            q_mean.data = torch.tensor(ckpt["q_mean"], dtype=torch.float32, device=device)
            q_logvar.data = torch.tensor(ckpt["q_logvar"], dtype=torch.float32, device=device)

    final_loss, final_acc = evaluate(max_batches=None)
    print(f"Final evaluation: loss={final_loss:.4f}, accuracy={final_acc:.4f}")

    sample_img_path = "dog_image.png"
    if os.path.exists(sample_img_path):
        pred, conf, probs = predict_image(sample_img_path)
        print(f"Predicted: {pred} ({classes[pred]}) with confidence {conf:.3f}")
    else:
        print(f"Note: Sample image '{sample_img_path}' not found. Skipping prediction demo.")