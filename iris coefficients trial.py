import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch

# Load and preprocess CIFAR-10 data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Convert to numpy arrays and flatten images
X_train = []
Y_train = []
for img, label in trainset:
    # Flatten 3x32x32 to 3072 features
    X_train.append(img.numpy().flatten())
    Y_train.append(label)

X_test = []
Y_test = []
for img, label in testset:
    X_test.append(img.numpy().flatten())
    Y_test.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train).reshape(-1, 1)
X_test = np.array(X_test)
Y_test = np.array(Y_test).reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)

# Use subset for faster training
subset_size = 5000
X_train = X_train[:subset_size]
Y_train = Y_train[:subset_size]

print(f"Training samples: {X_train.shape[0]}")
print(f"Input features: {X_train.shape[1]}")
print(f"Output classes: {Y_train.shape[1]}")

# Initialize polynomial coefficients: (outputs=10, inputs=3072, degrees=4)
# This is large, so we'll use a reduced approach
num_outputs = 10
num_inputs = 128  # Reduced from 3072 for computational efficiency
degree = 4

# Use PCA-like reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=num_inputs)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

coeffs = np.random.randn(num_outputs, num_inputs, degree) * 0.01  # small random start

lr = 0.001  # learning rate
epochs = 50  # Reduced for CIFAR-10
batch_size = 256
loss_plot = []
acc_plot = []

for epoch in range(epochs):
    # Mini-batch training
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    epoch_losses = []
    
    for batch_start in range(0, X_train.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, X_train.shape[0])
        batch_indices = indices[batch_start:batch_end]
        X_batch = X_train[batch_indices]
        Y_batch = Y_train[batch_indices]
        
        # Forward pass: polynomial photonic layer
        O = np.zeros((X_batch.shape[0], num_outputs))
        for o in range(num_outputs):
            for i in range(num_inputs):
                for k, a in enumerate(coeffs[o, i]):
                    O[:, o] += a * (X_batch[:, i] ** k)
        
        # Softmax output
        expO = np.exp(O - O.max(axis=1, keepdims=True))
        probs = expO / expO.sum(axis=1, keepdims=True)
        loss = -np.mean(np.sum(Y_batch * np.log(probs + 1e-12), axis=1))
        epoch_losses.append(loss)

        # Compute gradients (simple approx)
        grad_output = probs - Y_batch
        for o in range(num_outputs):
            for i in range(num_inputs):
                for k in range(degree):
                    grad = np.mean(grad_output[:, o] * (X_batch[:, i] ** k))
                    coeffs[o, i, k] -= lr * grad
    
    avg_loss = np.mean(epoch_losses)
    loss_plot.append(avg_loss)
    
    # Calculate training accuracy
    O_train = np.zeros((X_train.shape[0], num_outputs))
    for o in range(num_outputs):
        for i in range(num_inputs):
            for k, a in enumerate(coeffs[o, i]):
                O_train[:, o] += a * (X_train[:, i] ** k)
    y_pred_train = np.argmax(O_train, axis=1)
    train_acc = accuracy_score(np.argmax(Y_train, axis=1), y_pred_train)
    acc_plot.append(train_acc)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}, train_acc={train_acc*100:.2f}%")

# Plot training metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(loss_plot)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True)

ax2.plot(acc_plot)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy')
ax2.grid(True)
plt.tight_layout()
plt.show()

# Evaluate on test set
O_test = np.zeros((X_test.shape[0], num_outputs))
for o in range(num_outputs):
    for i in range(num_inputs):
        for k, a in enumerate(coeffs[o, i]):
            O_test[:, o] += a * (X_test[:, i] ** k)
y_pred = np.argmax(O_test, axis=1)
acc = accuracy_score(np.argmax(Y_test, axis=1), y_pred)
print(f"\nTest accuracy: {acc*100:.2f}%")
print(f"Coefficient matrix shape: {coeffs.shape}")

# Save trained coefficients
save_path = "./trained_coeffs_cifar10.npy"
np.save(save_path, coeffs)
print(f"✅ Trained coefficients saved to {save_path}")

# Also save PCA transformer for later use
import pickle
with open("./pca_transformer_cifar10.pkl", "wb") as f:
    pickle.dump(pca, f)
print("✅ PCA transformer saved")