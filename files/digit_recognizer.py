"""
Handwritten Digit Recognizer (0-9)
===================================
A simple neural network trained on the MNIST dataset with a tkinter
drawing canvas for live prediction.

Usage:
    python digit_recognizer.py

First run will download MNIST and train the model (~1-2 minutes).
Subsequent runs load the saved model instantly.

To retrain, delete digit_model.npz and run again.
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# 1. Neural Network (pure NumPy)
# ---------------------------------------------------------------------------
# Architecture: 784 -> 256 (ReLU) -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def one_hot(labels, num_classes=10):
    oh = np.zeros((labels.size, num_classes))
    oh[np.arange(labels.size), labels] = 1
    return oh


class SimpleNeuralNetwork:
    """A 4-layer fully-connected network for digit classification."""

    def __init__(self):
        # He initialization
        self.W1 = np.random.randn(784, 256) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros((1, 256))
        self.W2 = np.random.randn(256, 128) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros((1, 128))
        self.W3 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros((1, 64))
        self.W4 = np.random.randn(64, 10) * np.sqrt(2.0 / 64)
        self.b4 = np.zeros((1, 10))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = relu(self.z3)
        self.z4 = self.a3 @ self.W4 + self.b4
        self.a4 = softmax(self.z4)
        return self.a4

    def backward(self, X, y_onehot, lr=0.001):
        n = X.shape[0]

        # Output layer
        dz4 = self.a4 - y_onehot
        dW4 = self.a3.T @ dz4 / n
        db4 = np.sum(dz4, axis=0, keepdims=True) / n

        # Hidden layer 3
        dz3 = (dz4 @ self.W4.T) * relu_deriv(self.z3)
        dW3 = self.a2.T @ dz3 / n
        db3 = np.sum(dz3, axis=0, keepdims=True) / n

        # Hidden layer 2
        dz2 = (dz3 @ self.W3.T) * relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2 / n
        db2 = np.sum(dz2, axis=0, keepdims=True) / n

        # Hidden layer 1
        dz1 = (dz2 @ self.W2.T) * relu_deriv(self.z1)
        dW1 = X.T @ dz1 / n
        db1 = np.sum(dz1, axis=0, keepdims=True) / n

        # Update weights
        self.W4 -= lr * dW4;  self.b4 -= lr * db4
        self.W3 -= lr * dW3;  self.b3 -= lr * db3
        self.W2 -= lr * dW2;  self.b2 -= lr * db2
        self.W1 -= lr * dW1;  self.b1 -= lr * db1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1,
                       W2=self.W2, b2=self.b2,
                       W3=self.W3, b3=self.b3,
                       W4=self.W4, b4=self.b4)

    def load(self, path):
        data = np.load(path)
        self.W1 = data["W1"]; self.b1 = data["b1"]
        self.W2 = data["W2"]; self.b2 = data["b2"]
        self.W3 = data["W3"]; self.b3 = data["b3"]
        self.W4 = data["W4"]; self.b4 = data["b4"]


# ---------------------------------------------------------------------------
# 2. MNIST data loader
# ---------------------------------------------------------------------------

def fetch_mnist():
    """Download MNIST using sklearn."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        print("Installing scikit-learn (needed once for MNIST download)...")
        os.system(f"{sys.executable} -m pip install scikit-learn")
        from sklearn.datasets import fetch_openml

    print("Downloading MNIST dataset... (one-time)")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 3. Training (with learning rate decay)
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "digit_model.npz")

def train_model():
    X_train, y_train, X_test, y_test = fetch_mnist()

    nn = SimpleNeuralNetwork()
    epochs = 30
    batch_size = 64
    lr_start = 0.01

    print(f"\nTraining neural network for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        # Learning rate decay
        lr = lr_start * (0.95 ** (epoch - 1))

        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        for i in range(0, len(X_train), batch_size):
            Xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]
            nn.forward(Xb)
            nn.backward(Xb, one_hot(yb), lr=lr)

        # Evaluate every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            preds = nn.predict(X_test)
            acc = np.mean(preds == y_test) * 100
            print(f"  Epoch {epoch:2d}/{epochs}  lr={lr:.4f}  Test accuracy: {acc:.1f}%")

    nn.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return nn


def load_or_train():
    nn = SimpleNeuralNetwork()
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        nn.load(MODEL_PATH)
    else:
        print("No saved model found — training a new one.")
        nn = train_model()
    return nn


# ---------------------------------------------------------------------------
# 4. MNIST-style preprocessing (THE KEY TO ACCURACY)
# ---------------------------------------------------------------------------

def preprocess_drawing(pil_image):
    """
    Convert the canvas drawing into a 28x28 image that matches
    the MNIST format as closely as possible.

    MNIST preprocessing steps:
    1. Find bounding box of the drawn content
    2. Crop to content
    3. Fit into a 20x20 box (preserving aspect ratio)
    4. Place centered in a 28x28 image using center of mass
    """
    from PIL import Image, ImageFilter, ImageOps

    img = pil_image.copy()

    # Apply slight blur to smooth strokes
    img = img.filter(ImageFilter.GaussianBlur(radius=2))

    # Convert to numpy to find bounding box
    arr = np.array(img)

    # Find bounding box of non-zero pixels
    rows = np.any(arr > 30, axis=1)
    cols = np.any(arr > 30, axis=0)

    if not rows.any() or not cols.any():
        # Empty canvas
        return np.zeros(784, dtype=np.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box with small padding
    pad = 20
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)

    cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Resize to fit in 20x20 box (preserving aspect ratio) — MNIST standard
    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(20 * h / w))
    else:
        new_h = 20
        new_w = max(1, int(20 * w / h))

    cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

    # Place in 28x28 canvas, centered by center of mass
    arr_small = np.array(cropped, dtype=np.float32)

    # Compute center of mass
    total = arr_small.sum()
    if total > 0:
        gy = np.sum(np.arange(arr_small.shape[0]).reshape(-1, 1) * arr_small) / total
        gx = np.sum(np.arange(arr_small.shape[1]).reshape(1, -1) * arr_small) / total
    else:
        gy, gx = new_h / 2, new_w / 2

    # Calculate offset to center the center-of-mass at (14, 14) — center of 28x28
    offset_y = int(round(14 - gy))
    offset_x = int(round(14 - gx))

    # Place on 28x28 canvas
    result = np.zeros((28, 28), dtype=np.float32)
    # Calculate source and destination ranges
    src_y_start = max(0, -offset_y)
    src_y_end = min(new_h, 28 - offset_y)
    src_x_start = max(0, -offset_x)
    src_x_end = min(new_w, 28 - offset_x)

    dst_y_start = max(0, offset_y)
    dst_x_start = max(0, offset_x)

    h_copy = src_y_end - src_y_start
    w_copy = src_x_end - src_x_start

    if h_copy > 0 and w_copy > 0:
        result[dst_y_start:dst_y_start + h_copy,
               dst_x_start:dst_x_start + w_copy] = \
            arr_small[src_y_start:src_y_start + h_copy,
                      src_x_start:src_x_start + w_copy]

    # Normalize to [0, 1]
    if result.max() > 0:
        result = result / 255.0

    return result.reshape(784)


# ---------------------------------------------------------------------------
# 5. Drawing GUI (tkinter)
# 5. Network Visualization (matplotlib)
# ---------------------------------------------------------------------------

def visualize_network(nn, flat_input):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Installing matplotlib for visualization...")
        os.system(f"{sys.executable} -m pip install matplotlib")
        import matplotlib.pyplot as plt

    # Forward pass to get activations
    nn.forward(flat_input.reshape(1, 784))
    
    # Create an "X-Ray" figure
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    fig.canvas.manager.set_window_title('Neural Network X-Ray')
    
    # 1. Input Image
    axes[0].imshow(flat_input.reshape(28, 28), cmap='gray')
    axes[0].set_title('1. Network Input (28x28)')
    axes[0].axis('off')
    
    # 2. Layer 1 Activations (256 neurons reshaped to 16x16)
    axes[1].imshow(nn.a1.reshape(16, 16), cmap='viridis')
    axes[1].set_title('2. Layer 1 (256 Neurons)')
    axes[1].axis('off')
    
    # 3. Layer 2 Activations (128 neurons reshaped to 16x8)
    axes[2].imshow(nn.a2.reshape(16, 8), cmap='viridis')
    axes[2].set_title('3. Layer 2 (128 Neurons)')
    axes[2].axis('off')
    
    # 4. Layer 3 Activations (64 neurons reshaped to 8x8)
    axes[3].imshow(nn.a3.reshape(8, 8), cmap='viridis')
    axes[3].set_title('4. Layer 3 (64 Neurons)')
    axes[3].axis('off')
    
    # 5. Output Probabilities
    probs = nn.a4[0] * 100
    colors = ['#45475a' if i != np.argmax(probs) else '#89b4fa' for i in range(10)]
    axes[4].bar(range(10), probs, color=colors)
    axes[4].set_title('5. Output Probabilities (%)')
    axes[4].set_xticks(range(10))
    axes[4].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Drawing GUI (tkinter)
# ---------------------------------------------------------------------------

def run_gui(nn):
    import tkinter as tk
    from PIL import Image, ImageDraw

    CANVAS_SIZE = 280
    BG = "#1e1e2e"
    FG = "#cdd6f4"
    ACCENT = "#89b4fa"

    root = tk.Tk()
    root.title("✏️  Digit Recognizer")
    root.configure(bg=BG)
    root.resizable(False, False)

    # --- PIL image to capture strokes ---
    pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    pil_draw = ImageDraw.Draw(pil_image)

    # --- Header ---
    tk.Label(root, text="Draw a digit (0 – 9)", font=("Segoe UI", 16, "bold"),
             bg=BG, fg=FG).pack(pady=(14, 4))
    tk.Label(root, text="Draw BIG and centered for best results",
             font=("Segoe UI", 9), bg=BG, fg="#6c7086").pack()

    # --- Canvas ---
    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                       bg="black", cursor="crosshair", highlightthickness=2,
                       highlightbackground=ACCENT)
    canvas.pack(padx=20, pady=10)

    # --- Prediction label ---
    result_var = tk.StringVar(value="Draw something!")
    confidence_var = tk.StringVar(value="")
    all_probs_var = tk.StringVar(value="")
    tk.Label(root, textvariable=result_var, font=("Segoe UI", 28, "bold"),
             bg=BG, fg=ACCENT).pack()
    tk.Label(root, textvariable=confidence_var, font=("Segoe UI", 12),
             bg=BG, fg=FG).pack()
    tk.Label(root, textvariable=all_probs_var, font=("Consolas", 9),
             bg=BG, fg="#6c7086", justify="center").pack(pady=(2, 0))

    # --- Drawing callbacks ---
    brush_radius = 10  # Thicker brush for better strokes
    last_x, last_y = None, None

    def paint(event):
        nonlocal last_x, last_y
        x, y = event.x, event.y
        r = brush_radius

        # Draw line between last point and current point for smooth strokes
        if last_x is not None and last_y is not None:
            canvas.create_line(last_x, last_y, x, y,
                              fill="white", width=r*2, capstyle="round",
                              joinstyle="round")
            pil_draw.line([last_x, last_y, x, y], fill=255, width=r*2)
        else:
            canvas.create_oval(x - r, y - r, x + r, y + r,
                              fill="white", outline="white")
            pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        last_x, last_y = x, y

    def reset_last(event):
        nonlocal last_x, last_y
        last_x, last_y = None, None

    canvas.bind("<B1-Motion>", paint)
    canvas.bind("<Button-1>", paint)
    canvas.bind("<ButtonRelease-1>", reset_last)

    # --- Predict ---
    def predict():
        flat = preprocess_drawing(pil_image)

        if flat.max() == 0:
            result_var.set("Canvas is empty!")
            confidence_var.set("")
            all_probs_var.set("")
            return

        probs = nn.forward(flat.reshape(1, 784))[0]
        digit = int(np.argmax(probs))
        conf = probs[digit] * 100

        result_var.set(f"Prediction:  {digit}")
        confidence_var.set(f"Confidence: {conf:.1f}%")

        # Show all probabilities as a bar
        bars = "  ".join(f"{i}:{probs[i]*100:4.1f}%" for i in range(10))
        all_probs_var.set(bars)

        # Console output
        top3 = np.argsort(probs)[::-1][:3]
        print(f"  Top 3 → {', '.join(f'{d}({probs[d]*100:.1f}%)' for d in top3)}")

    def clear():
        nonlocal last_x, last_y
        canvas.delete("all")
        pil_draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        result_var.set("Draw something!")
        confidence_var.set("")
        all_probs_var.set("")
        last_x, last_y = None, None

    def on_visualize():
        flat = preprocess_drawing(pil_image)
        if flat.max() == 0:
            result_var.set("Canvas is empty!")
            return
        visualize_network(nn, flat)

    # --- Buttons ---
    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(pady=10)

    btn_style = dict(font=("Segoe UI", 12), width=10, relief="flat",
    btn_style = dict(font=("Segoe UI", 12), width=9, relief="flat",
                     cursor="hand2", bd=0, pady=6)

    tk.Button(btn_frame, text="🔍 Predict", bg=ACCENT, fg="#1e1e2e",
              command=predict, **btn_style).pack(side="left", padx=8)
              command=predict, **btn_style).pack(side="left", padx=4)
    tk.Button(btn_frame, text="🧠 X-Ray", bg="#a6e3a1", fg="#1e1e2e",
              command=on_visualize, **btn_style).pack(side="left", padx=4)
    tk.Button(btn_frame, text="🗑️ Clear", bg="#45475a", fg=FG,
              command=clear, **btn_style).pack(side="left", padx=8)
              command=clear, **btn_style).pack(side="left", padx=4)

    # --- Footer ---
    tk.Label(root, text="Neural Net: 784→256→128→64→10  •  Trained on MNIST",
             font=("Segoe UI", 9), bg=BG, fg="#6c7086").pack(pady=(4, 12))

    root.mainloop()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check for Pillow
    try:
        from PIL import Image
    except ImportError:
        print("Installing Pillow (needed for image processing)...")
        os.system(f"{sys.executable} -m pip install Pillow")

    nn = load_or_train()
    print("\nOpening drawing canvas...")
    run_gui(nn)
