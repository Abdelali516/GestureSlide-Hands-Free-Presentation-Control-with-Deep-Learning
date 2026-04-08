# 🖐️ Hand Gesture Recognition — GRU-Based Left/Right Detection

A real-time hand gesture recognition system that uses a **GRU (Gated Recurrent Unit)** neural network to detect whether a hand moves to the **right or left**, and maps those gestures to keyboard actions. Built with PyTorch, MediaPipe, and OpenCV.

> Achieved **99% accuracy** on the test set.

---

## 📌 What It Does

The model watches a live webcam feed, tracks hand landmarks over a **20-frame sliding window**, and classifies whether the hand gesture is directed to the **right (`r`)** or **left (`l`)**. Once a gesture is detected, it triggers a keyboard shortcut — by default, the **Left/Right arrow keys** — making it useful for controlling presentations, slideshows, or any keyboard-driven interface, hands-free.

---

## 🧠 Model Architecture

- **Input:** 20 frames × 63 features (21 MediaPipe hand landmarks × x, y, z)
- **GRU:** 2 layers, hidden size 128, dropout 0.3
- **Output:** 2 classes — right (`r`) or left (`l`)
- **Optimizer:** Adam (lr = 0.001)
- **Loss:** CrossEntropyLoss

The temporal nature of a sweeping hand gesture makes GRU a natural fit — it captures the motion trajectory across frames rather than treating each frame in isolation.

---

## 📁 Project Structure

```
├── data_creation.py                  # Collect gesture samples and save to CSV
├── creation_of_the_neural_network_(GRU).py   # Train the GRU model
├── results.py                        # Run real-time inference and trigger keys
├── dataset.csv                       # Your collected gesture data (not included)
└── gru_model.pth                     # Saved model weights (generated after training)
```

---

## ⚙️ Requirements

```bash
pip install torch torchvision opencv-python mediapipe scikit-learn pandas numpy
```

On Linux, keyboard control also requires:
```bash
sudo apt install xdotool
```

---

## 🚀 How to Use

### Step 1 — Collect Data

Run `data_creation.py` to record gesture samples via your webcam.

```bash
python data_creation.py
```

- Perform a **right** or **left** hand sweep gesture
- Press **`s`** to save a 20-frame sequence to `dataset.csv`
- The first column of each row is the label (`r` or `l`) — **edit the label in the script** before recording each class
- Press **`q`** to quit

> 💡 The more samples you collect, the better. Start with at least **100 samples per class**, and you can always add more later — the CSV just grows and you retrain.

### Step 2 — Train the Model

```bash
python "creation_of_the_neural_network_(GRU).py"
```

- Trains for 50 epochs with an 80/20 train/test split
- Prints loss and accuracy each epoch
- Saves the trained model to `gru_model.pth`

### Step 3 — Run Real-Time Inference

```bash
python results.py
```

- Opens the webcam and starts detecting gestures in real time
- Detected **right** gesture → triggers `Right` arrow key
- Detected **left** gesture → triggers `Left` arrow key
- A **1.5-second cooldown** prevents repeated triggers
- Press **`q`** to quit

---

## 🔧 Customization

### Change the target window or key bindings

In `results.py`, locate the `subprocess.run` calls and update them to match your use case:

```python
# Right gesture
subprocess.run(['xdotool', 'windowfocus', 'YOUR_WINDOW_ID', 'key', 'Right'])

# Left gesture
subprocess.run(['xdotool', 'windowfocus', 'YOUR_WINDOW_ID', 'key', 'Left'])
```

To get your window ID, run:
```bash
xdotool getactivewindow
```

### Adjust the cooldown

```python
cool_time = 1.5  # seconds between triggers — lower = more responsive
```

### Add more samples without restarting from scratch

Just re-run `data_creation.py` — it **appends** to the existing `dataset.csv`. Then retrain.

---

## 📊 Dataset Format

Each row in `dataset.csv` has **1261 values**:

```
label, [frame_1: x1,y1,z1,...,x21,y21,z21], [frame_2: ...], ..., [frame_20: ...]
```

- `label`: `r` (right) or `l` (left)
- Followed by 20 × 63 = 1260 landmark coordinates

---

## 🖥️ Platform Notes

- Tested on **Linux** with an **NVIDIA RTX 3070 Ti**
- The `xdotool` key injection is **Linux-only** — on Windows/macOS, replace it with `pyautogui` or `pynput`
- CUDA is used automatically if available, otherwise falls back to CPU

---

## 🔭 Context & Future Work

This module is part of a larger **ASL Hand Sign Recognition** system that combines:
- A **feedforward neural network (FFNN)** for static signs (A–Z)
- This **GRU model** for dynamic gestures (J and Z, or directional motion)
- An **LLM via Ollama** for natural language communication

The full unified pipeline is in active development.

---

## 📄 License

MIT License — free to use, modify, and build on.
