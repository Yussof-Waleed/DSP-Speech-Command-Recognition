# 🎤 DSP-Speech-Command-Recognition

> A **Distance-Based Speech Recognition System** using classical Digital Signal Processing (DSP) techniques — no deep learning required.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Local%20%7C%20Colab%20%7C%20Kaggle-orange.svg)](#-quick-start)

---

## 📋 Overview

This project implements a complete speech recognition pipeline for classifying **8 spoken commands** using traditional DSP methods. It demonstrates fundamental signal processing concepts without relying on neural networks or deep learning frameworks.

### 🎯 Supported Commands

| | | | |
|:---:|:---:|:---:|:---:|
| `down` | `go` | `left` | `no` |
| `right` | `stop` | `up` | `yes` |

---

## ✨ Features

- **🔧 Complete DSP Pipeline** — From raw audio to classification
- **📊 No Deep Learning** — Pure signal processing approach
- **🌐 Cross-Platform** — Runs on Local, Google Colab, and Kaggle
- **💾 Smart Caching** — Preprocessed data saved for fast reloading
- **📈 Visualizations** — Waveforms, spectrograms, and processing steps
- **🧹 Clean Code** — Follows SOLID, DRY, and KISS principles

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SPEECH RECOGNITION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────┐    ┌───────────────┐  │
│  │  Audio   │───▶│ Preprocessing │───▶│ Framing │───▶│   Features    │  │
│  │  Input   │    │              │    │         │    │  (Coming Soon) │  │
│  └──────────┘    └──────────────┘    └─────────┘    └───────────────┘  │
│       │                │                  │                 │          │
│       ▼                ▼                  ▼                 ▼          │
│   .wav files     • DC removal       • 25ms frames     • MFCC          │
│   16kHz mono     • Pre-emphasis     • 10ms hop        • Energy        │
│   1 second       • Normalization    • Hamming window  • ZCR           │
│                  • Deduplication                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Google Colab
1. Upload the notebook to Colab
2. Run all cells

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DSP-Speech-Command-Recognition.git
cd DSP-Speech-Command-Recognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook speech_recognition_pipeline.ipynb
```

### Option 3: Kaggle
1. Upload the notebook to Kaggle
2. Enable Internet access in notebook settings
3. Run all cells

---

## 📁 Project Structure

```
DSP-Speech-Command-Recognition/
│
├── 📓 speech_recognition_pipeline.ipynb  # Main notebook (self-contained)
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 LICENSE                            # MIT License
│
├── 📂 mini_speech_commands/              # Dataset (auto-downloaded)
│   └── 📂 mini_speech_commands/
│       ├── 📂 down/
│       ├── 📂 go/
│       ├── 📂 left/
│       ├── 📂 no/
│       ├── 📂 right/
│       ├── 📂 stop/
│       ├── 📂 up/
│       └── 📂 yes/
│
├── 📂 processed_data/                    # Cached preprocessed data
    ├── preprocessed_data.npz
    └── metadata.json

```

---

## 📊 Dataset

This project uses the **Mini Speech Commands** dataset from Google:

| Property | Value |
|----------|-------|
| **Source** | [TensorFlow Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) |
| **Samples** | 8,000 total (1,000 per class) |
| **Classes** | 8 speech commands |
| **Format** | WAV, 16-bit PCM |
| **Sample Rate** | 16,000 Hz |
| **Duration** | ~1 second per sample |

The dataset is automatically downloaded when you run the notebook.

---

## 🔬 Technical Details

### Preprocessing Pipeline

| Step | Description | Purpose |
|------|-------------|---------|
| **1. Duplicate Removal** | MD5 hash comparison | Remove redundant samples |
| **2. DC Offset Removal** | Subtract mean | Center signal at zero |
| **3. Length Normalization** | Pad/truncate to 16,000 samples | Ensure uniform length |
| **4. Pre-emphasis** | `y[n] = x[n] - 0.97·x[n-1]` | Boost high frequencies |
| **5. Amplitude Normalization** | Scale to [-1, 1] | Standardize amplitude range |

### Framing Parameters

| Parameter | Value | Samples |
|-----------|-------|---------|
| Frame Length | 25 ms | 400 |
| Hop Length | 10 ms | 160 |
| Overlap | 60% | 240 |
| Window | Hamming | — |
| Frames/Sample | 98 | — |

---

## 📈 Results

### Dataset Statistics After Preprocessing

```
┌────────────────────────────────────────┐
│         PREPROCESSING RESULTS          │
├────────────────────────────────────────┤
│  Original samples:     8,000           │
│  Duplicates removed:      27           │
│  Final samples:        7,973           │
│  Total frames:       781,354           │
├────────────────────────────────────────┤
│  Samples per class:   ~996-998         │
│  Dataset balanced:    ✅ Yes           │
└────────────────────────────────────────┘
```

---

## Technical Implementation Details

### Train / Test Split 
```python
X_train_frames, X_test_frames, y_train, y_test = train_test_split(
    frames_data, labels, test_size=0.20, random_state=42, stratify=labels
)


### FFT Feature Extraction 
X_train_spectrum = np.abs(np.fft.rfft(X_train_frames, axis=2)).astype(np.float32)
X_test_spectrum  = np.abs(np.fft.rfft(X_test_frames,  axis=2)).astype(np.float32)
# Output: (6378, 98, 201) → 98 frames × 201 frequency bins
```


## 🛠️ Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

See [`requirements.txt`](requirements.txt) for exact versions.

---

## 🗺️ Roadmap

- [x] **Stage 1:** Data Loading & Verification
- [x] **Stage 2:** Preprocessing & Framing
- [ ] **Stage 3:** Feature Extraction (MFCC, Energy, ZCR)
- [ ] **Stage 4:** Template Creation (Reference patterns)
- [ ] **Stage 5:** Distance Metrics (DTW, Euclidean)
- [ ] **Stage 6:** Classification & Evaluation
- [ ] **Stage 7:** Real-time Demo

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Yussof Waleed**
- GitHub: [@Yussof-waleed](https://github.com/Yussof-Waleed)
- University: Helwan University - Level 4 - DSP Course

---

