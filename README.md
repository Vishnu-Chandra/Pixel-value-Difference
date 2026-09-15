# AI-EPVD: AI-Enhanced Pixel Value Differencing Image Steganography

An advanced image steganography framework combining **Pixel Value Differencing (EPVD)**, **Machine Learning (Random Forest Adaptive Bit Allocation)**, and **Chaotic Encryption** for high-capacity, visually imperceptible, and secure data hiding in digital images.

---

## 🌟 Key Features

- **Standard EPVD & Extended EPVD**: Pixel Value Differencing adapted for color spaces (YCbCr) across high/low texture regions.
- **AI-Adaptive Embedding**: Trained Machine Learning models (Random Forest) predicting optimal bit embedding capacities per pixel block to maximize PSNR while preserving secret payload integrity.
- **Chaotic Encryption & Security**: Logistic Map / Chaotic permutation ensuring embedded secrets remain cipher-grade and resistant to statistical steganalysis attacks.
- **Interactive Full-Stack Web App**:
  - **Backend**: Flask API with REST endpoints for embedding, extraction, evaluation, and benchmark diagnostics.
  - **Frontend**: Modern React + Vite UI for interactive drag-and-drop cover image uploading, secret message insertion, embedding mode selection, and real-time visual inspection (PSNR, SSIM, BPP, BER).

---

## 📁 Project Architecture

```
AI_EPVD/
├── backend/
│   ├── app.py                # Flask REST API server
│   └── uploads/              # Runtime upload & stego output directory
├── data/
│   ├── cover/                # Standard benchmark cover images (Lena, Baboon, etc.)
│   ├── secret/               # Test secret messages / payloads
│   └── *.csv                 # Extracted dataset features & labels
├── frontend/
│   ├── src/                  # React components & UI
│   ├── package.json          # Node dependencies
│   └── vite.config.js        # Vite configuration
├── models/
│   ├── random_forest.pkl     # Trained ML classifier models
│   ├── random_forest_5to7.pkl
│   └── random_forest_5to8.pkl
├── results/                  # Benchmark plots, generated stego images, and CSVs
├── src/
│   ├── ai_adaptive.py        # Adaptive ML embedding routines
│   ├── chaotic.py            # Chaotic encryption and map generators
│   ├── embedding.py          # Core embedding implementations
│   ├── extraction.py         # Secret message extraction routines
│   ├── epvd.py               # Pixel Value Differencing calculations
│   └── evaluation.py         # PSNR, SSIM, BPP, BER metric calculators
├── main.py                   # Root execution entrypoint
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)

### 2. Python Backend Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate # Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python main.py
```
The server will start at `http://127.0.0.1:5000`.

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🔬 Supported Embedding Modes

1. **Standard EPVD**: Classical range-table based pixel value differencing on YCbCr components.
2. **AI-EPVD**: Random Forest classification dynamically assigning 5-to-8 bit capacities based on local texture and variance.
3. **Secure AI-EPVD**: AI-guided bit distribution fused with chaotic key-based bit permutation.
