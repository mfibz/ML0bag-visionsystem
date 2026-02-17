# How to Run: RTX 5090 + Google Colab

## Your Two Options

### Option A: Google Colab Connected to Your RTX 5090 (Recommended)

This gives you Colab's notebook UI but runs everything on YOUR GPU.

**On your PC (one-time setup):**

```bash
# 1. Install Python 3.10+ if you don't have it
# Download from python.org or use your package manager

# 2. Create a virtual environment
python -m venv bag-vision-env
source bag-vision-env/bin/activate   # Linux/Mac
# bag-vision-env\Scripts\activate    # Windows

# 3. Install PyTorch with CUDA (for RTX 5090, use CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install Jupyter with Colab support
pip install jupyter jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws

# 5. Install project dependencies
pip install ultralytics>=8.3.0 opencv-python huggingface-hub datasets roboflow \
    Pillow numpy pandas pyyaml tqdm matplotlib seaborn loguru click albumentations

# 6. Clone your project (or copy it to your PC)
git clone <your-repo-url>
cd ML0bag-visionsystem
```

**Start training session (every time):**

```bash
# 1. Activate your environment
source bag-vision-env/bin/activate

# 2. Start Jupyter with Colab access
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0
```

**In Google Colab:**

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click the **Connect** dropdown (top right corner)
3. Select **"Connect to a local runtime"**
4. Paste the URL from your terminal (looks like `http://localhost:8888/?token=abc123...`)
5. Click **Connect**
6. Open `notebooks/00_colab_training.ipynb` and run the cells

Your RTX 5090's 32GB VRAM will handle all three training phases easily.

---

### Option B: Run Locally Without Colab

Same setup but run Jupyter directly:

```bash
# After the one-time setup above, just:
cd ML0bag-visionsystem
jupyter notebook

# Open notebooks/00_colab_training.ipynb in your browser
```

Or skip notebooks entirely and use the CLI:

```bash
# Download datasets
python -m src.data.download --config datasets

# Preprocess
python -m src.data.preprocess all

# Train each phase
python -m src.models.trainer train phase1
python -m src.models.trainer train phase2
python -m src.models.trainer train phase3

# Run inference on an image
python -m src.inference.camera --source path/to/bag/photo.jpg

# Run inference on live camera
python -m src.inference.camera --source 0
```

---

## Why NOT Docker (For Now)

Docker adds a layer between your GPU and the code. While it works, it requires:
- NVIDIA Container Toolkit installed
- Docker GPU passthrough configured
- More complex debugging

For training on your own machine, running directly with Python is simpler and faster. Save Docker for when you deploy the model as a service later.

---

## API Keys You Need

| Service | What For | How to Get |
|---------|----------|------------|
| **Roboflow** | Download bag logo dataset (Phase 1) | Free at [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api) |
| **HuggingFace** | Download LFFD dataset (Phase 3) | Optional, at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| **Kaggle** | Download leather defects (Phase 2) | At [kaggle.com/settings](https://www.kaggle.com/settings) > API > Create Token |

---

## Expected Training Times (RTX 5090)

| Phase | Model | Epochs | Estimated Time |
|-------|-------|--------|---------------|
| Phase 1: Detection | YOLOv11m | 150 | 30-60 min |
| Phase 2: Condition | YOLOv11m | 200 | 1-3 hours |
| Phase 3: Auth | YOLOv11s-cls | 100 | 15-30 min |

---

## Quick Start Checklist

- [ ] Python 3.10+ installed on PC with RTX 5090
- [ ] NVIDIA drivers installed (550+ for RTX 5090)
- [ ] CUDA 12.4+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch with CUDA installed (`python -c "import torch; print(torch.cuda.is_available())"` should print `True`)
- [ ] Project cloned/copied to PC
- [ ] Roboflow API key obtained
- [ ] Jupyter started with Colab support
- [ ] Colab connected to local runtime
- [ ] Run the notebook cells in order
