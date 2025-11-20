# GPU Setup Guide

To use GPU acceleration for faster OCR processing:

## 1. Check Your CUDA Version

Open Command Prompt and run:
```bash
nvidia-smi
```

Look for the CUDA version in the top right corner (e.g., "CUDA Version: 11.8" or "12.3")

## 2. Uninstall Current PaddlePaddle

```bash
pip uninstall paddlepaddle paddlepaddle-gpu -y
```

## 3. Install GPU Version

**For CUDA 11.8:**
```bash
pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

**For CUDA 12.3:**
```bash
pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

**For other CUDA versions:**
Visit: https://www.paddlepaddle.org.cn/install/quick

## 4. Enable GPU in main.py

Edit `main.py` and set:
```python
USE_GPU = True
```

## 5. Test GPU

Run the script - you should see:
```
GPU mode: ENABLED
```

And PaddleOCR will use GPU for processing (much faster!).

## WSL2 (Windows Subsystem for Linux) Setup

WSL2 requires additional setup for GPU:

1. **Install NVIDIA CUDA Toolkit in WSL2:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
```

2. **Set environment variables:**
Add to `~/.bashrc`:
```bash
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```

3. **Reload shell:**
```bash
source ~/.bashrc
```

4. **Verify CUDA:**
```bash
nvcc --version
```

5. **Then install paddlepaddle-gpu** as shown above.

## Troubleshooting

**"libcusolver.so.12: cannot open shared object file"** (WSL2)
- CUDA libraries not installed in WSL2
- Follow WSL2 setup steps above
- Or set `USE_GPU = False` to use CPU mode

**"ImportError: cannot import name..."**
- Your paddlepaddle installation is broken
- Follow step 2 and 3 above to reinstall

**"Using CPU. Note: This module is much faster with a GPU."**
- This message is from EasyOCR
- To enable GPU for EasyOCR, you need PyTorch with CUDA
- Install: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

**GPU not detected**
- Make sure NVIDIA drivers are installed
- Check CUDA is installed: run `nvidia-smi`
- Verify CUDA version matches your paddlepaddle-gpu installation
