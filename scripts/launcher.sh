#!/bin/bash

# Navigate to the script's directory to ensure paths are correct
cd "$(dirname "$0")/.."

# --- MIGRATION REF -> VENDOR ---
if [ -d "ref" ] && [ ! -d "vendor" ]; then
    echo "[System] Renaming deprecated 'ref' folder to 'vendor'..."
    mv ref vendor
fi

# --- FOLDERS SETUP ---
mkdir -p data output vendor/libs

# --- Configuration ---
DIFFUSION_PIPE_REPO="https://github.com/tdrussell/diffusion-pipe.git"
DIFFUSION_PIPE_DIR="vendor/diffusion-pipe"
COMFYUI_REPO="https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_DIR="vendor/ComfyUI"
HUNYUAN_REPO="https://github.com/Tencent/HunyuanVideo.git"
HUNYUAN_DIR="vendor/HunyuanVideo"

DPGUI_REQUIREMENTS="requirements.txt"
BACKEND_APP="main:app"

# --- Network Configuration ---
HOST="${DPGUI_HOST:-0.0.0.0}"
FRONTEND_PORT="${DPGUI_PORT:-5173}"
FRONTEND_DIR="frontend"
NODE_VERSION="v22.12.0" 

echo "=========================================="
echo "    DPGui Launcher (Stable 2025)"
echo "    Target: PyTorch 2.9.1 + CUDA 12.8"
echo "=========================================="

# --- Step 0: Ensure Node.js ---
if ! command -v node &> /dev/null; then
    NODE_DIR="$(pwd)/tools/node"
    if [ ! -f "$NODE_DIR/bin/node" ]; then
        ARCH=$(uname -m)
        case $ARCH in
            x86_64)  NODE_ARCH="x64" ;;
            aarch64) NODE_ARCH="arm64" ;;
            *)       echo "Error: Unsupported architecture $ARCH"; exit 1 ;;
        esac
        NODE_DIST="node-$NODE_VERSION-linux-$NODE_ARCH"
        NODE_URL="https://nodejs.org/dist/$NODE_VERSION/$NODE_DIST.tar.xz"
        mkdir -p "$NODE_DIR"
        
        if command -v curl &> /dev/null; then
            curl -L "$NODE_URL" | tar -xJ -C "$NODE_DIR" --strip-components=1
        elif command -v wget &> /dev/null; then
            wget -qO- "$NODE_URL" | tar -xJ -C "$NODE_DIR" --strip-components=1
        fi
    fi
    export PATH="$NODE_DIR/bin:$PATH"
fi

# --- Step 1: Clone Repositories ---
if [ ! -d "$DIFFUSION_PIPE_DIR" ]; then git clone "$DIFFUSION_PIPE_REPO" "$DIFFUSION_PIPE_DIR"; fi
if [ ! -d "$COMFYUI_DIR" ]; then git clone "$COMFYUI_REPO" "$COMFYUI_DIR"; fi
if [ ! -d "$HUNYUAN_DIR" ]; then git clone "$HUNYUAN_REPO" "$HUNYUAN_DIR"; fi

# --- Step 2: Symlinks ---
if [ ! -L "vendor/libs/comfy" ] && [ -d "$COMFYUI_DIR/comfy" ]; then ln -s ../ComfyUI/comfy vendor/libs/comfy; fi
if [ ! -L "vendor/libs/hyvideo" ] && [ -d "$HUNYUAN_DIR/hyvideo" ]; then ln -s ../HunyuanVideo/hyvideo vendor/libs/hyvideo; fi

# --- Step 3: Install Backend Dependencies (STABLE CHECK) ---

# Vérification stricte de la version 2.9.1
if ! python3 -c "import torch; assert torch.__version__.startswith('2.9.1'), f'Wrong version: {torch.__version__}'" &> /dev/null; then
    echo "[Backend] PyTorch 2.9.1 not found. Starting installation..."
    
    # Nettoyage préventif
    pip cache purge
    
    # 1. INSTALLATION PYTORCH 2.9.1 (STABLE / CUDA 12.8)
    echo "[Backend] Installing PyTorch 2.9.1 (Stable) with CUDA 12.8..."
    # Note : Index cu128 pour CUDA 12.8 support
    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    
    # 2. INSTALLATION DEPENDANCES (Numpy < 2.0 pour DeepSpeed)
    echo "[Backend] Installing Core Requirements..."
    if [ -f "$DPGUI_REQUIREMENTS" ]; then
        pip install -r "$DPGUI_REQUIREMENTS"
    fi
    
    # 3. VENDOR DEPS
    echo "[Backend] Installing vendor utils..."
    if [ -f "$DIFFUSION_PIPE_DIR/requirements.txt" ]; then
        pip install --no-deps -r "$DIFFUSION_PIPE_DIR/requirements.txt"
    fi
    if [ -f "$COMFYUI_DIR/requirements.txt" ]; then
        pip install --no-deps -r "$COMFYUI_DIR/requirements.txt"
    fi
    
    echo "[Backend] Environment ready."
else
    echo "[Backend] correct PyTorch 2.9.1 detected."
fi

# --- Step 4: Setup Frontend ---
if [ ! -d "$FRONTEND_DIR" ]; then
    npx --yes create-vite@5.2.0 "$FRONTEND_DIR" --template react
    cd "$FRONTEND_DIR" && npm install && npm install axios react-hook-form react-router-dom lucide-react && cd ..
elif [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    cd "$FRONTEND_DIR" && npm install && cd ..
fi

# --- Step 5: Launch Services ---
BACKEND_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "VITE_API_BASE_URL=http://127.0.0.1:$BACKEND_PORT" > "$FRONTEND_DIR/.env.local"
export PYTHONPATH="$PYTHONPATH:$(pwd)/vendor/libs"

cleanup() {
    echo "Stopping DPGui..."
    if [ ! -z "$BACKEND_PID" ]; then kill "$BACKEND_PID" 2>/dev/null; fi
    exit
}
trap cleanup SIGINT

echo "Starting Backend..."
uvicorn "$BACKEND_APP" --host "$HOST" --port "$BACKEND_PORT" > backend.log 2>&1 &
BACKEND_PID=$!

sleep 2
echo "Starting Frontend..."
echo "Access URL: http://localhost:$FRONTEND_PORT"
cd "$FRONTEND_DIR"
npm run dev -- --host --port "$FRONTEND_PORT"