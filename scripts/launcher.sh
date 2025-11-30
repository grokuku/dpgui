#!/bin/bash

# Navigate to the script's directory to ensure paths are correct
cd "$(dirname "$0")/.."

# --- MIGRATION REF -> VENDOR ---
if [ -d "ref" ] && [ ! -d "vendor" ]; then
    echo "[System] Renaming deprecated 'ref' folder to 'vendor'..."
    mv ref vendor
fi

# --- FOLDERS SETUP ---
# Creation of default project directories for portability
if [ ! -d "data" ]; then
    echo "[System] Creating default 'data' directory..."
    mkdir -p data
fi
if [ ! -d "output" ]; then
    echo "[System] Creating default 'output' directory..."
    mkdir -p output
fi

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

# Frontend settings
FRONTEND_DIR="frontend"
NODE_VERSION="v22.12.0" 

echo "=========================================="
echo "    DPGui Launcher (Full Stack)"
echo "=========================================="

# --- Step 0: Ensure Node.js AND NPM are available ---
echo "[System] Checking Node.js and NPM..."

if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
    if ! command -v npm &> /dev/null; then
        echo "NPM not found. Using local portable version..."
    else
        echo "Node.js not found. Using local portable version..."
    fi
    
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)  NODE_ARCH="x64" ;;
        aarch64) NODE_ARCH="arm64" ;;
        *)       echo "Error: Unsupported architecture $ARCH"; exit 1 ;;
    esac
    
    NODE_DIR="$(pwd)/tools/node"
    NODE_DIST="node-$NODE_VERSION-linux-$NODE_ARCH"
    NODE_URL="https://nodejs.org/dist/$NODE_VERSION/$NODE_DIST.tar.xz"
    
    mkdir -p "$NODE_DIR"
    
    if [ ! -f "$NODE_DIR/bin/node" ]; then
        echo "Downloading Node.js ($NODE_VERSION for $NODE_ARCH)..."
        if command -v curl &> /dev/null; then
            curl -L "$NODE_URL" | tar -xJ -C "$NODE_DIR" --strip-components=1
        elif command -v wget &> /dev/null; then
            wget -qO- "$NODE_URL" | tar -xJ -C "$NODE_DIR" --strip-components=1
        else
            echo "Error: Neither curl nor wget found. Cannot download Node.js."
            exit 1
        fi
    fi
    export PATH="$NODE_DIR/bin:$PATH"
else
    echo "System Node.js and NPM found."
fi

echo "Node version: $(node -v)"
echo "NPM version:  $(npm -v)"

# --- Step 1: Clone Repositories ---

# 1.1 Diffusion Pipe
if [ ! -d "$DIFFUSION_PIPE_DIR" ]; then
    echo "[Backend] diffusion-pipe not found in 'vendor/'. Cloning..."
    git clone "$DIFFUSION_PIPE_REPO" "$DIFFUSION_PIPE_DIR"
else
    echo "[Backend] diffusion-pipe directory exists."
fi

# 1.2 ComfyUI
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "[Backend] ComfyUI not found. Cloning..."
    git clone "$COMFYUI_REPO" "$COMFYUI_DIR"
else
    echo "[Backend] ComfyUI directory exists."
fi

# 1.3 HunyuanVideo (New Dependency)
if [ ! -d "$HUNYUAN_DIR" ]; then
    echo "[Backend] HunyuanVideo not found. Cloning..."
    git clone "$HUNYUAN_REPO" "$HUNYUAN_DIR"
else
    echo "[Backend] HunyuanVideo directory exists."
fi


# --- Step 2: Setup Isolated Libraries (Vendor Libs) ---
# We link specific folders here to control exactly what is exposed to PYTHONPATH
if [ ! -d "vendor/libs" ]; then
    mkdir -p "vendor/libs"
fi

# Link 'comfy'
if [ ! -L "vendor/libs/comfy" ]; then
    echo "[System] Linking 'comfy' module..."
    ln -s ../ComfyUI/comfy vendor/libs/comfy
fi

# Link 'hyvideo' (HunyuanVideo)
if [ ! -L "vendor/libs/hyvideo" ]; then
    echo "[System] Linking 'hyvideo' module..."
    ln -s ../HunyuanVideo/hyvideo vendor/libs/hyvideo
fi


# --- Step 3: Install Backend Dependencies ---
echo "[Backend] Checking Python dependencies..."

# FIX: Ensure build tools are up to date for Python 3.12 compatibility
pip install --upgrade pip setuptools wheel

if [ -f "$DIFFUSION_PIPE_DIR/requirements.txt" ]; then
    pip install -q -r "$DIFFUSION_PIPE_DIR/requirements.txt"
fi

if [ -f "$COMFYUI_DIR/requirements.txt" ]; then
    echo "[Backend] Installing ComfyUI requirements..."
    pip install -q -r "$COMFYUI_DIR/requirements.txt"
fi

if [ -f "$HUNYUAN_DIR/requirements.txt" ]; then
    echo "[Backend] Installing HunyuanVideo requirements..."
    
    # FIX: Python 3.12 Issue with old Numpy
    # HunyuanVideo might pin an old version of numpy that fails to build on Py3.12.
    # We remove the numpy requirement from this file to let the system use the 
    # already installed (and compatible) numpy version.
    sed -i '/numpy/d' "$HUNYUAN_DIR/requirements.txt"
    
    pip install -q -r "$HUNYUAN_DIR/requirements.txt"
fi

if [ -f "$DPGUI_REQUIREMENTS" ]; then
    pip install -q -r "$DPGUI_REQUIREMENTS"
fi

# --- Step 4: Setup Frontend ---
echo "[Frontend] Checking React environment..."

if [ ! -d "$FRONTEND_DIR" ]; then
    echo "[Frontend] Initializing new React project..."
    npx --yes create-vite@5.2.0 "$FRONTEND_DIR" --template react
    
    if [ ! -d "$FRONTEND_DIR" ]; then
        echo "Error: Failed to create frontend directory."
        exit 1
    fi
    
    cd "$FRONTEND_DIR"
    echo "[Frontend] Installing dependencies..."
    npm install
    npm install axios react-hook-form
    cd ..
else
    echo "[Frontend] Directory found."
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
         echo "[Frontend] Installing missing dependencies..."
         cd "$FRONTEND_DIR" && npm install && cd ..
    fi
fi

# --- Step 5: Launch Services ---

BACKEND_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "------------------------------------------"
echo "Configuration:"
echo " - Frontend Port: $FRONTEND_PORT"
echo " - Backend Port:  $BACKEND_PORT"

echo "VITE_API_BASE_URL=http://127.0.0.1:$BACKEND_PORT" > "$FRONTEND_DIR/.env.local"

# UPDATE PYTHONPATH:
# Points to 'vendor/libs' where our clean symlinks live.
export PYTHONPATH="$PYTHONPATH:$(pwd)/vendor/libs"
echo "[System] PYTHONPATH updated."

cleanup() {
    echo ""
    echo "Stopping DPGui..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null
    fi
    exit
}

trap cleanup SIGINT

echo "Starting Backend (Hidden)..."
uvicorn "$BACKEND_APP" --host "$HOST" --port "$BACKEND_PORT" > backend.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for Backend to initialize..."
sleep 3

echo "Starting Frontend..."
echo "------------------------------------------"
echo "DPGui is running!"
echo "Backend Logs: backend.log"
echo "Access URL:   http://localhost:$FRONTEND_PORT"
echo "------------------------------------------"

cd "$FRONTEND_DIR"
npm run dev -- --host --port "$FRONTEND_PORT"