#!/bin/bash
# Full startup script - handles everything

# Install system deps
apt-get update
apt-get install -y wget curl git libmagic1 python3-venv

# Clone wrapper with retry
if [ ! -d /root/comfyui-api-wrapper ]; then
    for i in 1 2 3 4 5; do
        echo "Cloning repo (attempt $i)..."
        git clone https://github.com/ImmarKarim/comfyui-api-wrapper.git /root/comfyui-api-wrapper && break
        echo "Clone failed, retrying in 5s..."
        sleep 5
    done
fi

# Verify clone succeeded
if [ ! -f /root/comfyui-api-wrapper/requirements.txt ]; then
    echo "ERROR: Repo not cloned properly. Exiting."
    exit 1
fi

# Setup venv if missing
if [ ! -d /root/comfyui-api-wrapper/.venv ]; then
    python3 -m venv /root/comfyui-api-wrapper/.venv
fi

# Install dependencies with retry
install_deps() {
    /root/comfyui-api-wrapper/.venv/bin/pip install --upgrade pip
    /root/comfyui-api-wrapper/.venv/bin/pip install -r /root/comfyui-api-wrapper/requirements.txt
}

if [ ! -f /root/comfyui-api-wrapper/.venv/bin/uvicorn ]; then
    for i in 1 2 3 4 5; do
        echo "Installing dependencies (attempt $i)..."
        install_deps && break
        echo "Install failed, retrying in 5s..."
        sleep 5
    done
fi

# Final check
if [ ! -f /root/comfyui-api-wrapper/.venv/bin/uvicorn ]; then
    echo "ERROR: Failed to install dependencies. Exiting."
    exit 1
fi

# Start Ollama
OLLAMA_MODELS=/workspace/cache/ollama ollama serve > /ComfyUI/ollama.log 2>&1 &

# Start Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=6969 &

# Start ComfyUI
cd /ComfyUI
/ComfyUI/venv/bin/python3 main.py --listen 0.0.0.0 --port 8188 &

# Wait for ComfyUI
echo "Waiting for ComfyUI..."
until curl -s http://127.0.0.1:8188/history > /dev/null 2>&1; do sleep 2; done
echo "ComfyUI ready!"

# Start wrapper with auto-restart
cd /root/comfyui-api-wrapper
while true; do
    /root/comfyui-api-wrapper/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a /wrapper.log
    echo "Wrapper crashed, restarting..."
    sleep 3
done

