#!/bin/bash
# Full startup script - handles everything

# Install system deps
apt-get update
apt-get install -y wget curl git libmagic1

# Clone wrapper if missing
if [ ! -d /root/comfyui-api-wrapper ]; then
    git clone https://github.com/ImmarKarim/comfyui-api-wrapper.git /root/comfyui-api-wrapper
fi

# Setup venv if missing
if [ ! -d /root/comfyui-api-wrapper/.venv ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv -p 3.12 /root/comfyui-api-wrapper/.venv
    . /root/comfyui-api-wrapper/.venv/bin/activate
    uv pip install -r /root/comfyui-api-wrapper/requirements.txt
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
. .venv/bin/activate
while true; do
    uvicorn main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a /wrapper.log
    echo "Wrapper crashed, restarting..."
    sleep 3
done

