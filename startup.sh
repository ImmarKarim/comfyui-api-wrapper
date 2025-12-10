#!/bin/bash

# Start ComfyUI
cd /ComfyUI
/ComfyUI/venv/bin/python3 main.py --listen 0.0.0.0 --port 8188 &

echo "Waiting for ComfyUI..."
until curl -s http://127.0.0.1:8188/history > /dev/null 2>&1; do
    sleep 2
done
echo "ComfyUI ready!"

# Start wrapper with auto-restart
cd /root/comfyui-api-wrapper
. .venv/bin/activate

while true; do
    uvicorn main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a /wrapper.log
    echo "Wrapper crashed, restarting..."
    sleep 3
done
