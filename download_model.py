"""
download_model.py
-----------------
Optional: fetch the embedding model (all-MiniLM-L6-v2, ~90 MB) into model/ once, so the
app runs offline. Without it, the app downloads the model from Hugging Face the first
time a CSV is uploaded.

    python download_model.py
"""

import os
import shutil

from huggingface_hub import snapshot_download

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

if __name__ == "__main__":
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    snapshot_download(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_dir=MODEL_DIR,
        ignore_patterns=["onnx/*", "openvino/*", "*.h5", "*.ot", "*.msgpack",
                         "pytorch_model.bin"],
    )
    shutil.rmtree(os.path.join(MODEL_DIR, ".cache"), ignore_errors=True)
    print(f"model saved to {MODEL_DIR}")
