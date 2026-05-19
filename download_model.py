import os
from huggingface_hub import snapshot_download

LOCAL_DIR_CV = '/workspace/xtts_luganda'
LOCAL_DIR_UG = '/workspace/xtts_uganda'
MARKER = '/workspace/.downloaded_models'

os.makedirs(LOCAL_DIR_CV, exist_ok=True)
os.makedirs(LOCAL_DIR_UG, exist_ok=True)

if os.path.exists(MARKER):
    print("✅ Models already downloaded, skipping.")
else:
    print("⬇️  Downloading models...")
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError("HF_TOKEN env var not set! Add it in RunPod endpoint settings.")

    # Download cv model
    snapshot_download(
        repo_id='reuben256/xtts-cv',
        local_dir=LOCAL_DIR_CV,
        allow_patterns=['best_model.pth', 'config.json'],
        token=token
    )
    # Download ug-lang model
    snapshot_download(
        repo_id='reuben256/xtts-ug-lang',
        local_dir=LOCAL_DIR_UG,
        allow_patterns=['model.pth', 'config.json'],
        token=token
    )
    # Download vocab
    snapshot_download(
        repo_id='coqui/XTTS-v2',
        local_dir=LOCAL_DIR_CV,
        allow_patterns=['vocab.json'],
    )
    snapshot_download(
        repo_id='coqui/XTTS-v2',
        local_dir=LOCAL_DIR_UG,
        allow_patterns=['vocab.json'],
    )
    open(MARKER, 'w').close()
    print("✅ Download complete!")
