"""
Upload ChromaDB corpus to Modal volume.

Usage:
    modal run modal_upload.py --local-path /Users/user/pse/000-research/iO-papers/combined_db
"""
import modal
from modal.mount import Mount
import os

app = modal.App("dr-zero-upload")

image = modal.Image.debian_slim(python_version="3.11")

corpus_volume = modal.Volume.from_name("dr-zero-corpus", create_if_missing=True)
CORPUS_PATH = "/corpus"
MOUNT_PATH = "/mnt/local_db"


@app.local_entrypoint()
def main(local_path: str):
    """Upload local ChromaDB to Modal volume."""
    
    print(f"Uploading {local_path} to Modal volume...")
    
    # Create mount from local directory
    local_mount = Mount().add_local_dir(local_path, remote_path=MOUNT_PATH)
    
    # Create the upload function with the mount
    @app.function(
        image=image, 
        volumes={CORPUS_PATH: corpus_volume}, 
        mounts=[local_mount],
        timeout=3600
    )
    def do_upload():
        import shutil
        import os
        
        dest = f"{CORPUS_PATH}/chromadb"
        
        # Remove existing if present
        if os.path.exists(dest):
            shutil.rmtree(dest)
            print(f"Removed existing {dest}")
        
        print(f"Copying from {MOUNT_PATH} to {dest}...")
        shutil.copytree(MOUNT_PATH, dest)
        corpus_volume.commit()
        
        # Count files and size
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(dest):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
                file_count += 1
        
        return f"Uploaded {file_count} files ({total_size / 1024 / 1024:.1f} MB) to {dest}"
    
    result = do_upload.remote()
    print(result)
