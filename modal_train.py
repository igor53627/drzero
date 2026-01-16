"""
Dr. Zero training on Modal.com

Usage:
    modal run modal_train.py --corpus-path /path/to/chromadb
    
Requirements:
    pip install modal
    modal token new  # authenticate
"""
import modal
import os

# Modal app and image setup
app = modal.App("dr-zero-training")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "libnuma-dev", "build-essential", "ninja-build", "cmake", "python3-dev")
    .run_commands(
        "pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio",
    )
    .pip_install(
        "transformers>=4.40.0",
        "datasets",
        "pandas",
        "numpy",
        "faiss-cpu",
        "chromadb",
        "fastapi",
        "uvicorn",
        "sglang[all]>=0.4.0",
        "verl>=0.3.0",
        "wandb",
        "huggingface_hub",
        "sentence-transformers",
    )
    .run_commands(
        "cd /root && git clone https://github.com/igor53627/drzero.git",
    )
)

# Volumes for persistent storage
corpus_volume = modal.Volume.from_name("dr-zero-corpus", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("dr-zero-checkpoints", create_if_missing=True)

CORPUS_PATH = "/corpus"
CHECKPOINTS_PATH = "/checkpoints"


def wait_for_server(url: str, timeout: int = 120, interval: int = 5):
    """Poll a server until it responds or timeout."""
    import time
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


@app.function(
    image=image,
    gpu="H100:8",
    timeout=60 * 60 * 24,  # 24 hours
    volumes={
        CORPUS_PATH: corpus_volume,
        CHECKPOINTS_PATH: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train_challenger(iteration: int = 1, hop_ratio: str = "4321", model_path: str = "Qwen/Qwen2.5-14B-Instruct"):
    """Train the proposer/challenger model."""
    import subprocess
    import sys
    import torch
    
    os.chdir("/root/drzero")
    
    # Log CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Disable SGLang kernel to avoid compilation issues (can re-enable later for perf)
    os.environ["SGLANG_DISABLE_SGL_KERNEL"] = "1"
    
    model = model_path
    
    # Determine tool-call-parser based on model
    if "mistral" in model.lower() or "ministral" in model.lower():
        tool_parser = "mistral"
    elif "qwen" in model.lower():
        tool_parser = "qwen25"
    else:
        tool_parser = "mistral"  # default
    
    # Start retrieval server (using ChromaDB with matching embedding model)
    retrieval_proc = subprocess.Popen([
        sys.executable, "search/chromadb_server.py",
        f"--chroma_path={CORPUS_PATH}/chromadb",
        "--collection_name=papers",
        "--port=8000",
        "--embedding_model=intfloat/e5-large-v2",
    ])
    
    # Start sglang inference server
    sglang_proc = subprocess.Popen([
        sys.executable, "-m", "sglang.launch_server",
        f"--model={model}",
        "--port=8001",
        f"--tool-call-parser={tool_parser}",
        "--mem-fraction-static=0.25",
        "--dp-size=4",
        "--tp-size=2",
        "--log-level=error"
    ])
    
    # Wait for servers with health checks
    print("Waiting for retrieval server...")
    if not wait_for_server("http://127.0.0.1:8000/docs", timeout=60):
        raise RuntimeError("Retrieval server failed to start")
    print("Retrieval server ready")
    
    print("Waiting for SGLang server...")
    if not wait_for_server("http://127.0.0.1:8001/health", timeout=300):
        raise RuntimeError("SGLang server failed to start")
    print("SGLang server ready")
    
    # Training data from corpus volume
    train_data = f"{CORPUS_PATH}/data/zero_ratio{hop_ratio}.parquet"
    val_data = f"{CORPUS_PATH}/data/test.parquet"
    
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-path=./config",
        "--config-name=search_multiturn_grpo",
        f"data.train_files={train_data}",
        f"data.val_files={val_data}",
        "data.train_batch_size=256",
        "algorithm.use_kl_in_reward=False",
        "algorithm.adv_estimator=grpo_batch",
        f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.actor.grad_clip=0.1",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.multi_turn.tool_config_path=./config/search_tool_config.yaml",
        "reward_model.reward_manager=batch",
        "custom_reward_function.name=compute_challenger_score_batch",
        "custom_reward_function.path=verl/custom_reward/reward_function.py",
        f"custom_reward_function.reward_kwargs.model_name={model}",
        "custom_reward_function.reward_kwargs.base_url=http://127.0.0.1:8001",
        f"trainer.experiment_name=challenger_iter{iteration}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.save_freq=25",
        f"trainer.default_local_dir={CHECKPOINTS_PATH}/challenger_iter{iteration}",
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return f"Challenger training iteration {iteration} complete"
    finally:
        retrieval_proc.terminate()
        sglang_proc.terminate()
        retrieval_proc.wait()
        sglang_proc.wait()


@app.function(
    image=image,
    gpu="H100:8",
    timeout=60 * 60 * 24,
    volumes={
        CORPUS_PATH: corpus_volume,
        CHECKPOINTS_PATH: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train_solver(iteration: int = 1, model_path: str = "Qwen/Qwen2.5-14B-Instruct"):
    """Train the solver model on synthetic data."""
    import subprocess
    import sys
    import torch
    
    os.chdir("/root/drzero")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    os.environ["SGLANG_DISABLE_SGL_KERNEL"] = "1"
    
    model = model_path
    if iteration > 1:
        model = f"{CHECKPOINTS_PATH}/solver_iter{iteration-1}_hf"
    
    # Start retrieval server
    retrieval_proc = subprocess.Popen([
        sys.executable, "search/chromadb_server.py",
        f"--chroma_path={CORPUS_PATH}/chromadb",
        "--collection_name=papers",
        "--port=8000",
        "--embedding_model=intfloat/e5-large-v2",
    ])
    
    print("Waiting for retrieval server...")
    if not wait_for_server("http://127.0.0.1:8000/docs", timeout=60):
        raise RuntimeError("Retrieval server failed to start")
    
    train_data = f"{CORPUS_PATH}/data/synthetic_iter{iteration}.parquet"
    
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-path=./config",
        "--config-name=search_multiturn_grpo",
        f"data.train_files={train_data}",
        "data.train_batch_size=256",
        "algorithm.adv_estimator=grpo_batch",
        f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "trainer.n_gpus_per_node=8",
        f"trainer.default_local_dir={CHECKPOINTS_PATH}/solver_iter{iteration}",
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return f"Solver training iteration {iteration} complete"
    finally:
        retrieval_proc.terminate()
        retrieval_proc.wait()


@app.function(image=image, volumes={CORPUS_PATH: corpus_volume})
def upload_corpus(local_chromadb_path: str):
    """Upload your local ChromaDB to Modal volume."""
    import shutil
    
    dest = f"{CORPUS_PATH}/chromadb"
    shutil.copytree(local_chromadb_path, dest)
    corpus_volume.commit()
    
    return f"Uploaded ChromaDB to {dest}"


@app.function(
    image=image,
    timeout=60 * 60,  # 1 hour
    volumes={CORPUS_PATH: corpus_volume},
)
def prepare_data(hop_ratio: str = "4321"):
    """
    Export ChromaDB to JSONL and generate training parquet.
    Run this before training to prepare the data.
    """
    import subprocess
    import sys
    
    os.chdir("/root/drzero")
    
    jsonl_path = f"{CORPUS_PATH}/wiki-18.jsonl"
    data_dir = f"{CORPUS_PATH}/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Export ChromaDB to JSONL (if not already done)
    if not os.path.exists(jsonl_path):
        print("Exporting ChromaDB to JSONL...")
        subprocess.run([
            sys.executable, "scripts/export_chromadb_to_jsonl.py",
            f"--chroma_path={CORPUS_PATH}/chromadb",
            "--collection=papers",
            f"--out={jsonl_path}",
        ], check=True)
    else:
        print(f"JSONL already exists at {jsonl_path}")
    
    # Step 2: Generate training parquet
    print("Generating training parquet...")
    subprocess.run([
        sys.executable, "process_train.py",
        f"--corpus_dir={jsonl_path}",
        f"--local_dir={data_dir}",
    ], check=True)
    
    # Commit volume changes
    corpus_volume.commit()
    
    # List generated files
    files = os.listdir(data_dir)
    return f"Data preparation complete. Files: {files}"


# Default model options
MODELS = {
    "ministral-14b": "mistralai/Ministral-3-14B-Reasoning-2512",
    "ministral-8b": "mistralai/Ministral-3-8B-Reasoning-2512", 
    "ministral-3b": "mistralai/Ministral-3-3B-Reasoning-2512",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
}


@app.local_entrypoint()
def main(
    action: str = "train",
    iteration: int = 1,
    corpus_path: str = None,
    model: str = "qwen-14b",
):
    """
    Main entrypoint for Dr. Zero training on Modal.
    
    Examples:
        # Step 1: Upload your ChromaDB corpus (already done via CLI)
        # modal volume put dr-zero-corpus /path/to/chromadb /chromadb -f
        
        # Step 2: Prepare training data (export ChromaDB -> JSONL -> parquet)
        modal run modal_train.py --action prepare-data
        
        # Step 3: Train challenger with Qwen 14B (default)
        modal run modal_train.py --action challenger --iteration 1
        
        # Train with a specific model
        modal run modal_train.py --action challenger --model qwen-7b
        modal run modal_train.py --action challenger --model qwen-3b
        
        # Train solver
        modal run modal_train.py --action solver --iteration 1
    """
    # Resolve model shorthand or use as-is for HuggingFace paths
    model_path = MODELS.get(model, model)
    
    if action == "upload":
        if not corpus_path:
            raise ValueError("--corpus-path required for upload")
        result = upload_corpus.remote(corpus_path)
        print(result)
    elif action == "prepare-data":
        result = prepare_data.remote()
        print(result)
    elif action == "challenger":
        result = train_challenger.remote(iteration=iteration, model_path=model_path)
        print(result)
    elif action == "solver":
        result = train_solver.remote(iteration=iteration, model_path=model_path)
        print(result)
    elif action == "list-models":
        print("Available model shortcuts:")
        for name, path in MODELS.items():
            print(f"  {name}: {path}")
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: upload, prepare-data, challenger, solver, list-models")
