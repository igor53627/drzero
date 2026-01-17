"""
Dr. Zero training on Modal.com

Usage:
    modal run modal_train.py --action smoke           # Test environment first
    modal run modal_train.py --action prepare-data    # Prepare training data
    modal run modal_train.py --action challenger      # Train challenger model
    
Requirements:
    pip install modal
    modal token new  # authenticate
"""
import modal
import os
import subprocess
import threading

# Modal app and image setup
app = modal.App("dr-zero-training")

# Pin versions explicitly for reproducibility (oracle recommendation)
# sglang 0.5.6 is in the base image; verl 0.3.0.post1 is compatible
image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.6-cu129-amd64")
    .apt_install("git", "curl", "libnuma-dev")
    .pip_install(
        "requests",
        "fastapi",
        "uvicorn[standard]",
        "datasets",
        "pandas",
        "faiss-cpu",
        "chromadb",
        "verl==0.3.0.post1",
        "wandb",
        "sentence-transformers",
    )
    .run_commands(
        "python -m pip uninstall -y flash-attn flash_attn || true",
        "python -m pip install "
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl "
        "--no-deps",
    )
    .add_local_dir(
        ".",
        remote_path="/root/drzero",
        ignore=lambda p: any(x in str(p) for x in [".git", "__pycache__", ".pyc", "node_modules"]),
    )
)

# Volumes for persistent storage
corpus_volume = modal.Volume.from_name("dr-zero-corpus", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("dr-zero-checkpoints", create_if_missing=True)

CORPUS_PATH = "/corpus"
CHECKPOINTS_PATH = "/checkpoints"


def _stream_output(prefix: str, proc: subprocess.Popen):
    """Stream subprocess output to Modal logs."""
    for line in iter(proc.stdout.readline, ""):
        if line:
            print(f"[{prefix}] {line.rstrip()}")


def popen_logged(prefix: str, args: list, env=None) -> subprocess.Popen:
    """Start a subprocess with output streaming to logs."""
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env or os.environ.copy(),
    )
    thread = threading.Thread(target=_stream_output, args=(prefix, proc), daemon=True)
    thread.start()
    return proc


def wait_for_server(url: str, proc: subprocess.Popen = None, timeout: int = 120, interval: int = 5):
    """Poll a server until it responds or timeout. Check proc exit if provided."""
    import time
    import requests
    start = time.time()
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Process exited early: {proc.args}, rc={proc.returncode}")
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(f"Timeout waiting for {url}")


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
def train_challenger(iteration: int = 1, hop_ratio: str = "4321", model_path: str = "Qwen/Qwen3-32B"):
    """Train the proposer/challenger model."""
    import sys
    import torch
    
    os.chdir("/root/drzero")
    
    # Log environment info
    print(f"Python: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"PyTorch: {torch.__version__}")
    
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
    
    # Start retrieval server with logged output
    print("Starting retrieval server...")
    retrieval_proc = popen_logged("retrieval", [
        sys.executable, "search/chromadb_server.py",
        f"--chroma_path={CORPUS_PATH}/chromadb",
        "--collection_name=papers",
        "--port=8000",
        "--embedding_model=intfloat/e5-large-v2",
    ])
    
    # Start sglang inference server with logged output
    print(f"Starting SGLang server with model {model}...")
    sglang_proc = popen_logged("sglang", [
        sys.executable, "-m", "sglang.launch_server",
        f"--model={model}",
        "--port=8001",
        f"--tool-call-parser={tool_parser}",
        "--mem-fraction-static=0.5",
        "--dp-size=4",
        "--tp-size=2",
    ])
    
    # Wait for servers with health checks (embedding model load takes time)
    print("Waiting for retrieval server...")
    wait_for_server("http://127.0.0.1:8000/docs", proc=retrieval_proc, timeout=180)
    print("Retrieval server ready")
    
    print("Waiting for SGLang server (may take 15+ min for model download)...")
    wait_for_server("http://127.0.0.1:8001/health", proc=sglang_proc, timeout=1200)
    print("SGLang server ready")
    
    # Training data from corpus volume
    train_data = f"{CORPUS_PATH}/data/zero_ratio{hop_ratio}.parquet"
    
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-path=/root/drzero/config",
        "--config-name=search_multiturn_grpo",
        f"data.train_files={train_data}",
        f"data.val_files={train_data}",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "data.train_batch_size=256",
        "algorithm.use_kl_in_reward=False",
        "algorithm.adv_estimator=grpo_batch",
        f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.actor.grad_clip=0.1",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "+actor_rollout_ref.actor.micro_batch_size_per_gpu=2",
        "+actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "+actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        "+actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "+critic.forward_micro_batch_size_per_gpu=2",
        "+critic.ppo_micro_batch_size_per_gpu=2",
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
        print("Running verl training command:")
        print(" ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"verl training failed with exit code {result.returncode}")
        return f"Challenger training iteration {iteration} complete"
    finally:
        retrieval_proc.terminate()
        sglang_proc.terminate()
        retrieval_proc.wait()
        sglang_proc.wait()


@app.function(
    image=image,
    gpu="H100:1",
    timeout=30 * 60,  # 30 minutes (first build can be slow)
)
def smoke():
    """Smoke test to verify the environment works before running full training."""
    import sys
    print(f"Python: {sys.version}")
    
    print("Importing torch...")
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}, GPU 0: {torch.cuda.get_device_name(0)}")
    
    print("Importing sglang...")
    import sglang
    print(f"SGLang: {sglang.__version__}")
    
    print("Importing verl...")
    import verl
    print(f"verl: {verl.__version__}")
    
    print("Importing chromadb...")
    import chromadb
    print(f"ChromaDB: {chromadb.__version__}")
    
    print("Importing sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers: OK")
    
    print("\n[OK] All imports successful!")
    return "Smoke test passed"


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
def train_solver(iteration: int = 1, model_path: str = "Qwen/Qwen3-32B"):
    """Train the solver model on synthetic data."""
    import sys
    import torch
    
    os.chdir("/root/drzero")
    
    print(f"Python: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch: {torch.__version__}")
    os.environ["SGLANG_DISABLE_SGL_KERNEL"] = "1"
    
    model = model_path
    if iteration > 1:
        model = f"{CHECKPOINTS_PATH}/solver_iter{iteration-1}_hf"
    
    # Start retrieval server with logged output
    print("Starting retrieval server...")
    retrieval_proc = popen_logged("retrieval", [
        sys.executable, "search/chromadb_server.py",
        f"--chroma_path={CORPUS_PATH}/chromadb",
        "--collection_name=papers",
        "--port=8000",
        "--embedding_model=intfloat/e5-large-v2",
    ])
    
    print("Waiting for retrieval server...")
    wait_for_server("http://127.0.0.1:8000/docs", proc=retrieval_proc, timeout=180)
    
    train_data = f"{CORPUS_PATH}/data/synthetic_iter{iteration}.parquet"
    
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-path=/root/drzero/config",
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
    Generate training parquet directly from ChromaDB.
    Run this before training to prepare the data.
    """
    import subprocess
    import sys
    
    os.chdir("/root/drzero")
    
    data_dir = f"{CORPUS_PATH}/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate training parquet using ChromaDB directly
    print("Generating training parquet from ChromaDB...")
    subprocess.run([
        sys.executable, "process_train.py",
        f"--chroma_path={CORPUS_PATH}/chromadb",
        "--collection_name=papers",
        f"--local_dir={data_dir}",
    ], check=True)
    
    # Commit volume changes
    corpus_volume.commit()
    
    # List generated files
    files = os.listdir(data_dir)
    return f"Data preparation complete. Files: {files}"


# Default model options (use volume paths when available)
MODELS = {
    # Qwen 3 (recommended) - use cached version from volume
    "qwen3-32b": "/corpus/models/Qwen3-32B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    # Qwen 2.5 (fallback)
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    # Ministral (not supported by SGLang 0.5.6)
    "ministral-14b": "mistralai/Ministral-3-14B-Reasoning-2512",
}


@app.local_entrypoint()
def main(
    action: str = "smoke",
    iteration: int = 1,
    corpus_path: str = None,
    model: str = "qwen3-32b",
):
    """
    Main entrypoint for Dr. Zero training on Modal.
    
    Examples:
        # Step 0: Run smoke test to verify environment (ALWAYS DO THIS FIRST)
        modal run modal_train.py --action smoke
        
        # Step 1: Upload your ChromaDB corpus (already done via CLI)
        # modal volume put dr-zero-corpus /path/to/chromadb /chromadb -f
        
        # Step 2: Prepare training data (export ChromaDB -> JSONL -> parquet)
        modal run modal_train.py --action prepare-data
        
        # Step 3: Train challenger with Qwen 14B (default)
        modal run modal_train.py --action challenger --iteration 1
        
        # Train with a specific model (start small to validate!)
        modal run modal_train.py --action challenger --model qwen-3b
        modal run modal_train.py --action challenger --model qwen-7b
        
        # Train solver
        modal run modal_train.py --action solver --iteration 1
    """
    # Resolve model shorthand or use as-is for HuggingFace paths
    model_path = MODELS.get(model, model)
    
    if action == "smoke":
        result = smoke.remote()
        print(result)
    elif action == "upload":
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
        print("Valid actions: smoke, upload, prepare-data, challenger, solver, list-models")
