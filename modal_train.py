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
    .apt_install("git", "curl", "libnuma-dev")
    .pip_install(
        "torch>=2.2.0",
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
        "cd /root && git clone https://github.com/facebookresearch/drzero.git",
    )
    .add_local_dir("/Users/user/sources/facebook/drzero/search", "/root/drzero/search")
)

# Volumes for persistent storage
corpus_volume = modal.Volume.from_name("dr-zero-corpus", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("dr-zero-checkpoints", create_if_missing=True)

CORPUS_PATH = "/corpus"
CHECKPOINTS_PATH = "/checkpoints"


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
def train_challenger(iteration: int = 1, hop_ratio: str = "4321", model_path: str = "mistralai/Ministral-3-14B-Reasoning-2512"):
    """Train the proposer/challenger model."""
    import subprocess
    import sys
    
    os.chdir("/root/drzero")
    
    model = model_path
    
    # Determine tool-call-parser based on model
    if "mistral" in model.lower() or "ministral" in model.lower():
        tool_parser = "mistral"
    elif "qwen" in model.lower():
        tool_parser = "qwen25"
    else:
        tool_parser = "mistral"  # default
    
    # Start retrieval server (using ChromaDB)
    retrieval_proc = subprocess.Popen([
        sys.executable, "search/chromadb_server.py",
        f"--chroma_path={CORPUS_PATH}/chromadb",
        "--collection_name=papers",
        "--port=8000"
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
    
    import time
    time.sleep(60)  # Wait for servers to start
    
    # Run training
    train_data = f"./data/zero_ratio{hop_ratio}.parquet"
    val_data = "./data/test.parquet"
    
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
    
    result = subprocess.run(cmd, check=True)
    
    # Cleanup
    retrieval_proc.terminate()
    sglang_proc.terminate()
    
    return f"Challenger training iteration {iteration} complete"


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
def train_solver(iteration: int = 1, model_path: str = "mistralai/Ministral-3-14B-Reasoning-2512"):
    """Train the solver model on synthetic data."""
    import subprocess
    import sys
    
    os.chdir("/root/drzero")
    
    model = model_path
    if iteration > 1:
        model = f"{CHECKPOINTS_PATH}/solver_iter{iteration-1}_hf"
    
    # Start retrieval server
    retrieval_proc = subprocess.Popen([
        sys.executable, "search/chromadb_server.py",
        f"--chroma_path={CORPUS_PATH}/chromadb",
        "--collection_name=papers",
        "--port=8000"
    ])
    
    import time
    time.sleep(30)
    
    train_data = f"./data/synthetic_iter{iteration}.parquet"
    
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
    
    result = subprocess.run(cmd, check=True)
    retrieval_proc.terminate()
    
    return f"Solver training iteration {iteration} complete"


@app.function(image=image, volumes={CORPUS_PATH: corpus_volume})
def upload_corpus(local_chromadb_path: str):
    """Upload your local ChromaDB to Modal volume."""
    import shutil
    
    dest = f"{CORPUS_PATH}/chromadb"
    shutil.copytree(local_chromadb_path, dest)
    corpus_volume.commit()
    
    return f"Uploaded ChromaDB to {dest}"


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
    model: str = "ministral-14b",
):
    """
    Main entrypoint for Dr. Zero training on Modal.
    
    Examples:
        # First, upload your ChromaDB corpus
        modal run modal_train.py --action upload --corpus-path /Users/user/pse/000-research/iO-papers/full_db
        
        # Train challenger with default model (Ministral 3 14B)
        modal run modal_train.py --action challenger --iteration 1
        
        # Train with a specific model
        modal run modal_train.py --action challenger --model ministral-8b
        modal run modal_train.py --action challenger --model qwen-14b
        
        # Use a custom HuggingFace model path
        modal run modal_train.py --action challenger --model mistralai/Mistral-7B-Instruct-v0.3
        
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
        print("Valid actions: upload, challenger, solver, list-models")
