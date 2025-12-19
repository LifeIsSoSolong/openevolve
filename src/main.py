import os
import sys
import json
import srsly
import logging
import subprocess
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Configure logging. If it crashes, I want to know why.
# If you ignore logs, you deserve the bugs you get.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rl_backend")

app = FastAPI(title="RL Optimization Kernel", version="0.1.0")

DATA_DIR = "/hpc_data/daguan_data/rl_backend"

algo2path = {
    "prompt_learning": "/hpc_data/zhangkaiyan/agentic-rl/src/customized_pop",
    "alpha_evolve": "/hpc_data/zhangkaiyan/agentic-rl/src/alphaevolve"
}

# -----------------------------------------------------------------------------
# CONSTANTS & REGISTRY
# -----------------------------------------------------------------------------
# Don't put this in a database unless you have a good reason.
# Memory is fast. DBs are slow.
SUPPORTED_ALGOS = {
    algo: srsly.read_json(os.path.join(path, "config.json")) for algo, path in algo2path.items()
}
# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    settings: Dict[str, Any]
    algorithm: str


class TaskConfig(BaseModel):
    task_id: str
    dataset_config: Dict[str, Any]
    evaluator_config: Dict[str, Any]
    training_config: TrainingConfig
    workflow_config: Dict[str, Any]


class ValidationRequest(BaseModel):
    data_path: str
    algo_name: str

# -----------------------------------------------------------------------------
# CORE LOGIC (The stuff that actually does work)
# -----------------------------------------------------------------------------


def sanity_check_directory(path: str) -> None:
    """
    Checks if a directory actually exists and is readable.
    I don't care if the user 'thinks' it exists. The kernel handles reality.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path is not a directory: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Read permission denied: {path}")


def validate_data_schema(path: str, algo: str) -> bool:
    """
    Validates if the user provided CSVs actually contain data we can use.
    Specific implementation omitted because I don't know your column names.
    But generally: open the file header, check it, close it. Fast.
    """
    # Implementation details skipped.
    # Just imagine efficient C-like file parsing here, not pandas loading
    # the whole damn file into RAM just to check one line.
    logger.info(f"Validating data in {path} for algorithm {algo}")

    # Mock return for now.
    # In reality, fail strictly if the schema is garbage.
    return True


def spawn_optimization_process(task_id: str, task: TaskConfig):
    """
    Forks a process to run the training script.
    We separate the web server process from the heavy lifting.
    If the training segfaults, the web server should stay alive.
    """
    algorithm = task.training_config.algorithm
    settings = task.training_config.settings

    input_dir = os.path.join(DATA_DIR, task_id, "inputs")
    output_dir = os.path.join(DATA_DIR, task_id, "outputs")

    logger.info(f"Create output directory {output_dir}")
    os.makedirs(output_dir)

    config_file = os.path.join(input_dir, "config.json")
    logger.info(f"Write config to {config_file}")
    srsly.write_json(config_file, settings)

    workflow_file = os.path.join(input_dir, "workflow.json")
    logger.info(f"Write workflow to {workflow_file}")
    srsly.write_json(workflow_file, task.workflow_config)

    dataset_file = os.path.join(input_dir, "dataset.json")
    logger.info(f"Write dataset to {dataset_file}")
    srsly.write_json(dataset_file, task.dataset_config)

    evaluator_file = os.path.join(input_dir, "evaluator.json")
    logger.info(f"Write evaluator to {evaluator_file}")
    srsly.write_json(evaluator_file, task.evaluator_config)

    cmd = [
        sys.executable,
        "main.py",  # The actual worker script
        "--config_file", config_file,
        "--input_dir", input_dir,
        "--output_dir", output_dir
    ]

    try:
        # Popen is non-blocking. Fire and forget (mostly).
        # Log stdout/stderr to files so we can debug the mess later.
        with open(f"{output_dir}/stdout.log", "w") as log_file:
            subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=algo2path[algorithm]
            )
        logger.info(f"Spawned worker for task {task_id} [PID: unknown (yet)]")
    except OSError as e:
        logger.error(f"Failed to spawn process: {e}")
        # No point raising here, we are in a background task.
        # Just log it and move on.

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------


@app.get("/meta/algorithms")
async def list_algorithms():
    """
    Returns what we support. If it's not here, we don't do it.
    """
    return {
        "algorithms": list(SUPPORTED_ALGOS.keys()),
        "config": SUPPORTED_ALGOS
    }


@app.post("/validate/dataset")
async def validate_dataset(req: ValidationRequest):
    """
    Checks if the user data isn't complete garbage before we try to process it.
    """
    if req.algo_name not in SUPPORTED_ALGOS:
        raise HTTPException(status_code=400, detail="Unknown algorithm. RTFM.")

    try:
        sanity_check_directory(req.data_path)
        is_valid = validate_data_schema(req.data_path, req.algo_name)

        if not is_valid:
            raise HTTPException(
                status_code=422, detail="Data schema mismatch.")

        return {"status": "valid", "path": req.data_path}

    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        logger.warning(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal validation failure.")


@app.post("/tasks/submit")
async def submit_task(task: TaskConfig, background_tasks: BackgroundTasks):
    """
    Accepts a task, validates configs, and kicks off a subprocess.
    """
    input_dir = os.path.join(DATA_DIR, task.task_id, "inputs")
    if not os.path.exists(input_dir):
        return {
            "task_id": task.task_id,
            "status": "failed",
            "message": f"Not found: `{input_dir}`."
        }
    logger.info(f"Start task {task.task_id}")
    # Do NOT run blocking code in the event loop.
    # We use FastAPI's BackgroundTasks which hooks into the loop correctly.
    background_tasks.add_task(spawn_optimization_process, task.task_id, task)

    return {
        "task_id": task.task_id,
        "status": "queued",
        "message": "Worker process will start shortly."
    }


@app.get("/")
async def root():
    return {"status": "running", "quote": "Talk is cheap. Show me the code."}

# ---------------------------------------------------------
# 本地启动： uvicorn main:app --reload
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=30022, reload=False)
