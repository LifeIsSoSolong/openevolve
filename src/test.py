import os
import uuid
import shutil
import requests

DATA_DIR = "/hpc_data/daguan_data/rl_backend"
SRC_DIR = "/hpc_data/zhangkaiyan/agentic-rl/src/customized_pop/data"
SPLITS = ("train.jsonl", "test.jsonl")
RL_URL = "http://10.200.4.4:30022"


def upload(task_id):
    target = os.path.join(DATA_DIR, task_id, "inputs")
    os.makedirs(target, exist_ok=True)

    for name in SPLITS:
        src = os.path.join(SRC_DIR, name)
        dst = os.path.join(target, name)

        if not os.path.exists(src):
            raise SystemExit("missing file: %s" % src)

        shutil.copy(src, dst)


def get_algos():
    url = RL_URL.rstrip("/") + "/meta/algorithms"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    print("algos:", data)
    return data


def submit(task_id, algos):
    cfg = algos["config"]["prompt_learning"]

    payload = {
        "task_id": task_id,
        "dataset_config": {"key": "value"},
        "evaluator_config": {"key": "value"},
        "training_config": {
            "algorithm": "prompt_learning",
            "settings": cfg,
        },
        "workflow_config": {"key": "value"},
    }

    url = RL_URL.rstrip("/") + "/tasks/submit"
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    print("submit resp:", r.json())


if __name__ == "__main__":
    task_id = str(uuid.uuid4())
    print("task_id:", task_id)

    upload(task_id)
    algos = get_algos()
    submit(task_id, algos)