# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""
import time
import dataclasses
import json
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
    ]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import os

            import wandb

            settings = None
            if config and config["trainer"].get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])
            entity = os.environ.get("WANDB_ENTITY", None)
            wandb.init(project=project_name, name=experiment_name, entity=entity, config=config, settings=settings)
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearml" in self.logger:
            self.logger["clearml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()
        if "file" in self.logger:
            self.logger["file"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.close()

import json
from collections.abc import Mapping, Sequence
import torch

def to_jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # PyTorch Tensor
    if torch is not None and isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return obj.tolist()

    # dict / Mapping
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [to_jsonable(x) for x in obj]

    return str(obj)

# class FileLogger:
#     def __init__(self, project_name: str, experiment_name: str):
#         self.project_name = project_name
#         self.experiment_name = experiment_name

#         self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
#         if self.filepath is None:
#             root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
#             directory = os.path.join(root_path, self.project_name)
#             os.makedirs(directory, exist_ok=True)
#             self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
#             print(f"Creating file logger at {self.filepath}")

#     def log(self, data, step):
#         data = {"step": step, "data": to_jsonable(data)}
#         with open(self.filepath, "a") as f:
#             f.write(json.dumps(data) + "\n")

#     def finish(self):
#         pass

class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        # VERL_FILE_LOGGER_PATH 视为【目录】
        log_root = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if log_root is None:
            # 没显式指定就用 VERL_FILE_LOGGER_ROOT / project / experiment
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            log_root = os.path.join(root_path, self.project_name, self.experiment_name)

        os.makedirs(log_root, exist_ok=True)

        # 事件日志：一行一个 JSON
        self.filepath = os.path.join(log_root, "events.jsonl")
        # 状态文件：始终只保留当前状态
        self.status_path = os.path.join(log_root, "status.json")

        # 缓存最新状态，方便 finish() 写 completed
        self._last_status: dict[str, Any] | None = None
        self._last_step: int | None = None

        print(f"Creating file logger at {self.filepath}")

        # 初始化一个默认的 running 状态
        init_status = {
            "state": "running",
            "current_step": 0,
            "total_steps": None,
            "last_update": int(time.time()),
            "error": None,
            "extra": {
                "final_score": None,
            },
        }
        self._write_status(init_status)

    # ---------- 事件格式化：写 events.jsonl ----------

    def _build_event_record(self, data: dict[str, Any], step: int) -> dict[str, Any] | None:

        if not isinstance(data, dict):
            return None

        ts = int(time.time())

        def _to_float(v):
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        # ---------- 1) eval / val 情况 ----------
        val_reward = data.get("val/reward", None)
        if val_reward is not None or any(k.startswith("val/") for k in data.keys()):
            acc = _to_float(val_reward)
            return {
                "step": step,
                "type": "eval",
                "accuracy": acc,
                "timestamp": ts,
            }

        # ---------- 2) train 情况 ----------
        # 规则：有 "training/reward" 或 "training/loss" 或 key 以 "training/" 开头就认为是 train
        train_reward = _to_float(data.get("training/reward", None))

        loss = data.get("actor/entropy_loss", None)

        loss = _to_float(loss)

        if train_reward is not None or loss is not None or any(
            k.startswith("training/") for k in data.keys()
        ):
            return {
                "step": step,
                "type": "train",
                "loss": loss,
                "reward": train_reward,
                "timestamp": ts,
            }

        # 既不是我们关心的 eval / train，就不写入
        return None

    # ---------- 状态格式化 & 写 status.json ----------

    def _build_status_record(
        self,
        data: dict[str, Any],
        step: int,
        state: str = "running",
        error: str | None = None,
    ) -> dict[str, Any]:
        ts = int(time.time())

        # current_step 优先用 training/global_step，否则退回 step
        current_step_raw = data.get("training/global_step", step)
        try:
            current_step = int(current_step_raw)
        except (TypeError, ValueError):
            current_step = int(step)
        
        # total_steps 如果有就转成 int，没有就 None
        # total_steps_raw = data.get("training/total_steps", None)
        # try:
        #     total_steps = int(total_steps_raw) if total_steps_raw is not None else None
        # except (TypeError, ValueError):
        #     total_steps = None
        env_total = os.getenv("VERL_TOTAL_STEPS", None)
        # error 优先使用参数，其次尝试从 data["error"] 里拿
        if error is None and "error" in data:
            err_val = data["error"]
            error = str(err_val) if err_val is not None else None
        total_steps = int(env_total)
        def _to_float(v):
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        # 默认沿用上一次 status.json 里的 final_score
        final_score = None
        if self._last_status is not None:
            last_extra = self._last_status.get("extra") or {}
            final_score = last_extra.get("final_score")

        # 如果当前这一步有 val/reward，就尝试更新为新的分数
        if "val/reward" in data:
            new_score = _to_float(data.get("val/reward"))
            if new_score is not None:
                final_score = new_score        
        return {
            "state": state,
            "current_step": current_step,
            "total_steps": total_steps,
            "last_update": ts,
            "error": error,
            "extra": {
                "final_score": final_score,
            },
        }

    def _write_status(self, status: dict[str, Any]) -> None:
        """覆盖写入 status.json"""
        try:
            with open(self.status_path, "w") as sf:
                json.dump(status, sf)
        except Exception as e:
            # 状态写失败不应该影响训练流程，这里只打印一下
            print(f"WARNING: failed to write status.json to {self.status_path}: {e}")

        self._last_status = status
        self._last_step = status.get("current_step")

    # ---------- 对外接口：log / finish ----------

    def log(self, data, step):
        # 1) 写 events.jsonl（可选：只对 train / eval 写）
        event = self._build_event_record(data if isinstance(data, dict) else {}, step)
        if event is not None:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(event) + "\n")

        # 2) 更新 status.json（每次 log 都刷新最新状态）
        status = self._build_status_record(
            data if isinstance(data, dict) else {},
            step,
            state="running",
        )
        self._write_status(status)

    def finish(self):
        # 训练正常结束时，把状态标成 completed
        if self._last_status is not None:
            status = dict(self._last_status)
            status["state"] = "completed"
            status["last_update"] = int(time.time())
        else:
            # 如果从未 log 过，也给一个兜底状态
            status = {
                "state": "completed",
                "current_step": 0,
                "total_steps": None,
                "last_update": int(time.time()),
                "error": None,
            }
        self._write_status(status)





class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def __init__(self):
        import logging
        import re

        self.logger = logging.getLogger(__name__)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(
            r"[^/\w.\- :]"
        )  # Allowed: slashes, alphanumerics, underscores, periods, dashes, colons, and spaces.

    def log(self, data, step):
        import mlflow

        def sanitize_key(key):
            # First replace @ with _at_ for backward compatibility
            sanitized = key.replace("@", "_at_")
            # Then replace any other invalid characters with _
            sanitized = self._invalid_chars_pattern.sub("_", sanitized)
            if sanitized != key:
                self.logger.warning(
                    "[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized
                )
            return sanitized

        results = {sanitize_key(k): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """Log samples to wandb as a table"""

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        headers = ["step", "input", "output", "score"]

        swanlab_row_list = [[step, *sample] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
