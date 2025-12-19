import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    def __init__(self, root: str = "outputs"):
        self.root = Path(root)
        self.events_path = self.root / "events.jsonl"
        self.status_path = self.root / "status.json"
        self.checkpoints_dir = self.root / "checkpoints"
        self.final_dir = self.root / "final_result"
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 status
        if not self.status_path.exists():
            self.update_status(
                state="running",
                current_step=0,
                total_steps=None,
                error=None,
            )

    def _now(self) -> int:
        return int(time.time())

    def log_event(self, data: Dict[str, Any]) -> None:
        """追加一行到 events.jsonl"""
        data = dict(data)
        if "timestamp" not in data:
            data["timestamp"] = self._now()
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def update_status(
        self,
        state: str,
        current_step: int,
        total_steps: Optional[int],
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """覆盖写 status.json（确保是原子写）"""
        payload = {
            "state": state,           # running, failed, completed
            "current_step": current_step,
            "total_steps": total_steps,
            "last_update": self._now(),
            "error": error,
        }
        if extra:
            payload.update(extra)

        tmp_path = self.status_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.status_path)

    def save_checkpoint(self, step: int, prompt_state: Dict[str, Any]) -> None:
        """简单版本：只存 prompt_state.json（你后面可以扩展加 pytorch.bin）"""
        step_dir = self.checkpoints_dir / f"step-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        out_path = step_dir / "prompt_state.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(prompt_state, f, ensure_ascii=False, indent=2)

    def save_final_model(self, best_state: Dict[str, Any]) -> None:
        out_path = self.final_dir / "best_model.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(best_state, f, ensure_ascii=False, indent=2)
