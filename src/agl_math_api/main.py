# Copyright (c) Microsoft. All rights reserved.

# """训练 API：通过配置文件、数据文件和奖励定义进行训练

# 这是一个通用的训练接口，接受配置文件、训练数据、评估数据和奖励定义文件作为输入。

# 奖励定义文件必须包含一个名为 compute_reward 的函数，严格遵循以下格式：
#     def compute_reward(prediction: str, ground_truth: str, metadata: dict) -> float:
#         ...

# Usage:
#     python train_api.py \
#         --config_file /inputs/config.json \
#         --train_file /inputs/train.jsonl \
#         --eval_file /inputs/test.jsonl \
#         --reward_def /inputs/reward_definition.py \
#         --output_dir /outputs
# """

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import agentlightning as agl

agl.configure_logger()
logger = logging.getLogger(__name__)


def load_reward_function(reward_def_path: str) -> Callable[[str, str, Dict[str, Any]], float]:
    """从 Python 文件动态加载奖励函数。
    
    奖励函数必须严格遵循以下格式：
    def compute_reward(prediction: str, ground_truth: str, metadata: dict) -> float:
    
    Args:
        reward_def_path: 奖励定义文件的绝对路径
        
    Returns:
        奖励函数，接受 (prediction, ground_truth, metadata) 并返回 float
        
    Raises:
        FileNotFoundError: 如果奖励定义文件不存在
        ValueError: 如果找不到 compute_reward 函数或函数签名不匹配
    """
    import inspect
    
    reward_def_path = Path(reward_def_path).resolve()
    
    if not reward_def_path.exists():
        raise FileNotFoundError(f"奖励定义文件不存在: {reward_def_path}")
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location("reward_module", reward_def_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载奖励定义模块: {reward_def_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 必须查找名为 compute_reward 的函数
    if not hasattr(module, "compute_reward"):
        raise ValueError(
            f"在 {reward_def_path} 中找不到 'compute_reward' 函数。"
            "请确保文件中定义了以下格式的函数：\n"
            "def compute_reward(prediction: str, ground_truth: str, metadata: dict) -> float:"
        )
    
    reward_func = getattr(module, "compute_reward")
    
    if not callable(reward_func):
        raise ValueError(f"在 {reward_def_path} 中，'compute_reward' 不是一个可调用对象")
    
    # 验证函数签名
    sig = inspect.signature(reward_func)
    params = list(sig.parameters.keys())
    
    # 检查参数数量（至少需要 3 个参数：prediction, ground_truth, metadata）
    if len(params) < 3:
        raise ValueError(
            f"'compute_reward' 函数签名不正确。"
            f"期望: compute_reward(prediction: str, ground_truth: str, metadata: dict) -> float\n"
            f"实际: {sig}"
        )
    
    # 检查参数名称（允许前三个参数名为 prediction/response, ground_truth/answer, metadata）
    expected_params = ["prediction", "ground_truth", "metadata"]
    param_names = params[:3]
    
    # 允许一些常见的变体名称
    param_mapping = {
        "prediction": ["prediction", "response", "answer", "generated_answer"],
        "ground_truth": ["ground_truth", "answer", "expected", "target"],
        "metadata": ["metadata", "meta", "task", "context"],
    }
    
    # 验证参数名称（允许变体）
    warnings = []
    for i, (expected, actual) in enumerate(zip(expected_params, param_names)):
        if actual not in param_mapping[expected]:
            warnings.append(
                f"参数 {i+1} 名称 '{actual}' 不是标准名称 '{expected}' "
                f"(允许的变体: {param_mapping[expected]})"
            )
    
    if warnings:
        logger.warning("函数签名参数名称警告：")
        for warning in warnings:
            logger.warning(f"  - {warning}")
        logger.warning(f"函数签名: {sig}")
        logger.warning("将继续使用该函数，但请确保参数顺序正确")
    
    logger.info(f"成功加载奖励函数 'compute_reward'，签名: {sig}")
    
    return reward_func


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载数据。
    
    Args:
        file_path: JSONL 文件路径（绝对路径）
        
    Returns:
        数据列表
    """
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过第 {line_num} 行（JSON 解析错误）: {e}")
                    continue
    
    return data


def load_config(config_file: str) -> Dict[str, Any]:
    """从 JSON 文件加载训练配置。
    
    Args:
        config_file: 配置文件路径（绝对路径）
        
    Returns:
        配置字典
    """
    config_file = Path(config_file).resolve()
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def _calculate_total_steps(config: Dict[str, Any], train_data: List[Dict[str, Any]]) -> Optional[int]:
    """根据配置和数据计算总训练步数。
    
    Args:
        config: 训练配置字典
        train_data: 训练数据列表
        
    Returns:
        总训练步数，如果无法计算则返回 None
    """
    trainer_config = config.get("trainer", {})
    data_config = config.get("data", {})
    
    # 如果配置中直接指定了 total_training_steps，直接使用
    if "total_training_steps" in trainer_config:
        return int(trainer_config["total_training_steps"])
    
    # 否则根据 total_epochs 和数据集大小计算
    total_epochs = trainer_config.get("total_epochs")
    if total_epochs is None:
        return None
    
    # 获取训练批次大小
    train_batch_size = data_config.get("train_batch_size", 1)
    if train_batch_size <= 0:
        train_batch_size = 1
    
    # 计算每个 epoch 的步数
    dataset_size = len(train_data)
    if dataset_size == 0:
        return None
    
    steps_per_epoch = (dataset_size + train_batch_size - 1) // train_batch_size  # 向上取整
    
    # 总步数 = epoch 数 * 每个 epoch 的步数
    total_steps = total_epochs * steps_per_epoch
    
    logger.info(
        f"计算总训练步数: total_epochs={total_epochs}, "
        f"dataset_size={dataset_size}, train_batch_size={train_batch_size}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}"
    )
    
    return total_steps


class GenericAgent(agl.LitAgent[Dict[str, Any]]):
    """通用智能体，使用动态加载的奖励函数。
    
    这个智能体可以处理各种任务，任务数据格式应包含：
    - messages: 消息列表，问题在 messages[0]["content"]
    - ground_truth: 正确答案
    并使用动态加载的奖励函数进行评估。
    
    也兼容旧格式（problem 和 answer 字段）。
    """
    
    def __init__(
        self,
        reward_func: Callable[[str, str, Dict[str, Any]], float],
        trained_agents: Optional[str] = None,
        val_temperature: Optional[float] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        """初始化通用智能体。
        
        Args:
            reward_func: 奖励函数，接受 (prediction, ground_truth, metadata) 并返回 float
            trained_agents: 已训练的智能体路径
            val_temperature: 验证时的温度参数
            prompt_template: 可选的提示词模板，使用 {problem} 作为占位符
        """
        super().__init__(trained_agents=trained_agents)
        self.reward_func = reward_func
        self.val_temperature = val_temperature
        self.prompt_template = prompt_template or (
            "Solve the following problem. Provide your answer clearly.\n\n"
            "Problem: {problem}\n\nSolution:"
        )
    
    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        """执行单轮任务。
        
        Args:
            task: 任务字典，应包含 'messages' 和 'ground_truth' 字段
                - messages: 消息列表，问题在 messages[0]["content"]
                - ground_truth: 正确答案
            resources: 资源字典，包含 'main_llm' (LLM 资源)
            rollout: Rollout 元数据
            
        Returns:
            奖励分数，如果出错则返回 None
        """
        # 从新格式中提取问题：messages[0]["content"]
        messages = task.get("messages", [])
        if messages and isinstance(messages, list) and len(messages) > 0:
            problem = messages[0].get("content", "")
        else:
            # 兼容旧格式：如果 messages 不存在，尝试从 problem 字段获取
            problem = task.get("problem", "")
        
        # 从新格式中提取答案：ground_truth
        ground_truth = task.get("ground_truth", "")
        # 兼容旧格式：如果 ground_truth 不存在，尝试从 answer 字段获取
        if not ground_truth:
            ground_truth = task.get("answer", "")
        
        if not problem:
            logger.error(f"[Rollout {rollout.rollout_id}] 任务中缺少问题（messages[0][\"content\"] 或 problem 字段）")
            return None
        
        if not ground_truth:
            logger.warning(f"[Rollout {rollout.rollout_id}] 任务中缺少答案（ground_truth 或 answer 字段），将使用空字符串")
            ground_truth = ""
        
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        
        # 构建提示词
        prompt = self.prompt_template.format(problem=problem)
        
        rollout_id = rollout.rollout_id
        logger.info(f"[Rollout {rollout_id}] Problem: {problem[:100]}...")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")
        
        try:
            # 调用 LLM 生成答案
            from langchain.chat_models import init_chat_model
            
            temperature = (
                self.val_temperature
                if self.val_temperature is not None and rollout.mode == "validation"
                else llm.sampling_parameters.get("temperature", 0.7)
            )
            
            # 获取 VERL proxy 的 endpoint
            endpoint = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
            
            chat_model = init_chat_model(
                llm.model,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=temperature,
                max_retries=0,
                max_tokens=2048,
            )
            
            from langchain_core.messages import HumanMessage
            
            response = chat_model.invoke([HumanMessage(content=prompt)])
            generated_answer = response.content if hasattr(response, "content") else str(response)
            
            logger.info(f"[Rollout {rollout_id}] Generated Answer: {generated_answer[:200]}...")
            
            # 构建 metadata 字典，包含任务的其他信息和 rollout 元数据
            metadata: Dict[str, Any] = {
                "task": task.copy(),  # 包含任务的完整信息
                "rollout_id": rollout_id,
                "mode": rollout.mode,
                "problem": problem,
            }
            
            # 使用动态加载的奖励函数评估答案
            # 函数签名: compute_reward(prediction: str, ground_truth: str, metadata: dict) -> float
            reward = self.reward_func(generated_answer, ground_truth, metadata)
            logger.info(f"[Rollout {rollout_id}] Reward: {reward}")
            
            return reward
            
        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Error during rollout: {e}")
            return None


def train(
    config: Dict[str, Any],
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    reward_func: Callable[[str, str, Dict[str, Any]], float],
    output_dir: str,
    active_agent: Optional[str] = None,
) -> None:
    """训练智能体。
    
    Args:
        config: 训练配置字典
        train_data: 训练数据列表
        val_data: 验证数据列表
        reward_func: 奖励函数，接受 (prediction, ground_truth, metadata) 并返回 float
        output_dir: 输出目录（绝对路径）
        active_agent: 可选的智能体名称
    """
    # 确保输出目录存在
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 checkpoints 子目录
    checkpoints_dir = output_dir_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"输出目录: {output_dir_path}")
    logger.info(f"Checkpoints 目录: {checkpoints_dir}")
    
    # 更新配置中的输出路径和 logger 配置
    trainer_cfg = config.setdefault("trainer", {})
    # 默认使用 ["console", "file"]，如果用户没配置的话
    logger_list = trainer_cfg.get("logger")
    if not logger_list:
        logger_list = ["console", "file"]
        trainer_cfg["logger"] = logger_list
    
    # 设置保存路径（如果配置支持）
    trainer_cfg["default_local_dir"] = str(checkpoints_dir)
    
    # 计算总步数（给状态记录用，可选）
    total_steps = _calculate_total_steps(config, train_data)
    if total_steps is not None:
        # 存入环境变量，给 FileLogger 使用
        os.environ["VERL_TOTAL_STEPS"] = str(total_steps)   
    # # 通过自定义 loggers.py 替换 VERL 的 FileLogger
    # # 注意：如果使用 sitecustomize 做全局 patch，这里会是幂等的（内部有防重复标记）
    # from loggers import patch_file_logger
    # patch_file_logger(output_root=output_dir_path, total_steps=total_steps)
    
    # 创建智能体
    agent = GenericAgent(reward_func=reward_func)
    
    # 创建算法和训练器
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(
        n_runners=10,
        algorithm=algorithm,
        adapter={"agent_match": active_agent},
    )
    
    logger.info(f"Adapter agent match acknowledged: {trainer.adapter.agent_match}")
    logger.info(f"加载训练数据: {len(train_data)} 条")
    logger.info(f"加载验证数据: {len(val_data)} 条")
    
    # 开始训练
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main() -> None:
    """主函数，解析参数并运行训练。"""
    parser = argparse.ArgumentParser(
        description="训练 API：通过配置文件、数据文件和奖励定义进行训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="训练配置文件路径（绝对路径，JSON 格式）",
    )
    
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="训练数据文件路径（绝对路径，JSONL 格式）",
    )
    
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="评估数据文件路径（绝对路径，JSONL 格式）",
    )
    
    parser.add_argument(
        "--reward_def",
        type=str,
        required=True,
        help="奖励定义文件路径（绝对路径，Python 文件）",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录路径（绝对路径），checkpoints 将保存在 output_dir/checkpoints 中",
    )
    
    parser.add_argument(
        "--active-agent",
        type=str,
        default=None,
        help="覆盖智能体名称（默认：根据配置自动生成）",
    )
    
    args = parser.parse_args()
    
    # 验证所有路径都是绝对路径
    paths = {
        "config_file": args.config_file,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "reward_def": args.reward_def,
        "output_dir": args.output_dir,
    }
    
    for name, path in paths.items():
        if not os.path.isabs(path):
            logger.warning(f"{name} 不是绝对路径: {path}，将转换为绝对路径")
            paths[name] = os.path.abspath(path)
    
    logger.info("=" * 60)
    logger.info("训练 API 启动")
    logger.info("=" * 60)
    logger.info(f"配置文件: {paths['config_file']}")
    logger.info(f"训练文件: {paths['train_file']}")
    logger.info(f"评估文件: {paths['eval_file']}")
    logger.info(f"奖励定义: {paths['reward_def']}")
    logger.info(f"输出目录: {paths['output_dir']}")
    logger.info("=" * 60)
    
    # 加载配置
    logger.info("加载配置文件...")
    config = load_config(paths["config_file"])
    
    # 更新配置中的数据文件路径（使用绝对路径）
    if "data" not in config:
        config["data"] = {}
    config["data"]["train_files"] = paths["train_file"]
    config["data"]["val_files"] = paths["eval_file"]
    
    # 加载数据
    logger.info("加载训练数据...")
    train_data = load_jsonl(paths["train_file"])
    logger.info(f"训练数据: {len(train_data)} 条")
    
    logger.info("加载评估数据...")
    val_data = load_jsonl(paths["eval_file"])
    logger.info(f"评估数据: {len(val_data)} 条")
    
    # 加载奖励函数
    logger.info("加载奖励函数...")
    reward_func = load_reward_function(paths["reward_def"])
    
    # 开始训练
    logger.info("开始训练...")
    train(
        config=config,
        train_data=train_data,
        val_data=val_data,
        reward_func=reward_func,
        output_dir=paths["output_dir"],
        active_agent=args.active_agent,
    )
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
