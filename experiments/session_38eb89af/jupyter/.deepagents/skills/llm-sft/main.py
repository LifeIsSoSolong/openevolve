#!/usr/bin/env python3
"""
将 JSON 配置文件转换为 YAML 格式并运行 SFT 训练
"""

import json
import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import subprocess
import time


def parse_args():
    """解析命令行参数"""

    parser = argparse.ArgumentParser(
        description="将 JSON 配置文件转换为 YAML 格式并运行 SFT 训练"
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="config.json 的绝对路径"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="输出目录的绝对路径"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录的绝对路径（用于查找 agent.json 和训练数据文件）",
    )
    return parser.parse_args()


def load_json_config(json_path: str) -> dict:
    """加载 JSON 配置文件"""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def convert_to_alpaca_format(input_file: Path, output_file: Path):
    """将原始 JSONL 格式转换为 Alpaca 格式

    转换规则：
    - instruction: 对应 messages[0].content
    - input: 空字符串 ""
    - output: 对应 ground_truth
    """
    converted_count = 0
    error_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # 解析原始 JSON 对象
                data = json.loads(line)

                # 提取 instruction (messages[0].content)
                if 'messages' not in data or not isinstance(data['messages'], list) or len(data['messages']) == 0:
                    print(f"警告: 第 {line_num} 行缺少 messages 字段或 messages 为空，跳过")
                    error_count += 1
                    continue

                instruction = data['messages'][0].get('content', '')
                if not instruction:
                    print(f"警告: 第 {line_num} 行 messages[0].content 为空，跳过")
                    error_count += 1
                    continue

                # 提取 output (ground_truth)
                output = data.get('ground_truth', '')

                # 构建 Alpaca 格式的数据
                alpaca_data = {
                    "instruction": instruction,
                    "input": "",
                    "output": output
                }

                # 写入转换后的 JSON 行
                f_out.write(json.dumps(alpaca_data, ensure_ascii=False) + '\n')
                converted_count += 1

            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行 JSON 解析失败: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"错误: 第 {line_num} 行处理失败: {e}")
                error_count += 1
                continue

    print(f"数据转换完成: 成功转换 {converted_count} 条，失败 {error_count} 条")
    return converted_count > 0


def process_config(config: dict, agent_config: dict, input_dir: Path, output_dir: Path, train_data_file: Optional[Path] = None) -> dict:
    """处理配置：添加模型路径、转换路径为绝对路径"""

    processed_config = config.copy()

    # 从 agent.json 读取模型名称，拼接模型路径
    model_name = agent_config.get("model_name")
    if not model_name:
        raise ValueError("agent.json 中缺少 model_name 字段")

    model_base_path = Path(config["model_base_path"])  # 这里默认携带，用户不可以修改
    model_path = model_base_path / model_name
    processed_config["model_name_or_path"] = str(model_path)

    # "use_deepspeed" 默认为true，用户可以选择关闭；deepspeed_file 为默认的，用户不可以修改；
    if "use_deepspeed" in config and "deepspeed_file" in config:
        processed_config["deepspeed"] = str(
            input_dir / config["deepspeed_file"])

    # 使用命令行传入的 output_dir（已经是绝对路径）
    processed_config["output_dir"] = str(output_dir)

    processed_config["events.jsonl"] = str(output_dir / config["events.jsonl"])
    processed_config["status.jsonl"] = str(output_dir / config["status.jsonl"])

    # 如果指定了训练数据文件的绝对路径，进行格式转换并使用转换后的文件
    if train_data_file is not None:
        train_data_path = Path(train_data_file).resolve()
        if not train_data_path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {train_data_path}")

        # 创建临时转换后的文件路径（在同一目录下）
        temp_converted_file = train_data_path.parent / f"train_converted_tmp.jsonl"

        # 转换为 Alpaca 格式
        print(f"正在转换训练数据格式...")
        print(f"原始文件: {train_data_path}")
        success = convert_to_alpaca_format(
            train_data_path, temp_converted_file)

        if not success:
            raise ValueError(f"训练数据格式转换失败，请检查原始文件格式")

        # 如果未设置 dataset_dir，设置为文件所在目录（用于兼容性）
        if "dataset_dir" not in processed_config:
            processed_config["dataset_dir"] = str(temp_converted_file.parent)
        # 确保格式化格式设置为 alpaca（因为转换后的格式是 Alpaca 格式）
        # 注意：LLaMA-Factory 可能需要在 dataset_info.json 中配置，但直接使用文件路径时也会自动识别
        print(f"使用转换后的训练数据文件: {temp_converted_file}")
        if "dataset_dir" in processed_config:
            print(f"数据集目录: {processed_config['dataset_dir']}")

        processed_config["dataset"] = "train"
    return processed_config


def convert_json_to_yaml(config: dict) -> str:
    """将 JSON 配置转换为 YAML 格式字符串"""
    try:
        import yaml
    except ImportError:
        raise ImportError("需要安装 PyYAML 库: pip install pyyaml")

    # 删除多余的参数，否则参数解析不通过
    config.pop("deepspeed_file")
    config.pop("model_base_path")
    config.pop("use_deepspeed")
    config.pop("n_gpus_per_node")

    # 转换为 YAML 格式，使用更易读的格式
    yaml_str = yaml.dump(config,
                         default_flow_style=False,
                         allow_unicode=True,
                         sort_keys=False,
                         indent=2)

    return yaml_str


def _update_training_status(status_path: Path, new_status: str):
    """只更新 training.status 里的 status 字段"""
    if not status_path.exists():
        print(f"[WARN] 状态文件不存在: {status_path}")
        return

    with status_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data["status"] = new_status

    with status_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_training(yaml_config_path: Path, output_dir: Path):
    """运行训练命令"""

    # 构建训练命令

    cmd = ["llamafactory-cli", "train", str(yaml_config_path)]

    # cmd = ["ls","-all"]

    print(f"运行训练命令: {' '.join(cmd)}")
    print(f"YAML文件: {yaml_config_path} ")

    status_path = output_dir / "training.status"

    # time.sleep(180)

    # 运行训练
    env = os.environ.copy()
    print("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "ALL"))

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=False,   # 关键：不要直接抛异常
        )

        if result.returncode == 0:
            _update_training_status(status_path, "completed")
        else:
            _update_training_status(status_path, "failed")
            sys.exit(result.returncode)

    except Exception as e:
        # 包括 OSError / KeyboardInterrupt 等
        _update_training_status(status_path, "failed")
        raise


def get_free_gpus(mem_threshold=100):
    """获取空闲的GPU的编号"""

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits"
    ]
    out = subprocess.check_output(cmd).decode().strip().splitlines()

    free = []
    for line in out:
        idx, mem = line.split(",")
        if int(mem) < mem_threshold:
            free.append(int(idx))
    return free


def get_gpu_info() -> List[Tuple[int, str, int, int, int]]:
    """
    获取 GPU 信息。

    Returns:
        List of (index, name, memory_used_MB, memory_total_MB, utilization_percent)
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)

        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                index = int(parts[0])
                name = parts[1]
                mem_used = int(parts[2])
                mem_total = int(parts[3])
                util = int(parts[4])
                gpu_info.append((index, name, mem_used, mem_total, util))

        return gpu_info

    except subprocess.CalledProcessError as e:
        print(f"❌ 无法获取 GPU 信息: {e}", file=sys.stderr)
        print("   请确保 nvidia-smi 可用且系统有 NVIDIA GPU", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 解析 GPU 信息时出错: {e}", file=sys.stderr)
        sys.exit(1)


def is_gpu_available(mem_used: int, mem_total: int, util: int) -> bool:
    """
    判断 GPU 是否可用。

    标准：利用率 < 20% 且内存使用 < 10%
    """
    mem_usage_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 100
    return util < 20 and mem_usage_percent < 10


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 转换为 Path 对象
    config_file_path = Path(args.config_file).resolve()
    output_dir_path = Path(args.output_dir).resolve()
    input_dir_path = Path(args.input_dir).resolve()

    # 获取项目所在路径
    project_root = Path(__file__).resolve().parent.parent
    print("project_root", project_root)

    # agent.json 路径
    agent_config_path = input_dir_path / "agent.json"

    # 在 input_dir 中查找训练数据文件
    train_data_path = input_dir_path / "train.jsonl"
    try:
        # 验证文件存在
        if not config_file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file_path}")
        if not agent_config_path.exists():
            raise FileNotFoundError(f"agent.json 文件不存在: {agent_config_path}")

        # 加载配置文件
        print(f"加载配置文件: {config_file_path}")
        config = load_json_config(config_file_path)

        # 加载 agent 配置
        print(f"加载 agent 配置: {agent_config_path}")
        agent_config = load_json_config(agent_config_path)

        # 如果找到训练数据文件，显示信息
        if train_data_path:
            print(f"在输入目录中找到训练数据文件: {train_data_path}")
        else:
            print(
                "未在输入目录中找到以 'train' 开头的训练数据文件，将使用 config.json 中的 dataset 配置"
            )

        try:
            gpu_info = get_gpu_info()

            if not gpu_info:
                print("❌ 未检测到任何 GPU")
                sys.exit(1)

            free = [idx for idx, _, mem_used, mem_total, util in gpu_info
                    if is_gpu_available(mem_used, mem_total, util)]
            assert len(free) >= config.get("n_gpus_per_node", 4), "GPU 不够"

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, free[:config.get("n_gpus_per_node", 4)]))
            os.environ = env

        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)

        # 处理配置：添加模型路径、转换路径为绝对路径
        print("处理配置路径...")
        processed_config = process_config(
            config, agent_config, input_dir_path, output_dir_path, train_data_path
        )
        print("processed_config", processed_config)

        print(f"模型路径: {processed_config.get('model_name_or_path')}")
        print(f"数据集: {processed_config.get('dataset')}")
        print(f"DeepSpeed 配置: {processed_config.get('deepspeed')}")
        print(f"输出目录: {processed_config.get('output_dir')}")

        # 转换为 YAML
        print("将 JSON 配置转换为 YAML 格式...")
        yaml_content = convert_json_to_yaml(processed_config)

        # 创建临时 YAML 文件（放在 input_dir 中）
        temp_yaml = input_dir_path / "config_temp.yaml"
        with open(temp_yaml, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        print(f"已创建临时 YAML 文件: {temp_yaml}")

        # 运行训练
        run_training(temp_yaml, output_dir_path)

        # 训练完成后可以选择删除临时文件（可选）
        # temp_yaml.unlink()

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
