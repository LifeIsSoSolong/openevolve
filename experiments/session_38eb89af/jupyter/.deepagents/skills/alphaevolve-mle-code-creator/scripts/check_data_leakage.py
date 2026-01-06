#!/usr/bin/env python3
"""
检测 agent.py 中的数据泄露问题。

数据泄露类型：
1. 测试集参与训练：test_df/test_data 被用于 fit/train
2. 测试集参与特征工程：用全量数据（含测试集）做归一化、编码
3. 合并后处理：train 和 test 合并后一起做预处理

Usage:
    python check_data_leakage.py --file agent.py
    python check_data_leakage.py --file agent.py --json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple


@dataclass
class LeakageWarning:
    """数据泄露警告"""
    level: str  # "error" | "warning" | "info"
    type: str   # 泄露类型
    message: str
    line: int
    code_snippet: str
    suggestion: str


@dataclass
class LeakageReport:
    """泄露检测报告"""
    file_path: str
    has_leakage: bool
    errors: List[LeakageWarning] = field(default_factory=list)
    warnings: List[LeakageWarning] = field(default_factory=list)
    info: List[LeakageWarning] = field(default_factory=list)
    
    def add(self, warning: LeakageWarning):
        if warning.level == "error":
            self.errors.append(warning)
            self.has_leakage = True
        elif warning.level == "warning":
            self.warnings.append(warning)
        else:
            self.info.append(warning)


# 常见的测试集变量名模式
TEST_VAR_PATTERNS = [
    r'\btest_df\b', r'\btest_data\b', r'\bdf_test\b', r'\bdata_test\b',
    r'\bX_test\b', r'\by_test\b', r'\btest_set\b', r'\bval_df\b',
    r'\bval_data\b', r'\bvalidation_df\b', r'\bX_val\b', r'\by_val\b',
]

# 常见的训练集变量名模式
TRAIN_VAR_PATTERNS = [
    r'\btrain_df\b', r'\btrain_data\b', r'\bdf_train\b', r'\bdata_train\b',
    r'\bX_train\b', r'\by_train\b', r'\btrain_set\b',
]

# 会导致数据泄露的方法调用（如果在测试集上调用或合并数据上调用）
FITTING_METHODS = [
    'fit', 'fit_transform', 'fit_predict', 'fit_resample',
    'train', 'learn', 'update', 'partial_fit',
]

# 预处理/特征工程方法
PREPROCESSING_METHODS = [
    'fit', 'fit_transform', 'fillna', 'mean', 'std', 'min', 'max',
    'median', 'quantile', 'normalize', 'scale', 'encode',
]

# 合并操作
CONCAT_PATTERNS = [
    r'pd\.concat\s*\(\s*\[.*train.*test.*\]',
    r'pd\.concat\s*\(\s*\[.*test.*train.*\]',
    r'\.append\s*\(.*test',
    r'\.append\s*\(.*train',
]


def read_source(file_path: Path) -> Tuple[str, List[str]]:
    """读取源代码"""
    content = file_path.read_text(encoding="utf-8")
    lines = content.split('\n')
    return content, lines


def find_test_var_usage_in_fit(source: str, lines: List[str]) -> List[LeakageWarning]:
    """检测测试集变量是否参与了 fit/train 操作"""
    warnings = []
    
    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # 跳过注释
        if line_stripped.startswith('#'):
            continue
        
        # 检查是否有 fit 类方法调用
        for method in FITTING_METHODS:
            if f'.{method}(' in line:
                # 检查这行或附近是否使用了测试集变量
                context = '\n'.join(lines[max(0, i-3):min(len(lines), i+2)])
                
                for pattern in TEST_VAR_PATTERNS:
                    if re.search(pattern, line):
                        # 测试集变量直接出现在 fit 调用中
                        warnings.append(LeakageWarning(
                            level="error",
                            type="test_in_fit",
                            message=f"测试集变量参与了 {method}() 操作，这会导致数据泄露",
                            line=i,
                            code_snippet=line_stripped,
                            suggestion=f"确保 {method}() 只在训练集上调用，测试集应该用 transform() 处理"
                        ))
                        break
    
    return warnings


def find_concat_before_preprocessing(source: str, lines: List[str]) -> List[LeakageWarning]:
    """检测是否先合并数据再做预处理"""
    warnings = []
    
    # 查找 concat 操作
    concat_lines = []
    for i, line in enumerate(lines, 1):
        if 'concat' in line.lower() or 'append' in line.lower():
            # 检查是否同时包含 train 和 test
            has_train = any(re.search(p, line, re.IGNORECASE) for p in [r'train', r'trn'])
            has_test = any(re.search(p, line, re.IGNORECASE) for p in [r'test', r'val', r'tst'])
            
            if has_train and has_test:
                concat_lines.append((i, line.strip()))
    
    # 对每个合并操作，检查后续是否有 fit_transform
    for concat_line, concat_code in concat_lines:
        # 检查后续 50 行内是否有 fit_transform
        for j in range(concat_line, min(len(lines), concat_line + 50)):
            if 'fit_transform' in lines[j-1] or 'fit(' in lines[j-1]:
                warnings.append(LeakageWarning(
                    level="error",
                    type="concat_before_fit",
                    message="先合并 train/test 数据再做 fit_transform，这会导致测试集信息泄露到训练过程",
                    line=concat_line,
                    code_snippet=concat_code,
                    suggestion="应该先在训练集上 fit，再分别对训练集和测试集 transform"
                ))
                break
    
    return warnings


def find_global_statistics_leakage(source: str, lines: List[str]) -> List[LeakageWarning]:
    """检测使用全局统计量导致的泄露"""
    warnings = []
    
    # 常见的统计量计算模式
    stat_patterns = [
        (r'\.mean\(\)', 'mean'),
        (r'\.std\(\)', 'std'),
        (r'\.min\(\)', 'min'),
        (r'\.max\(\)', 'max'),
        (r'\.median\(\)', 'median'),
        (r'\.quantile\(', 'quantile'),
    ]
    
    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # 跳过注释
        if line_stripped.startswith('#'):
            continue
        
        # 检查是否在合并的数据上计算统计量
        # 例如: all_data['col'].mean() 或 df['col'].mean() 其中 df 是合并后的数据
        for pattern, stat_name in stat_patterns:
            if re.search(pattern, line):
                # 检查变量名是否暗示是合并数据
                if any(name in line.lower() for name in ['all_data', 'full_data', 'combined', 'merged', 'total_df']):
                    warnings.append(LeakageWarning(
                        level="warning",
                        type="global_statistics",
                        message=f"在合并数据上计算 {stat_name}() 可能导致测试集信息泄露",
                        line=i,
                        code_snippet=line_stripped,
                        suggestion="应该只使用训练集的统计量，然后应用到测试集"
                    ))
    
    return warnings


def find_target_in_features(source: str, lines: List[str]) -> List[LeakageWarning]:
    """检测目标变量是否被错误地用作特征"""
    warnings = []
    
    # 常见目标列名
    target_names = ['target', 'label', 'y', 'class', 'outcome', 'prediction']
    
    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        line_lower = line.lower()
        
        # 检查是否有明显的目标泄露模式
        # 例如: X = df.drop('target', axis=1) 之后又把 target 相关的列加回去
        if 'drop' in line_lower and any(t in line_lower for t in target_names):
            # 这是正确的模式，跳过
            continue
        
        # 检查特征中是否包含目标相关的列
        # 例如: features = ['col1', 'col2', 'target_lag1']
        if 'feature' in line_lower and '=' in line:
            for target in target_names:
                if target in line_lower and 'lag' not in line_lower and 'shift' not in line_lower:
                    warnings.append(LeakageWarning(
                        level="warning",
                        type="target_in_features",
                        message=f"特征列表中可能包含目标变量 '{target}'，请确认这不是数据泄露",
                        line=i,
                        code_snippet=line_stripped,
                        suggestion="确保目标变量不在训练特征中，除非是经过正确处理的滞后特征"
                    ))
    
    return warnings


def find_test_transform_before_fit(source: str, lines: List[str]) -> List[LeakageWarning]:
    """检测是否在 fit 之前就对测试集做了依赖训练集的 transform"""
    warnings = []
    
    # 查找 scaler/encoder 等对象的创建和使用顺序
    # 这个检测比较复杂，需要追踪变量的使用顺序
    
    # 简化版：检查是否有 test.transform() 出现在 train.fit() 之前
    fit_line = None
    for i, line in enumerate(lines, 1):
        if '.fit(' in line and any(re.search(p, line) for p in TRAIN_VAR_PATTERNS):
            fit_line = i
            break
    
    if fit_line:
        # 检查 fit 之前是否有对 test 的 transform
        for i, line in enumerate(lines[:fit_line], 1):
            if '.transform(' in line:
                for pattern in TEST_VAR_PATTERNS:
                    if re.search(pattern, line):
                        warnings.append(LeakageWarning(
                            level="error",
                            type="transform_before_fit",
                            message="在 fit() 之前就对测试集调用了 transform()，逻辑顺序错误",
                            line=i,
                            code_snippet=line.strip(),
                            suggestion="正确顺序：先 fit(train) -> transform(train) -> transform(test)"
                        ))
    
    return warnings


def check_evolve_block_safety(source: str, lines: List[str]) -> List[LeakageWarning]:
    """检查 EVOLVE-BLOCK 内的代码是否安全"""
    warnings = []
    
    in_evolve_block = False
    evolve_start = 0
    
    for i, line in enumerate(lines, 1):
        if 'EVOLVE-BLOCK-START' in line:
            in_evolve_block = True
            evolve_start = i
            continue
        if 'EVOLVE-BLOCK-END' in line:
            in_evolve_block = False
            continue
        
        if in_evolve_block:
            # 在进化块内，检查是否有直接读取测试集的操作
            for pattern in TEST_VAR_PATTERNS:
                if re.search(pattern, line):
                    # 检查是否是 fit 操作
                    if any(f'.{method}(' in line for method in FITTING_METHODS):
                        warnings.append(LeakageWarning(
                            level="error",
                            type="evolve_block_leakage",
                            message="EVOLVE-BLOCK 内对测试集调用了训练方法，进化可能产生泄露代码",
                            line=i,
                            code_snippet=line.strip(),
                            suggestion="EVOLVE-BLOCK 内的代码应该只处理训练数据，测试数据的处理应在块外"
                        ))
    
    return warnings


def analyze_file(file_path: Path) -> LeakageReport:
    """分析文件中的数据泄露问题"""
    report = LeakageReport(file_path=str(file_path), has_leakage=False)
    
    try:
        source, lines = read_source(file_path)
    except Exception as e:
        report.add(LeakageWarning(
            level="error",
            type="read_error",
            message=f"无法读取文件: {e}",
            line=0,
            code_snippet="",
            suggestion="检查文件是否存在且编码正确"
        ))
        return report
    
    # 运行各项检测
    checks = [
        find_test_var_usage_in_fit,
        find_concat_before_preprocessing,
        find_global_statistics_leakage,
        find_target_in_features,
        find_test_transform_before_fit,
        check_evolve_block_safety,
    ]
    
    for check_func in checks:
        try:
            warnings = check_func(source, lines)
            for w in warnings:
                report.add(w)
        except Exception as e:
            report.add(LeakageWarning(
                level="info",
                type="check_error",
                message=f"检测 {check_func.__name__} 时出错: {e}",
                line=0,
                code_snippet="",
                suggestion=""
            ))
    
    return report


def format_report(report: LeakageReport, use_json: bool = False) -> str:
    """格式化报告输出"""
    if use_json:
        return json.dumps({
            "file_path": report.file_path,
            "has_leakage": report.has_leakage,
            "errors": [vars(e) for e in report.errors],
            "warnings": [vars(w) for w in report.warnings],
            "info": [vars(i) for i in report.info],
        }, indent=2, ensure_ascii=False)
    
    lines = []
    lines.append("=" * 60)
    lines.append("数据泄露检测报告")
    lines.append("=" * 60)
    lines.append(f"文件: {report.file_path}")
    lines.append("")
    
    if report.has_leakage:
        lines.append("❌ 检测到数据泄露问题！")
    else:
        if report.warnings:
            lines.append("⚠️ 未检测到明确泄露，但有潜在风险")
        else:
            lines.append("✅ 未检测到数据泄露问题")
    
    lines.append("")
    
    if report.errors:
        lines.append(f"🚨 错误 ({len(report.errors)}):")
        for e in report.errors:
            lines.append(f"  [{e.type}] 第 {e.line} 行")
            lines.append(f"    {e.message}")
            lines.append(f"    代码: {e.code_snippet[:80]}...")
            lines.append(f"    建议: {e.suggestion}")
            lines.append("")
    
    if report.warnings:
        lines.append(f"⚠️ 警告 ({len(report.warnings)}):")
        for w in report.warnings:
            lines.append(f"  [{w.type}] 第 {w.line} 行")
            lines.append(f"    {w.message}")
            lines.append(f"    建议: {w.suggestion}")
            lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="检测 agent.py 中的数据泄露问题")
    parser.add_argument("--file", required=True, help="要检测的文件路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[ERROR] 文件不存在: {file_path}")
        sys.exit(1)
    
    report = analyze_file(file_path)
    print(format_report(report, args.json))
    
    # 返回状态码
    if report.has_leakage:
        sys.exit(1)
    elif report.warnings:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
