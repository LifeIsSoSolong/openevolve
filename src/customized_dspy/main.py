from __future__ import annotations

import argparse
from typing import Any, Optional, Tuple
import dspy
import concurrent.futures
from math_verify import parse, verify

import srsly
import os
import logging
from dspy.teleprompt.utils import (
    eval_candidate_program,
    get_program_with_highest_avg_score,
    save_candidate_program,
)

# ANSI escape codes for colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default


logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = "sk-5d2ZcXHg0SSTEB8V6005Dd8f74E14f1e95Ae3c0e205b3f2a"
os.environ["OPENAI_API_BASE"] = "https://api3.apifans.com/v1"


def load_math_dataset(data_path, prompt_key, answer_key, number=100):
    data = srsly.read_jsonl(data_path) if data_path.endswith(
        '.jsonl') else srsly.read_json(data_path)
    data = [dspy.Example(
        **{'prompt': item[prompt_key], 'answer': item[answer_key]}).with_inputs("prompt") for item in data]
    return data[:number]


# Define the SimplestAdapter as before
def format_demos(demos):
    parts = ["Here are examples of your expected behavior.", "<examples>"]
    for i, demo in enumerate(demos, 1):
        parts += [
            f"<example_{i}>",
            "User:",
            demo["prompt"],
            "Assistant:",
            demo["generation"],
            f"</example_{i}>",
        ]
    parts.append("</examples>")
    return "\n".join(parts)


class SimplestAdapter(dspy.Adapter):
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        system_content = signature.instructions or ""
        # if demos:
        #     system_content += "\n" + format_demos(demos)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": inputs["prompt"]},
        ]
        outputs = lm(messages=messages, **lm_kwargs)
        return [{"generation": outputs[0]}]

# https://maximerivest.com/posts/automatic-system-prompt-optimization.html


class WrapperPredictor(dspy.Predict):
    def forward(self, **kwargs):
        adapter = SimplestAdapter()
        with dspy.settings.context(adapter=adapter):
            return super().forward(**kwargs)


class MetaPromptModel:
    def __init__(self, base_lm: dspy.LM, meta_prompt: str):
        self.base_lm = base_lm
        self.meta_prompt = meta_prompt

    def __call__(self, *args, **kwargs):
        """
        MIPROv2 的 GroundedProposer 会大概率这样调用：
            prompt_model(prompt=big_context_string, **lm_kwargs)
        或者有些版本会用 messages=[...].
        我们两个都兼容。
        """
        if "prompt" in kwargs:
            original_prompt = kwargs["prompt"]
            kwargs["prompt"] = (
                f"{self.meta_prompt}\n\n"
                "================ CONTEXT ABOUT THE TASK / PROGRAM / DATA ================\n"
                f"{original_prompt}\n\n"
                "================ NEW INSTRUCTION FOR THE PREDICTOR ================\n"
            )
        elif "messages" in kwargs:
            # 保险起见也适配一下 chat-style 调用
            messages = kwargs["messages"]
            new_messages = [{"role": "system", "content": self.meta_prompt}]
            for m in messages:
                if m.get("role") == "system":
                    # 把原来的 system 当成用户给的“说明”
                    new_messages.append({
                        "role": "user",
                        "content": "Context about the program / data:\n" + m.get("content", "")
                    })
                else:
                    new_messages.append(m)
            kwargs["messages"] = new_messages

        # 统计一下被调用次数也无妨
        return self.base_lm(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # 让外界看起来它“就是一个 LM”
        return getattr(self.base_lm, name)


def verify_func(ex: dspy.Example, pred: dspy.Prediction) -> Tuple[float, Optional[str]]:
    ground_truth = str(getattr(ex, "answer", "")).strip()
    model_output = str(getattr(pred, "generation", "")).strip()

    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"

    def _compute():
        """Helper function to compute score with parsing and verification."""
        parsed_output = parse(model_output, parsing_timeout=None)
        parsed_ground_truth = parse(
            ground_truth_boxed, parsing_timeout=None)
        return verify(parsed_output, parsed_ground_truth, timeout_seconds=None)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_compute)
            ret_score = future.result(timeout=10)  # 10 seconds timeout
    except concurrent.futures.TimeoutError:
        ret_score = 0.0
    except Exception:
        pass

    return (ret_score, None) if ret_score == 1.0 else (0.0, f"mismatch: expected={ground_truth}, got={model_output}")


def metric(ex: dspy.Example, pred: dspy.Prediction, _trace=None):
    score, _feedback = verify_func(ex, pred)
    # DSPy accepts floats or bools; clamp for safety.
    try:
        s = float(score)
    except Exception:
        s = 0.0
    return max(0.0, min(1.0, s))


def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    score, _feedback = verify_func(example, prediction)
    # Do not coerce to int; answers may be non-integer text/LaTeX.
    correct_answer = str(example.get('answer', '')).strip()
    written_solution = example.get('solution', '')

    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."

    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    return dspy.Prediction(score=score, feedback=feedback_text)


def main():
    gpt4o_mini = dspy.LM('openai/gpt-4o-mini', max_tokens=2000)
    gpt4o = dspy.LM('openai/gpt-4o', temperature=1.0, max_tokens=32000)
    # we'll use gpt-4o-mini as the default LM, unless otherwise specified
    dspy.configure(lm=gpt4o_mini)

    trainset = load_math_dataset(
        data_path='/Users/kaiyan/Project/FrontisAgents/CRL/data/MATH/train.json',
        prompt_key='prompt',
        answer_key='answer',
        number=16
    )
    devset = load_math_dataset(
        data_path='/Users/kaiyan/Project/FrontisAgents/CRL/data/MATH/test.json',
        prompt_key='prompt',
        answer_key='answer',
        number=16
    )

    system_prompt = """
    You are a competition math solver.
    Instructions:
    - If fractional, use simplified form like a/b.
    - Do not include steps, units, or extra text.
    - Put the final answer in LaTeX \\boxed{} format.
    """.strip()

    OptimSignature = dspy.Signature(
        "prompt -> generation",
        instructions=system_prompt,
    )

    program = WrapperPredictor(OptimSignature)
    program.set_lm(gpt4o_mini)

    kwargs = dict(num_threads=8, display_progress=True, display_table=5)
    evaluate = dspy.Evaluate(devset=devset, metric=metric, **kwargs)
    evaluate(program=program)

    ####################### GEPA (Disabled) ########################
    # optimizer = dspy.GEPA(
    #     metric=metric_with_feedback,
    #     auto="light",
    #     num_threads=8,
    #     track_stats=True,
    #     reflection_minibatch_size=8,
    #     reflection_lm=gpt4o
    # )
    # optimized_program = optimizer.compile(
    #     program,
    #     trainset=trainset[:48],
    #     valset=trainset[48:],
    # )

    ####################### MIPROv2 ########################
    META_PROMPT = """
    You are a prompt-engineering expert.

    Your job:
    - You are given a *description* of a DSPy predictor, some example inputs/outputs, and possibly tips.
    - You must write a single high-quality *instruction string* to be placed in `signature.instructions`.

    Hard constraints:
    - Output ONLY the instruction text, no JSON, no commentary, no quotes.
    - At most 10 sentences, <= 250 English tokens.
    - The target task: competition-level math question answering.
    - The student model will receive the problem in `prompt` and must output ONLY the final answer in LaTeX \\boxed{} without steps.

    Please write an instruction that:
    - Emphasizes accuracy on math competition problems.
    - Tells the model to think briefly, but NOT to show the reasoning.
    - Forces output to be a single LaTeX \\boxed{...} with simplified fraction format a/b when needed.
    """
    prompt_model = MetaPromptModel(
        base_lm=gpt4o_mini,
        meta_prompt=META_PROMPT,
    )


    def _perform_full_evaluation(
        self,
        trial_num: int,
        adjusted_num_trials: int,
        param_score_dict: dict,
        fully_evaled_param_combos: dict,
        evaluate: dspy.Evaluate,
        valset: list,
        trial_logs: dict,
        total_eval_calls: int,
        score_data,
        best_score: float,
        best_program: Any,
        study: "optuna.Study",
        instruction_candidates: list,
        demo_candidates: list,
    ):
        import optuna

        logger.info(f"===== Trial {trial_num + 1} / {adjusted_num_trials} - Full Evaluation =====")

        # Identify best program to evaluate fully
        highest_mean_program, mean_score, combo_key, params = get_program_with_highest_avg_score(
            param_score_dict, fully_evaled_param_combos
        )
        logger.info(f"Doing full eval on next top averaging program (Avg Score: {mean_score}) from minibatch trials...")
        full_eval_score = eval_candidate_program(len(valset), valset, highest_mean_program, evaluate, self.rng).score
        score_data.append({"score": full_eval_score, "program": highest_mean_program, "full_eval": True})

        # Log full eval as a trial so that optuna can learn from the new results
        trial = optuna.trial.create_trial(
            params=params,
            distributions=self._get_param_distributions(best_program, instruction_candidates, demo_candidates),
            value=full_eval_score,
        )
        study.add_trial(trial)

        # Log full evaluation results
        fully_evaled_param_combos[combo_key] = {
            "program": highest_mean_program,
            "score": full_eval_score,
        }
        total_eval_calls += len(valset)
        trial_logs[trial_num + 1] = {}
        trial_logs[trial_num + 1]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num + 1]["full_eval_program_path"] = save_candidate_program(
            program=highest_mean_program,
            log_dir=self.log_dir,
            trial_num=trial_num + 1,
            note="full_eval",
        )
        trial_logs[trial_num + 1]["full_eval_program"] = highest_mean_program
        trial_logs[trial_num + 1]["full_eval_score"] = full_eval_score

        # Update best score and program if necessary
        if full_eval_score > best_score:
            logger.info(f"{GREEN}New best full eval score!{ENDC} Score: {full_eval_score}")
            best_score = full_eval_score
            best_program = highest_mean_program.deepcopy()
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        trajectory = "[" + full_eval_scores + "]"
        logger.info(f"Full eval scores so far: {trajectory}")
        logger.info(f"Best full score so far: {best_score}")
        logger.info(len(f"===== Full Eval {len(fully_evaled_param_combos) + 1} =====") * "=")
        logger.info("\n")
        
        return best_score, best_program, total_eval_calls

    optimizer = dspy.MIPROv2(metric=metric,
                             auto="medium",
                             num_threads=8,
                             teacher_settings=dict(lm=gpt4o),
                             prompt_model=prompt_model)
    optimized_program = optimizer.compile(
        program,
        trainset=trainset,
        valset=devset,
        num_trials=5,
        max_bootstrapped_demos=0,
        max_labeled_demos=0
    )

    ########################### COPRO (Disabled) ########################
    # optimizer = dspy.COPRO(
    #     prompt_model=gpt4o_mini,
    #     metric=metric,
    #     breadth=4,
    #     depth=4,
    #     init_temperature=1.0,
    #     verbose=True,
    # )
    # eval_kwargs = dict(num_threads=8, display_progress=True, display_table=0)
    # optimized_program = optimizer.compile(
    #     program,
    #     trainset=trainset,
    #     eval_kwargs=eval_kwargs
    # )

    evaluate(program=optimized_program)
    print("Instruction before Optimization:",
          repr(program.signature.instructions))
    print("Instruction after Optimization :", repr(
        optimized_program.signature.instructions))


if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", type=str, required=True)
    # parser.add_argument("--train_file", type=str, default=None)
    # parser.add_argument("--eval_file", type=str, default=None)
    # parser.add_argument("--reward_def", type=str, default=None)
    # parser.add_argument("--input_dir", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, required=True)
    # args = parser.parse_args()
    
