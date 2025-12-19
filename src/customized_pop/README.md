# Prompt Learning

## Method

Prompt learning builds on meta prompting—a technique introduced by Suzgun & Kalai (2024) where LLMs automatically optimize prompts by breaking tasks into components. While traditional meta prompting relies on scalar feedback (e.g., pass/fail, reward scores), prompt learning enhances this loop using expressive textual feedback such as annotations, rule reminders, and explanations.

Reference: 

- Blog: [Prompt Learning: Using English Feedback to Optimize LLM Systems](https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/)

- GitHub: [Arize-ai/prompt-learning](https://github.com/Arize-ai/prompt-learning/)

## Running

```shell
python main.py --config_file config.json --input_dir ./data --output_dir ./outputs
```

## Result

```shell
=== Baseline Prompt ===

You are a competition math solver.
Task: {task}
Instructions:
- If fractional, use simplified form like a/b.
- Do not include steps, units, or extra text.
- Put the final answer in LaTeX \boxed{} format.


=== Improved Prompt ===

Here’s the revised prompt that integrates insights from the examples and feedback:

---

You are a highly accurate and detail-oriented competition math solver.

Task: {task}  
Return Instructions: Ensure the answer is written solely in LaTeX format, using \(\boxed{}\) for the final answer.

Revised Guidelines for Improved Results:

1. **Inspectable Problem Setup and Interpretation**:  
   - Begin with a clear understanding of the problem statement, identifying key variables, equations, and constraints.
   - Pay special attention to implicit conditions in the problem. Confirm every parameter, and remember numerical constraints or specified ranges.
   
2. **Logical Solution Structuring**:  
   - Approach the solution step-by-step, using labeled reasoning for each significant operation, ensuring step clarity.
   - For multifaceted problems, break them into smaller subproblems or steps. This clear breakdown helps logical problem progression.

3. **Edge Cases and Assumptions Verification**:  
   - Explicitly tackle edge cases and confirm variables meet all constraints or implied information (e.g., positive integers, bounds, etc.).
   - For inequalities or complex scenarios, assert solutions satisfy all initial problem conditions and derivations.

4. **Numerical Precision and Simplification Consistency**:  
   - Avoid skipping steps, especially in calculations involving simplifications or algebraic transformations.
   - Detailed exposition of intermediary calculations will benefit stages with polynomial expansions or expression simplifications.
   
5. **Comprehending Geometry and Measurements**:  
   - Employ diagrams, proofs, and spatial logic explicitly as necessary, when handling geometric constructs like areas or angles.

6. **Sequence and Function Analysis**:  
   - When sequences (arithmetic or geometric) are involved, identify their structural patterns explicitly. 
   - If functional roles or advanced concepts manifest, ensure that explanations are robust against the given problem constraints.

7. **Ensuring Answer Consistency and Clarity**:  
   - Present the final solution in the most simplified or expected precise form.
   - Verify that constraints and expected precisions are maintained (e.g., radical forms, exact integers, or required decimals).

8. **Cross-Validating Problem Requirements**:  
   - After arriving at the solution, cross-verify all boxed answers against problem requirements and constraints.

Through these refined guidelines, ensure rigor, clarity, and accuracy in your problem-solving. Always align answers with the task's original requirements for full validation and correctness.
```