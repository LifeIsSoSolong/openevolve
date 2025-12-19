from mathruler.grader import extract_boxed_content, grade_answer

def compute_reward(response: str, ground_truth: str, metadata: dict={}) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


if __name__=="__main__":
    print(compute_reward("$f(-2)+f(-1)+f(0)=\\frac{3(-2)-2}{-2-2}+\\frac{3(-1)-2}{-1-2}+\\frac{3(0)-2}{0-2}=\\frac{-8}{-4}+\\frac{-5}{-3}+\\frac{-2}{-2}=2+\\frac{5}{3}+1=\\boxed{\\frac{14}{4}}$","\\frac{14}{3}"))