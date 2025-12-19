bean和iron任务暂时结束，接下来我们需要构建并跑通一个类似的新任务：experiments/press01

# 任务背景
## 任务目标
使用openevolve优化生成符合黄继妍风格和质量的新闻稿的 prompt_generate_press_base ，这个prompt是优化的对象

## 黄继妍风格质量新闻稿撰写
1. 黄继妍是一个有名的专家记者
2. 假设已经有一个generate_press_agent，它的输入是 prompt_generate_press_base ,一份采访稿内容interview_context, 和 采访类型interview_type，输出是一份新闻稿
3. generate_press_agent 固定不变，我们希望用openevolve优化prompt，使得能够生成符合黄继妍风格和质量的新闻稿
4. 假设已经有一个evaluate_press_agent，它的输入是generate_press_agent生成的新闻稿 + 这个采访稿对应的黄继妍本人撰写的新闻稿groundtruth，以及一个固定的llm打分prompt(prompt_evaluate_press)。 

## 数据
1.data/train.jsonl
27条数据，每条数据包含 interview_context, interview_type, groundtruth. 
iterview_context: 采访稿内容
groundtruth: 黄继妍本人撰写的新闻稿,和 interview_context是一对，一个 interview_context 对应一个 groundtruth

train.jsonl 用于给openevolve优化 prompt_generate_press_base, openevolve的每一轮都对train.jsonl中的所有数据进行推理、评估、evaluator生成combinedscore (所有训练数据上得分的平均值)

2.data/test.jsonl
格式和train.jsonl一样，一共有7条数据.
这个数据完全不参与openevolve的训练，但是openevolve每一轮得到新的prompt后，会用这个prompt去生成test.jsonl中的数据，并评估，得到test.jsonl的test_combinedscore. 这个test_combinedsscore不返回给openevolve，只是每一轮展示出来，方便查看泛化性

# 需要你构造的内容
## mle-openevolve/experiments/press01/inputs/source/initial_program.py
主要内容就是初始的prompt_generate_press，以及一个供evaluator.py加载和调用的入口函数

## mle-openevolve/experiments/press01/inputs/source/generate_press_agent.py
这个是固定的 generate_press_agent

## mle-openevolve/experiments/press01/inputs/source/evaluate_press_agent.py
这个是固定的 evaluate_press_agent

## mle-openevolve/experiments/press01/inputs/source/evaluator.py
这个脚本包含的内容较多，每一轮openevolve生成新的child program，都会使用这个脚本的，完成新闻稿生成和评估，并返回combinedscore给openevolve进化提供方向. 可能的流程包括：加载 initial / child program并运行其中的入口函数得到prompt_generate_press_base, 加载训练数据，构造最终的prompt_generate_press，并发调用generate_press_agent生成新闻稿(这个agent是从外部的generate_press_agent.py加载)，再调用evaluate_press_agent (这个也是外部加载)，得到原始得分结果. 然后再使用合适的方法计算出combinedscore用于进化。除了训练数据外，还要加载测试数据计算原始得分和test_combinedscore(测试集结果仅用于展示不返回给openevolve)

我在上述4个脚本中写了一些你会用的到的伪代码信息，你可以参考。

## mle-openevolve/experiments/press01/inputs/source/config_evolve.yaml
核心可能是需要修改里边的prompt: system_message,设计一个适合我们这个任务的.这个对openevolve进化应该也挺重要的

## mle-openevolve/experiments/press01/inputs/config.json
参考bean01的config.json,这个主要是和前后端对接需要

## mle-openevolve/main_press.py
参考main_bean.py的封装


构造过程的high level逻辑思路 可以参考mle-openevolve/experiments/bean01 以及 main_bean.py