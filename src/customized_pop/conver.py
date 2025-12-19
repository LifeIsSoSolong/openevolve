import srsly

for split in ["train", "test"]:
    data = srsly.read_json(
        f"/hpc_data/zhangkaiyan/agentic-rl/src/customized_pop/data/{split}.json")

    outputs = []
    for idx, sample in enumerate(data):
        outputs.append({
            "id": sample["id"],
            "messages": [{"role": "user", "content": sample["prompt"]}],
            "ground_truth": sample["answer"]})
    print(len(outputs))
    srsly.write_jsonl(f"./data/{split}.jsonl", outputs)
