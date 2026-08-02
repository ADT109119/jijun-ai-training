import json, os, random, shutil, sys

def split_file(src, dst_dir="dataset", seed=42):
    os.makedirs(dst_dir, exist_ok=True)
    samples = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_data = samples[:split_idx]
    test_data = samples[split_idx:]

    train_path = os.path.join(dst_dir, "train_strata.jsonl")
    test_path = os.path.join(dst_dir, "test_strata.jsonl")
    for p in (train_path, test_path):
        if os.path.exists(p):
            shutil.copyfile(p, p + ".bak")
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        for s in test_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Split done: total={len(samples)}, train={len(train_data)}, test={len(test_data)}")
    print(f"  train -> {train_path}")
    print(f"  test  -> {test_path}")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "dataset/qwen_full/raw_filtered.jsonl"
    split_file(src)
