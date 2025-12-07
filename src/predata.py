import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import chardet

INPUT_CSV = "data.csv"
OUTPUT_DIR = "data"

def detect_encoding(path, sample_size=2000000):
    with open(path, "rb") as f:
        raw = f.read(sample_size)
    info = chardet.detect(raw)
    enc = info.get("encoding", None)
    print(f"[predata] chardet 检测到编码: {enc}, 置信度: {info.get('confidence')}")
    return enc or "utf-8"

def row_to_record(row):
    department = str(row["department"]).strip()
    title = str(row["title"]).strip()
    ask = str(row["ask"]).strip()
    answer = str(row["answer"]).strip()

    instruction = (
        f"你是一名专业的儿童医生,请用通俗和耐心的语言回答家长的问题。"
    )

    return {
        "instruction": instruction,
        "input": title,
        "output": answer,
    }

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    enc = detect_encoding(INPUT_CSV)

    try:
        df = pd.read_csv(INPUT_CSV, encoding=enc)
    except UnicodeDecodeError as e:
        df = pd.read_csv(
            INPUT_CSV,
            encoding="gb18030",
            encoding_errors="ignore",
        )

    required_cols = {"department", "title", "ask", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 列名不匹配，当前列: {df.columns.tolist()}")

    df = df.dropna(subset=["ask", "answer"])

    records = [row_to_record(row) for _, row in df.iterrows()]

    train_records, test_records = train_test_split(
        records, test_size=0.1, random_state=42
    )
    save_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train_records)
    save_jsonl(os.path.join(OUTPUT_DIR, "test.jsonl"), test_records)
    print("[predata] Done!")
    print("[predata] train 样本数:", len(train_records))
    print("[predata] test  样本数:", len(test_records))

if __name__ == "__main__":
    main()
