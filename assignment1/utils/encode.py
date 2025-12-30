from bpe import BPE_Tokenizer
from pathlib import Path
import numpy as np 
import pickle


def encode_from_file(
    tokenizer: BPE_Tokenizer,
    in_filepath: str,
    out_filepath: str,
    chunk_size: int = 1024 * 1024
):
    in_path = Path(in_filepath)
    out_path = Path(out_filepath)
    assert in_path.exists()
    
    if not out_path.exists():
        with open(out_path, "wb"):
            pass
    
    buffer = []

    
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "ab") as fout:
        for i, line in enumerate(fin):
            ids = tokenizer.encode(line)
            buffer.extend(ids)
            
            if len(buffer) >= chunk_size:
                np.asarray(buffer, dtype=np.uint16).tofile(fout)
                buffer.clear()

            if i % 100_000 == 0:
                print("processed", i)
        
        
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(fout)

if __name__ == "__main__":
    with open(r"D:\dev\github\cs336\assignment1\data\vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(r"D:\dev\github\cs336\assignment1\data\merges.pkl", "rb") as f:
        merges = pickle.load(f)
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPE_Tokenizer(vocab, merges, special_tokens)
    encode_from_file(
        tokenizer,
        r"D:\dev\github\cs336\assignment1\data\TinyStoriesV2-GPT4-train.txt",
        "train_ids.bin"
    )