from collections import defaultdict
import regex as re


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    start = time.time()
    vocab = {}
    merges = []
    
    for i in range(256):
        bt = bytes([i])
        vocab[i] = bt
    for tok in special_tokens:
        # special_token 不可分割
        vocab[len(vocab)] = tok.encode("utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    ids = []
    if special_tokens:
        toks = sorted(special_tokens, key=len, reverse=True)
        union = "|".join(re.escape(t) for t in toks)
        parts = re.split(f"({union})", text)

        st = set(special_tokens)
        for part in parts:
            if not part or part in st:
                continue
            ids.extend(re.findall(PAT, part))
    else:
        ids = re.findall(PAT, text)
        
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    # pair_to_indice 存储的是 pair 位于 subwords的[i] ,set[i]
    pair_to_indice: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    
    ids = [list(subword.encode("utf-8")) for subword in ids]
    print(f"读取文件花费 {time.time() - start}s")
    start = time.time()
    
    for i, token_ids in enumerate(ids):
        for j in range(len(token_ids)-1):
            pair = (token_ids[j], token_ids[j+1])
            pair_counts[pair] += 1
            pair_to_indice[pair].add(i)
    
    print(f"初始化花费 {time.time() - start}s")
    start = time.time()
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        
        best_pair = max(pair_counts.items(), key=lambda x:(x[1], (vocab[x[0][0]], vocab[x[0][1]])))[0]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        related_indices = pair_to_indice[best_pair].copy()
        for i in related_indices:
            token_ids = ids[i]
            if len(token_ids) < 2:
                continue
            
            for old_pair in zip(token_ids[:-1], token_ids[1:]):
                pair_counts[old_pair] -= 1
                pair_to_indice[old_pair].discard(i)
            
            j = 0
            new_token_ids = []
            while j < len(token_ids):
                if j < len(token_ids) - 1 and (token_ids[j], token_ids[j+1]) == best_pair:
                    new_token_ids.append(new_token_id)
                    j += 2
                else:
                    new_token_ids.append(token_ids[j])
                    j += 1
            ids[i] = new_token_ids
            
            for new_pair in zip(new_token_ids[:-1], new_token_ids[1:]):
                pair_counts[new_pair] += 1
                pair_to_indice[new_pair].add(i)
    print(f"merges 花费 {time.time() - start}s")
    return vocab, merges

if __name__ == "__main__":
    import time
    start = time.time()
    INPUT_PATH = r"D:\dev\github\cs336\assignment1\data\TinyStoriesV2-GPT4-train.txt"
    # INPUT_PATH = r"D:\dev\github\cs336\assignment1\tests\fixtures\corpus.en"
    VOCAB_SIZE = 300
    SPECIAL_TOKENS = ["<|endoftext|>"]
    train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)
    print(f"cost {time.time() - start}s")